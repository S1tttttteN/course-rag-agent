"""Natural/web mode on the real LangChain graph, without network or paid calls."""
from contextlib import closing
import json
import sqlite3
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from sqlalchemy import select

from agent_fakes import FailingModel, PausedModel, ScriptedModel, StreamingModel, call
import test_conversations as shared
from app.repositories.conversations import completed_turns
from app.core.config import CHECKPOINT_PATH
from app.db.models import CleanupJob, Run
from app.db.session import SessionLocal
from app.jobs.indexing import cleanup_job
from app.services import runs


URL = 'https://example.org/course'
SNIPPET = {'url': URL, 'title': '公开教学样例', 'text': '课程外实验采用三次观察。', 'kind': 'web_snippet'}
PAGE = {**SNIPPET, 'text': '完整教学说明：连续观察三次后比较结果。', 'kind': 'web_page'}


class SummaryModel(ScriptedModel):
    """在真实摘要中间件调用时提供确定的摘要并记录输入。"""

    summary_inputs: list[str] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """区分摘要请求与用户问答，便于断言早期约定确实进入摘要。"""
        if '中文会话摘要' in messages[-1].content:
            prompt = messages[-1].content
            self.summary_inputs.append(prompt)
            label = '方案 A 指多级反馈队列' if '方案 A 指多级反馈队列' in prompt else '缺少方案 A 的定义'
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='摘要：' + label))])
        return super()._generate(messages, stop, run_manager, **kwargs)


class NaturalAgentTests(unittest.TestCase):
    """验证流式问答、会话记忆与可选网络来源。"""

    publish = shared.ConversationTests.publish
    ask = shared.ConversationTests.ask
    wait = shared.ConversationTests.wait
    tearDown = shared.ConversationTests.tearDown

    def setUp(self):
        """安装网络工具替身并创建对话测试数据。"""
        # Unexpected network/cloud activity is a hard test failure.
        self.web_search = self.enterContext(patch('app.qa.tools.web_search', side_effect=AssertionError('unexpected web search')))
        self.web_read = self.enterContext(patch('app.qa.tools.web_read', side_effect=AssertionError('unexpected web read')))
        shared.ConversationTests.setUp(self)

    def test_empty_course_can_answer_general_questions(self):
        """验证空课程下的通用问题可以直接流式回答。"""
        course = self.client.post(f'/api/courses/{self.other}/conversations', json={}).json()['id']
        self.path = f'/api/courses/{self.other}/conversations/{course}'
        model = StreamingModel(responses=[AIMessage(content='可以这样理解：用一个生活中的例子说明概念。')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(answer['content'], model.responses[0].content)
        self.assertEqual(answer['citations'], [])
        self.assertFalse(answer['run']['web_search'])
        self.web_search.assert_not_called()

    def test_summary_restores_after_restart(self):
        """验证首轮约定经过真实摘要后可在重启后续接。"""
        model = SummaryModel(responses=[AIMessage(content='已经记录。' + '复习内容。' * 120)])
        definition = '方案 A 指多级反馈队列'
        with patch('app.qa.agent.get_llm', return_value=model):
            for number in range(4):
                question = '请记录：' + definition if number == 0 else f'第 {number} 轮复习'
                answer = self.wait(self.ask(question + '上下文。' * 135))
                self.assertEqual(answer['status'], 'completed')
            fifth = self.wait(self.ask('方案 A 是什么？'))
            self.assertEqual(fifth['status'], 'completed')
        self.assertTrue(any(definition in prompt for prompt in model.summary_inputs))
        self.assertIn('摘要：' + definition, '\n'.join(item.content for item in model.seen[-1]))
        self.assertFalse(any(item.content.startswith('请记录：') for item in model.seen[-1]))
        self.assertGreaterEqual(fifth['run']['trace']['usage']['model_calls'], 2)
        self.assertEqual(fifth['run']['trace']['usage']['summary_calls'], 1)
        self.assertNotIn('摘要：', fifth['content'])
        events = self.client.get(f"{self.path}/runs/{fifth['run_id']}/events").text
        self.assertNotIn('摘要：', events)
        with SessionLocal() as session:
            self.assertIsNotNone(session.get(Run, fifth['run_id']).agent_checkpoint_id)

        self.context.__exit__(None, None, None)
        self.context = TestClient(shared.app)
        self.client = self.context.__enter__()
        resumed = SummaryModel(responses=[AIMessage(content='接着复习。')])
        with patch('app.qa.agent.get_llm', return_value=resumed):
            self.assertEqual(self.wait(self.ask('继续接着讲'))['status'], 'completed')
        context = '\n'.join(item.content for item in resumed.seen[-1])
        self.assertIn(definition, context)
        self.assertEqual(context.count('方案 A 是什么？'), 1)
        isolated_id = self.client.post(self.base, json={}).json()['id']
        self.path = f'{self.base}/{isolated_id}'
        with patch('app.qa.agent.get_llm', return_value=resumed):
            self.assertEqual(self.wait(self.ask('开始新对话'))['status'], 'completed')
        self.assertNotIn(definition, '\n'.join(item.content for item in resumed.seen[-1]))

    def test_summary_failure_keeps_previous_checkpoint(self):
        """验证摘要失败会终止本轮，下一轮仍从上次成功的检查点开始。"""
        model = SummaryModel(responses=[AIMessage(content='已记录。' + '复习内容。' * 750)])
        with patch('app.qa.agent.get_llm', return_value=model):
            original = self.wait(self.ask('保留上一轮。' + '上下文。' * 950))
            self.assertEqual(original['status'], 'completed')
            with patch('app.qa.history.SummarizationMiddleware.abefore_model', side_effect=RuntimeError):
                dropped = self.wait(self.ask('摘要调用失败的提问'))
            self.assertEqual(dropped['status'], 'failed')
            with SessionLocal() as session:
                self.assertIsNone(session.get(Run, dropped['run_id']).agent_checkpoint_id)
        followup = SummaryModel(responses=[AIMessage(content='继续。')])
        with patch('app.qa.agent.get_llm', return_value=followup):
            self.assertEqual(self.wait(self.ask('重新提问'))['status'], 'completed')
        self.assertNotIn('摘要调用失败的提问', '\n'.join(item.content for item in followup.seen[-1]))

    def test_failed_branch_does_not_enter_next_turn(self):
        """验证生成失败的会话分支和未提交的检查点不会进入下一轮。"""
        model = ScriptedModel(responses=[AIMessage(content='首轮正式回答')])
        with patch('app.qa.agent.get_llm', return_value=model):
            self.assertEqual(self.wait(self.ask('有效输入'))['status'], 'completed')
        with patch('app.qa.agent.get_llm', return_value=FailingModel(responses=[])):
            self.assertEqual(self.wait(self.ask('失败分支输入'))['status'], 'failed')
        with patch('app.qa.agent.get_llm', return_value=model):
            self.assertEqual(self.wait(self.ask('继续上一轮'))['status'], 'completed')
        contents = [item.content for item in model.seen[-1]]
        self.assertIn('首轮正式回答', contents)
        self.assertNotIn('失败分支输入', contents)
        self.assertNotIn('已生成的部分。', contents)

    def test_checkpoint_written_before_failed_commit_is_ignored(self):
        """模拟检查点已落盘但业务终态提交失败，验证下一轮仍使用上次成功的历史。"""
        model = ScriptedModel(responses=[AIMessage(content='正式回答')])
        finish = runs.finish

        def fail_commit(run_id, state, error='', *, agent_checkpoint_id=None):
            """在新检查点提交给业务库前模拟异常。"""
            if state == 'completed' and agent_checkpoint_id:
                raise RuntimeError('模拟检查点已写入后的业务提交失败')
            return finish(run_id, state, error, agent_checkpoint_id=agent_checkpoint_id)

        with patch('app.qa.agent.get_llm', return_value=model):
            self.assertEqual(self.wait(self.ask('应保留的问题'))['status'], 'completed')
            with patch('app.qa.manager.runs.finish', side_effect=fail_commit):
                dropped = self.wait(self.ask('未提交的问题'))
            self.assertEqual(dropped['status'], 'failed')
            self.assertEqual(self.wait(self.ask('继续前一个问题'))['status'], 'completed')
        contents = [item.content for item in model.seen[-1]]
        self.assertIn('应保留的问题', contents)
        self.assertNotIn('未提交的问题', contents)

    def test_conversation_and_course_deletion_cleanup_checkpoints(self):
        """验证会话删除与课程删除都会登记并回收各自的检查点。"""
        model = ScriptedModel(responses=[AIMessage(content='已保存')])
        with patch('app.qa.agent.get_llm', return_value=model):
            self.assertEqual(self.wait(self.ask())['status'], 'completed')
            first = self.conversation
            self.assertEqual(self.client.delete(self.path).status_code, 204)
            self.conversation = self.client.post(self.base, json={}).json()['id']
            self.path = f'{self.base}/{self.conversation}'
            self.assertEqual(self.wait(self.ask())['status'], 'completed')
        self.assertEqual(self.client.delete(f'/api/courses/{self.course}').status_code, 204)
        with SessionLocal() as session:
            jobs = dict(session.execute(
                select(CleanupJob.conversation_id, CleanupJob.id).where(CleanupJob.course_id == self.course)
            ).all())
        with closing(sqlite3.connect(CHECKPOINT_PATH)) as connection:
            for identifier in (first, self.conversation):
                query = 'SELECT count(*) FROM checkpoints WHERE thread_id = ?'
                self.assertGreater(connection.execute(query, (identifier,)).fetchone()[0], 0)
                cleanup_job(jobs[identifier])
                self.assertEqual(connection.execute(query, (identifier,)).fetchone()[0], 0)

    def test_course_then_web_sources_are_numbered_deduplicated_and_persisted(self):
        """验证课程与网络来源共享编号空间、重复来源去重且引用持久化。"""
        self.web_search.side_effect = None
        self.web_search.return_value = [SNIPPET]
        self.web_read.side_effect = None
        self.web_read.return_value = PAGE
        text = '课程资料：进程分配资源。[1]\n\n网络补充：连续观察三次。[2][3]'
        model = StreamingModel(responses=[call('search_course', {'query': '进程'}),
            call('search_web', {'query': '公开教学实验'}), call('search_web', {'query': '公开教学实验'}),
            call('read_webpage', {'url': URL}), AIMessage(content=text)])
        with patch('app.qa.agent.get_llm', return_value=model):
            run_id = self.ask('结合当前课程资料与公开网页讲解', web_search=True)
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(answer['content'], text)  # Streaming must not duplicate final content.
        self.assertEqual([item['number'] for item in answer['citations']], [1, 2, 3])
        self.assertEqual([item['kind'] for item in answer['citations']], ['course', 'web_snippet', 'web_page'])
        self.assertEqual(len(answer['run']['trace']['tools']), 4)
        self.assertTrue(answer['run']['web_search'])
        citation = answer['citations'][-1]
        self.assertTrue(citation['retrieved_at'])
        stream = self.client.get(f'{self.path}/runs/{run_id}/events').text
        self.assertGreater(stream.count('event: delta'), 1)
        self.assertIn('event: done', stream)
        snapshot = self.client.get(f"{self.path}/citations/{citation['id']}").json()
        self.assertEqual(snapshot['url'], URL)
        self.context.__exit__(None, None, None)
        self.context = shared.TestClient(shared.app)
        self.client = self.context.__enter__()
        restored = self.client.get(self.path).json()['messages'][-1]
        self.assertEqual(restored['content'], text)
        self.assertEqual(restored['citations'], answer['citations'])
        self.assertEqual(model.cursor, 5)

    def test_web_failure_is_recorded_without_exception_details_and_answer_can_continue(self):
        """验证网络工具失败时记录通用状态，并允许模型继续生成回答。"""
        self.web_search.side_effect = RuntimeError('private-search-error-fixture')
        model = StreamingModel(responses=[call('search_web', {'query': '公开知识'}),
            AIMessage(content='网络暂不可用；以下是一般知识补充，尚未核对网页。')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask(web_search=True))
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(answer['run']['trace']['tools'][0]['status'], 'failed')
        self.assertEqual(answer['citations'], [])
        self.assertNotIn('private-search-error-fixture', json.dumps(answer))
        self.assertNotIn('private-search-error-fixture', str(model.seen))

    def test_read_requires_this_run_search_and_foreign_course_is_rejected(self):
        """验证网页阅读要求本轮搜索记录，课程阅读拒绝其他课程的资料。"""
        foreign = self.client.post(f'/api/courses/{self.other}/documents', files={'file': ('另一课.md', '隔离样例正文'.encode())}).json()
        self.publish(foreign)
        model = StreamingModel(responses=[call('read_webpage', {'url': URL}),
            call('read_document', {'document_id': foreign['id'], 'version_id': foreign['latest']['id']}),
            AIMessage(content='未读取到所需资料。')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask(web_search=True))
        self.assertEqual(answer['status'], 'completed')
        self.web_read.assert_not_called()
        self.assertTrue(all(row['status'] == 'failed' for row in answer['run']['trace']['tools']))
        self.assertNotIn('隔离样例正文', str(model.seen))
        self.assertEqual(answer['citations'], [])

    def test_agent_does_not_carry_web_permission_between_questions(self):
        """上一轮允许联网时，下一轮省略选项仍不提供网络工具。"""
        enabled = StreamingModel(responses=[AIMessage(content='第一轮回答。')])
        disabled = StreamingModel(responses=[AIMessage(content='第二轮回答。')])
        with patch('app.qa.agent.get_llm', side_effect=[enabled, disabled]), patch.object(
            StreamingModel, 'bind_tools', autospec=True, side_effect=StreamingModel.bind_tools
        ) as bind_tools:
            first = self.wait(self.ask(web_search=True))
            second = self.wait(self.ask())
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(second['status'], 'completed')
        self.assertTrue(first['run']['web_search'])
        self.assertFalse(second['run']['web_search'])
        first_tools, second_tools = ({tool.name for tool in call.args[1]} for call in bind_tools.call_args_list)
        self.assertEqual(first_tools - second_tools, {'search_web', 'read_webpage'})
        self.assertEqual(second_tools, {'search_course', 'list_documents', 'read_document'})

    def test_cancel_preserves_streamed_text_and_excludes_incomplete_history(self):
        """验证取消保留已发布正文，并从历史上下文排除未完成问答。"""
        with patch('app.qa.agent.get_llm', return_value=PausedModel(responses=[])):
            run_id = self.ask()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                partial = self.client.get(self.path).json()['messages'][-1]
                if partial['content']:
                    break
                time.sleep(.02)
            self.assertEqual(partial['content'], '已生成的自然回答。')
            self.client.post(f'{self.path}/runs/{run_id}/cancel')
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'cancelled')
        self.assertEqual(answer['content'], partial['content'])
        with SessionLocal() as session:
            self.assertEqual(list(completed_turns(session, self.conversation, 'next')), [])

    def test_failed_stream_preserves_partial_text_without_leaking_provider_error(self):
        """验证流式输出失败时保留已发布正文，并隐藏服务商异常细节。"""
        with patch('app.qa.agent.get_llm', return_value=FailingModel(responses=[])):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'failed')
        self.assertEqual(answer['content'], '已生成的部分。')
        self.assertNotIn('private-provider-error-fixture', json.dumps(answer))

    def test_budget_limits_web_calls_and_finalizes_without_more_tools(self):
        """验证网络工具受统一调用预算约束，并在预算耗尽后结束本轮。"""
        self.web_search.side_effect = None
        self.web_search.return_value = [SNIPPET]
        model = StreamingModel(responses=[call('search_web', {'query': '公开实验'}),
            call('search_web', {'query': '公开实验'}), AIMessage(content='根据已有资料总结。[1]')])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.tool_middleware.MAX_CALLS', 1):
            answer = self.wait(self.ask(web_search=True))
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(self.web_search.call_count, 1)
        self.assertEqual(len(answer['run']['trace']['tools']), 1)
        self.assertIn('查阅上限', answer['content'])
        self.assertIn('根据已有资料总结', answer['content'])
