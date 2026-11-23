"""Real LangChain graph and real application lifecycle, with scripted provider responses."""
import asyncio
import json
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent_fakes import ScriptedModel, call
import test_conversations as shared
from app.db.models import Run
from app.db.session import SessionLocal
from app.repositories.conversations import completed_turns
from app.qa.tools import list_documents, read_document, read_webpage, search_course, search_web
from app.services import conversations, runs


class ToolSchemaTests(unittest.TestCase):
    """校验模型可见的工具参数。"""

    def test_tool_schema_keeps_runtime_and_course_scope_internal(self):
        """模型只能看到各工具的业务参数。"""
        expected = {
            "search_course": {"query"},
            "list_documents": {"offset"},
            "read_document": {"document_id", "version_id", "offset", "limit"},
            "search_web": {"query"},
            "read_webpage": {"url"},
        }
        tools = [search_course, list_documents, read_document, search_web, read_webpage]
        for tool in tools:
            with self.subTest(tool=tool.name):
                schema = tool.tool_call_schema.model_json_schema()
                self.assertEqual(set(schema["properties"]), expected[tool.name])


class AgentTests(unittest.TestCase):
    """验证 Agent 工具调用和问答生命周期。"""

    setUp = shared.ConversationTests.setUp
    tearDown = shared.ConversationTests.tearDown
    publish = shared.ConversationTests.publish
    ask = shared.ConversationTests.ask
    wait = shared.ConversationTests.wait

    def test_multiple_searches_deduplicate_citations_and_restore_records(self):
        """验证多次搜索合并重复引用，并能从历史中读取工具和引用记录。"""
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'}),
            call('search_course', {'query': '线程'}, 'call2'), AIMessage(content='进程分配资源，线程负责调度。[1]')])
        with patch('app.qa.agent.get_llm', return_value=model):
            run_id = self.ask()
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual([item['number'] for item in answer['citations']], [1])
        self.assertEqual(len(answer['run']['trace']['tools']), 2)
        self.assertTrue(all(item['status'] == 'completed' for item in answer['run']['trace']['tools']))
        self.assertTrue(all(item['retrieval']['keyword'] for item in answer['run']['trace']['tools']))
        events = self.client.get(f'{self.path}/runs/{run_id}/events').text
        self.assertEqual(events.count('event: tool'), 4)
        self.assertNotIn('event: tool', self.client.get(f"{self.path}/runs/{run_id}/events?after={answer['run']['last_event_id']}").text)
        self.context.__exit__(None, None, None)
        self.context = shared.TestClient(shared.app)
        self.client = self.context.__enter__()
        restored = self.client.get(self.path).json()['messages'][-1]
        self.assertEqual(restored['run']['trace']['tools'], answer['run']['trace']['tools'])
        self.assertEqual(model.cursor, 3)  # Restart never replays a model call.

    def test_list_and_paginated_read_share_registry(self):
        """验证目录分页读取与资料检索使用同一引用登记表。"""
        document = self.client.post(f'/api/courses/{self.course}/documents', files={
            'file': ('长文.md', ('第一节进程。' * 160 + '\n\n' + '最后一节线程。' * 160).encode())}).json()
        self.publish(document)
        args = {'document_id': document['id'], 'version_id': document['latest']['id'], 'limit': 1}
        model = ScriptedModel(responses=[call('list_documents', {}), call('read_document', {**args, 'offset': 0}, 'r1'),
            call('read_document', {**args, 'offset': 1}, 'r2'), AIMessage(content='这份资料介绍进程和线程。[1][2]')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual([item['number'] for item in answer['citations']], [1, 2])
        self.assertEqual(len(answer['run']['trace']['tools']), 3)
        output = [item for item in model.seen[-1] if item.type == 'tool']
        self.assertEqual(json.loads(output[1].content)['next_offset'], 1)

    def test_foreign_document_and_extra_scope_arguments_fail_closed(self):
        """验证工具拒绝其他课程的资料和额外作用域参数。"""
        foreign = self.client.post(f'/api/courses/{self.other}/documents', files={'file': ('隔离.md', '绝密口令不应出现'.encode())}).json()
        self.publish(foreign)
        model = ScriptedModel(responses=[call('read_document', {'document_id': foreign['id'], 'version_id': foreign['latest']['id']}),
            call('search_course', {'query': '进程', 'course_id': self.other}, 'inject'), AIMessage(content='本轮资料未找到足够依据。')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['citations'], [])
        self.assertIn('未找到足够依据', answer['content'])
        self.assertNotIn('绝密口令', str(model.seen))
        self.assertTrue(all(item['status'] == 'failed' for item in answer['run']['trace']['tools']))

    def test_budget_stops_repeated_invalid_calls(self):
        """验证反复提交无效工具参数会耗尽调用预算。"""
        model = ScriptedModel(responses=[
            call('search_course', {'query': '进程', 'invalid': True}),
            call('search_course', {'query': '进程', 'invalid': True}),
            call('search_course', {'query': '进程', 'invalid': True}),
            AIMessage(content='本轮未找到足够依据。'),
        ])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.tool_middleware.MAX_CALLS', 2):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(len(answer['run']['trace']['tools']), 2)
        self.assertEqual(model.cursor, 4)
        self.assertIn('未找到足够依据', answer['content'])

    def test_tool_timeout_is_recorded_without_provider_error_details(self):
        """验证工具超时会记录状态和通用提示。"""
        async def delayed(*args):
            """模拟执行时间超过工具时限的检索。

            Args:
                args: 检索调用传入的位置参数。
            """
            await asyncio.sleep(30)
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'}), AIMessage(content='没有依据')])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.manager.RunManager.retrieve', side_effect=delayed), patch('app.qa.tool_middleware.TOOL_TIMEOUT', .02):
            answer = self.wait(self.ask())
        tool = answer['run']['trace']['tools'][0]
        self.assertEqual(tool['status'], 'failed')
        self.assertIn('超时', tool['summary'])

    def test_cancel_retains_sources_and_excludes_incomplete_history(self):
        """验证取消保留已保存的引用，并将未完成问答排除出历史上下文。"""
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'})], wait_after=1)
        with patch('app.qa.agent.get_llm', return_value=model):
            run_id = self.ask()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                detail = self.client.get(self.path).json()
                if detail['messages'][-1]['citations']:
                    break
                time.sleep(.03)
            self.assertTrue(detail['messages'][-1]['citations'])
            self.client.post(f'{self.path}/runs/{run_id}/cancel')
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'cancelled')
        self.assertTrue(answer['citations'])
        with SessionLocal() as session:
            self.assertEqual(list(completed_turns(session, self.conversation, 'next')), [])

    def test_recovery_marks_unfinished_tool_interrupted_without_replay(self):
        """验证重启恢复会标记未完成工具调用为中断。"""
        with SessionLocal() as session:
            run = conversations.create_run(session, self.course, self.conversation, '重启验证')
        runs.tool_update(run['id'], {'id': 'unfinished', 'status': 'running', 'name': 'search_course'})
        self.context.__exit__(None, None, None)
        with patch('app.qa.agent.get_llm', side_effect=AssertionError('must not replay')):
            self.context = shared.TestClient(shared.app)
            self.client = self.context.__enter__()
            answer = self.wait(run['id'])
        self.assertEqual(answer['status'], 'interrupted')
        self.assertEqual(answer['run']['trace']['tools'][0]['status'], 'interrupted')

    def test_direct_answer_reports_model_usage(self):
        """验证无需查阅的回答能直接发布并记录模型用量。"""
        model = ScriptedModel(responses=[AIMessage(content='不受资料支持的解释', usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['content'], '不受资料支持的解释')
        self.assertEqual(answer['citations'], [])
        self.assertEqual(answer['run']['trace']['usage']['input_tokens'], 10)
        self.assertEqual(answer['run']['trace']['usage']['model_calls'], 1)
        self.assertGreaterEqual(answer['run']['trace']['elapsed_seconds'], 0)

    def test_removed_question_options_are_rejected(self):
        """问答请求仅接受问题和当轮联网许可。"""
        for field, value in {
            'mode': 'agent', 'semantic': True, 'rerank': True, 'top_k': 4, 'answer_review': False,
        }.items():
            with self.subTest(field=field):
                response = self.client.post(f'{self.path}/runs', json={'question': '测试', field: value})
                self.assertEqual(response.status_code, 422)
        config = self.client.get('/api/chat/config').json()
        self.assertEqual(set(config), {'provider', 'model', 'configured', 'note'})

    def test_budget_with_evidence_finishes_without_more_tools(self):
        """验证达到预算后能根据已取得的证据完成回答。"""
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'}),
            call('search_course', {'query': '线程'}, 'second'), AIMessage(content='进程分配资源。[1]')])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.tool_middleware.MAX_CALLS', 1):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(len(answer['run']['trace']['tools']), 1)
        self.assertIn('查阅上限', answer['content'])
        self.assertIn('进程分配资源', answer['content'])
