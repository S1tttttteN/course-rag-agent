"""Natural/web mode on the real LangChain graph, without network or paid calls."""
import asyncio
import json
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from test_agent import ScriptedModel, call
import test_conversations as shared
from app.qa import agent
from app.repositories.conversations import completed_turns
from app.db.session import SessionLocal


URL = 'https://example.org/course'
SNIPPET = {'url': URL, 'title': '公开教学样例', 'text': '课程外实验采用三次观察。', 'kind': 'web_snippet'}
PAGE = {**SNIPPET, 'text': '完整教学说明：连续观察三次后比较结果。', 'kind': 'web_page'}


class StreamingModel(ScriptedModel):
    """将预设响应转换为测试用流式模型输出。"""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """将预设响应作为工具调用或正文流发送。

        Args:
            messages: 传给模型的消息。
            stop: 框架提供的可选停止词。
            run_manager: 框架提供的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Yields:
            工具调用块或分段的正文块。
        """
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        if message.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[
                {'name': item['name'], 'args': json.dumps(item['args']), 'id': item['id'], 'index': index}
                for index, item in enumerate(message.tool_calls)]))
        else:
            for position in range(0, len(message.content), 5):
                yield ChatGenerationChunk(message=AIMessageChunk(content=message.content[position:position + 5]))
                await asyncio.sleep(.01)


class PausedModel(StreamingModel):
    """模拟发送首段正文后停顿的模型。"""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """发出一个正文片段后等待，用于测试运行取消。

        Args:
            messages: 流式接口收到的输入消息。
            stop: 框架提供的可选停止词。
            run_manager: 框架提供的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Yields:
            等待前发出的正文片段。
        """
        yield ChatGenerationChunk(message=AIMessageChunk(content='已生成的自然回答。'))
        await asyncio.sleep(30)


class FailingModel(StreamingModel):
    """模拟发布部分正文后失败的模型。"""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """发出正文片段后触发模拟服务商错误。

        Args:
            messages: 流式接口收到的输入消息。
            stop: 框架提供的可选停止词。
            run_manager: 框架提供的可选异步回调管理器。
            kwargs: 调用方附加的关键字参数。

        Yields:
            失败前发出的正文片段。

        Raises:
            RuntimeError: 用于验证服务商错误处理的模拟异常。
        """
        yield ChatGenerationChunk(message=AIMessageChunk(content='已生成的部分。'))
        raise RuntimeError('private-provider-error-fixture')


class NaturalAgentTests(unittest.TestCase):
    """验证关闭严格核对时的流式问答与可选网络来源。"""

    publish = shared.ConversationTests.publish
    wait = shared.ConversationTests.wait
    tearDown = shared.ConversationTests.tearDown

    def setUp(self):
        """关闭严格核对，安装网络工具替身，并创建对话测试数据。"""
        for target, value in (
            ('app.core.models.ANSWER_REVIEW', False), ('app.qa.manager.ANSWER_REVIEW', False),
            ('app.qa.agent.ANSWER_REVIEW', False), ('app.qa.agent.WEB_SEARCH_ENABLED', True),
        ):
            self.enterContext(patch(target, value))
        # Unexpected network/cloud activity is a hard test failure.
        self.web_search = self.enterContext(patch('app.qa.agent.web_search', side_effect=AssertionError('unexpected web search')))
        self.web_read = self.enterContext(patch('app.qa.agent.web_read', side_effect=AssertionError('unexpected web read')))
        self.review = self.enterContext(patch('app.qa.agent.publish_checked', side_effect=AssertionError('unexpected strict reviewer')))
        shared.ConversationTests.setUp(self)

    def ask(self, question='请解释一个概念', path=None, mode='agent'):
        """向指定测试对话提交所选模式的问答。

        Args:
            question: 用户提交的问题正文。
            path: 可选对话路径，省略时使用当前测试对话。
            mode: 问答模式，取值为 rag 或 agent。

        Returns:
            已接受的运行 ID。
        """
        response = self.client.post(f'{path or self.path}/runs', json={
            'question': question, 'mode': mode, 'semantic': False, 'rerank': False})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def test_empty_course_can_answer_without_retrieval_or_strict_review(self):
        """验证关闭严格审查时空课程可以直接流式回答。"""
        course = self.client.post(f'/api/courses/{self.other}/conversations', json={}).json()['id']
        self.path = f'/api/courses/{self.other}/conversations/{course}'
        model = StreamingModel(responses=[AIMessage(content='可以这样理解：用一个生活中的例子说明概念。')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(answer['content'], model.responses[0].content)
        self.assertEqual(answer['citations'], [])
        self.assertFalse(answer['run']['trace']['answer_review'])
        self.assertTrue(answer['run']['trace']['web_search'])
        self.review.assert_not_called()
        self.web_search.assert_not_called()

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
            run_id = self.ask('结合当前课程资料与公开网页讲解')
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(answer['content'], text)  # Streaming must not duplicate final content.
        self.assertEqual([item['number'] for item in answer['citations']], [1, 2, 3])
        self.assertEqual([item['kind'] for item in answer['citations']], ['course', 'web_snippet', 'web_page'])
        self.assertEqual(len(answer['run']['trace']['tools']), 4)
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
            answer = self.wait(self.ask())
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
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.web_read.assert_not_called()
        self.assertTrue(all(row['status'] == 'failed' for row in answer['run']['trace']['tools']))
        self.assertNotIn('隔离样例正文', str(model.seen))
        self.assertEqual(answer['citations'], [])

    def test_disabled_web_tools_are_not_bound(self):
        """验证关闭网络功能时只绑定课程工具。"""
        with patch('app.qa.agent.WEB_SEARCH_ENABLED', False):
            names = {item.name for item in agent.CourseTools(None, '', self.course, 4, False, False).build_tools()}
        self.assertEqual(names, {'search_course', 'list_documents', 'read_document'})

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
            self.assertEqual(completed_turns(session, self.conversation, 'next'), [])

    def test_failed_stream_preserves_partial_text_without_leaking_provider_error(self):
        """验证流式输出失败时保留已发布正文，并隐藏服务商异常细节。"""
        with patch('app.qa.agent.get_llm', return_value=FailingModel(responses=[])):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'failed')
        self.assertEqual(answer['content'], '已生成的部分。')
        self.assertNotIn('private-provider-error-fixture', json.dumps(answer))

    def test_rag_also_streams_without_strict_review(self):
        """验证关闭严格核对时固定 RAG 也会流式发布正文。"""
        with patch('app.qa.manager.get_llm', return_value=shared.FakeLLM(answer='课程说明。[1] 补充说明：可用例子帮助理解。')):
            answer = self.wait(self.ask('进程是什么？', mode='rag'))
        self.assertEqual(answer['status'], 'completed')
        self.assertIn('补充说明', answer['content'])
        self.assertFalse(answer['run']['trace']['answer_review'])

    def test_budget_limits_web_calls_and_finalizes_without_more_tools(self):
        """验证网络工具受统一调用预算约束，并在预算耗尽后结束本轮。"""
        self.web_search.side_effect = None
        self.web_search.return_value = [SNIPPET]
        model = StreamingModel(responses=[call('search_web', {'query': '公开实验'}),
            call('search_web', {'query': '公开实验'}), AIMessage(content='根据已有资料总结。[1]')])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.agent.MAX_CALLS', 1):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(self.web_search.call_count, 1)
        self.assertEqual(len(answer['run']['trace']['tools']), 1)
        self.assertIn('查阅上限', answer['content'])
        self.assertIn('根据已有资料总结', answer['content'])
