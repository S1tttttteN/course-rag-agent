"""验证对话持久化、SSE、引用和后台运行状态。"""

import asyncio
import json
import re
from threading import Event
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select

from test_courses import app
from test_documents import FakeEmbeddings
from app.db.models import Citation, CleanupJob, Message, Run
from app.db.models import Chunk
from app.db.session import SessionLocal, engine
from app.jobs.indexing import cleanup_job, process_version
from app.services import conversations, runs


def synthetic_review(messages):
    """为管道测试构造确定性的核对模型响应。

    语义支持判断由独立的证据核对测试覆盖。

    Args:
        messages: 包含草稿和来源 JSON 的核对消息列表。

    Returns:
        包含合成核对 JSON 的 AI 消息。
    """
    data = json.loads(messages[-1].content)
    references = [int(number) for number in re.findall(r'\[(\d+)\]', data['draft'])] or [data['sources'][0]['number']]
    return AIMessage(content=json.dumps({'claims': [{
        'text': re.sub(r'\[\d+\]', '', data['draft']).strip()[:180],
        'verdict': 'supported', 'scope': 'passage',
        'evidence': [{'number': number, 'quote': next((source['text'][:80] for source in data['sources'] if source['number'] == number), data['sources'][0]['text'][:80])} for number in dict.fromkeys(references)],
    }], 'insufficient': False, 'include_catalog': False}, ensure_ascii=False))


class FakeLLM:
    """提供确定的流式回答、追问改写和核对结果。

    Attributes:
        answer: 分块发送的预设回答。
        inputs: 已记录的生成输入。
        rewrites: 已记录的追问改写输入。
    """

    def __init__(self, answer="进程是资源分配单位，线程是调度单位。[1]"):
        """设置预设回答并初始化输入记录。

        Args:
            answer: 要分块发送的回答文本。
        """
        self.answer = answer
        self.inputs = []
        self.rewrites = []

    async def ainvoke(self, messages, **kwargs):
        """返回合成核对结果或固定的追问改写。

        Args:
            messages: 发送给测试模型的消息列表。
            kwargs: 调用方附加的关键字参数。

        Returns:
            包含核对结果或改写文本的 AI 消息。
        """
        if '证据核对器' in messages[0].content:
            return synthetic_review(messages)
        self.rewrites.append(messages)
        return AIMessage(content="线程与进程的区别")

    async def astream(self, messages, **kwargs):
        """将预设回答分块发出。

        Args:
            messages: 要记录的回答生成输入。
            kwargs: 调用方附加的关键字参数。

        Yields:
            最多八个字符的 AI 消息块。
        """
        self.inputs.append(messages)
        for position in range(0, len(self.answer), 8):
            await asyncio.sleep(0.01)
            yield AIMessageChunk(content=self.answer[position:position + 8])


class SlowLLM(FakeLLM):
    """模拟发出首段正文后长时间等待的模型。"""

    async def astream(self, messages, **kwargs):
        """先发出部分回答，再等待后续输出。

        Args:
            messages: 要记录的回答生成输入。
            kwargs: 调用方附加的关键字参数。

        Yields:
            初始正文和等待结束后的尾部消息块。
        """
        self.inputs.append(messages)
        yield AIMessageChunk(content="已依据资料开始整理。[1]" * 12)
        await asyncio.sleep(30)
        yield AIMessageChunk(content="结束")


class BrokenLLM(FakeLLM):
    """模拟发出部分回答后失败的模型。"""

    async def astream(self, messages, **kwargs):
        """发出部分正文后模拟服务商异常。

        Args:
            messages: 流式接口收到的输入消息。
            kwargs: 调用方附加的关键字参数。

        Yields:
            失败前的单个消息块。

        Raises:
            RuntimeError: 用于断言服务商内部错误不外泄的模拟异常。
        """
        yield AIMessageChunk(content="部分回答。[1]")
        raise RuntimeError("secret fixture must not be sent to browser")


class ConversationTests(unittest.TestCase):
    """验证对话接口及问答运行的持久化和隔离。"""

    def setUp(self):
        """启动测试客户端，创建两门课程和带资料的测试对话。"""
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        self.course = self.client.post('/api/courses', json={'name': '聊天验证'}).json()['id']
        self.other = self.client.post('/api/courses', json={'name': '另一门课'}).json()['id']
        self.base = f'/api/courses/{self.course}/conversations'
        self.conversation = self.client.post(self.base, json={}).json()['id']
        self.path = f'{self.base}/{self.conversation}'
        self.document = self.client.post(f'/api/courses/{self.course}/documents', files={'file': ('进程.md', '进程是资源分配单位。线程是调度单位。'.encode())}).json()
        self.publish(self.document)

    def publish(self, document):
        """使用固定嵌入向量处理并发布资料版本。

        Args:
            document: 包含最新版本 ID 的资料响应字典。
        """
        with patch('app.jobs.indexing.get_embeddings', return_value=FakeEmbeddings()), patch('app.jobs.indexing.embedding_profile', return_value='test:3'):
            process_version(document['latest']['id'], Event())

    def tearDown(self):
        """删除测试课程，执行存储清理并关闭客户端及向量索引。"""
        for course in (self.course, self.other):
            self.client.delete(f'/api/courses/{course}')
        self.context.__exit__(None, None, None)
        with SessionLocal() as session:
            jobs = list(session.scalars(select(CleanupJob.id)))
        for job in jobs:
            cleanup_job(job)
        from app.retrieval import index
        index.close()
        engine.dispose()

    def ask(self, question='进程和线程的区别是什么？'):
        """在当前测试对话中提交问答。

        Args:
            question: 用户提交的问题正文。

        Returns:
            已接受的运行 ID。
        """
        response = self.client.post(f'{self.path}/runs', json={'question': question, 'semantic': False, 'rerank': False})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def wait(self, run_id, partial=False):
        """轮询对话，直到助手消息达到预期状态。

        Args:
            run_id: 问答运行 ID。
            partial: 是否在已发布部分内容或进入带引用的生成阶段时返回。

        Returns:
            符合等待条件的助手消息。

        Raises:
            AssertionError: 运行在等待期限内未达到预期状态。
        """
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            detail = self.client.get(self.path).json()
            answer = next(message for message in detail['messages'] if message['run_id'] == run_id and message['role'] == 'assistant')
            if (partial and (answer['content'] or (answer['citations'] and answer['run']['phase'] == '正在生成回答'))) or (not partial and answer['status'] in {'completed', 'cancelled', 'failed', 'interrupted'}):
                return answer
            time.sleep(0.03)
        self.fail('run did not reach expected state')

    def test_persistent_conversation_stream_and_citation_survives_source_delete(self):
        """验证 SSE 事件可回放，删除资料后历史引用快照仍可读取。"""
        with patch('app.qa.manager.get_llm', return_value=FakeLLM()):
            run_id = self.ask()
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(answer['citations'][0]['number'], 1)
        stream = self.client.get(f'{self.path}/runs/{run_id}/events').text
        ids = [int(value) for value in re.findall(r'^id: (\d+)', stream, re.M)]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertIn('event: delta', stream)
        self.assertIn('event: done', stream)
        self.assertNotIn('event: delta', self.client.get(f'{self.path}/runs/{run_id}/events?after={ids[-1]}').text)
        citation_id = answer['citations'][0]['id']
        self.client.delete(f"/api/courses/{self.course}/documents/{self.document['id']}")
        snapshot = self.client.get(f'{self.path}/citations/{citation_id}').json()
        self.assertEqual(snapshot['source_state'], 'deleted')
        self.assertIn('资源分配', snapshot['text'])
        renamed = self.client.patch(self.path, json={'title': '进程复习'})
        self.assertEqual(renamed.status_code, 200)
        self.context.__exit__(None, None, None)
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        restored = self.client.get(self.path).json()
        self.assertEqual(restored['title'], '进程复习')
        self.assertEqual(restored['messages'][1]['content'], answer['content'])
        self.assertEqual(restored['messages'][1]['citations'][0]['source_state'], 'deleted')

    def test_replacement_marks_old_snapshot_updated(self):
        """验证发布替换版本后旧引用快照标记为已更新。"""
        with patch('app.qa.manager.get_llm', return_value=FakeLLM()):
            answer = self.wait(self.ask())
        replacement = self.client.put(f"/api/courses/{self.course}/documents/{self.document['id']}/file", files={'file': ('新版.md', '进程是程序执行实例。'.encode())}).json()
        self.publish(replacement)
        snapshot = self.client.get(f"{self.path}/citations/{answer['citations'][0]['id']}").json()
        self.assertEqual(snapshot['source_state'], 'updated')
        self.assertEqual(snapshot['version'], 1)
        self.assertIn('资源分配', snapshot['text'])

    def test_followup_is_rewritten_and_completed_history_is_used(self):
        """验证追问结合已完成历史改写检索词。"""
        model = FakeLLM()
        with patch('app.qa.manager.get_llm', return_value=model):
            self.wait(self.ask('解释进程和线程。'))
            second = self.wait(self.ask('它们有什么区别？'))
        self.assertEqual(len(model.rewrites), 1)
        self.assertEqual(second['run']['query'], '线程与进程的区别')
        self.assertEqual(second['run']['trace']['history_turns'], 1)
        self.assertIn('解释进程和线程', model.inputs[-1][1].content)

    def test_failed_draft_is_not_published_or_used_in_history(self):
        """验证生成失败的草稿不会发布或进入追问历史。"""
        with patch('app.qa.manager.get_llm', return_value=BrokenLLM()):
            failed = self.wait(self.ask())
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['content'], '')
        self.assertTrue(failed['citations'])
        self.assertNotIn('secret fixture', failed['run']['error'])
        model = FakeLLM()
        with patch('app.qa.manager.get_llm', return_value=model):
            completed = self.wait(self.ask())
        self.assertEqual(completed['run']['trace']['history_turns'], 0)
        self.assertEqual(model.rewrites, [])

    def test_cancel_preserves_evidence_not_unreviewed_draft_and_blocks_duplicate(self):
        """验证活动运行会阻止重复提交，取消时保留引用而未核对草稿仍不发布。"""
        with patch('app.qa.manager.get_llm', return_value=SlowLLM()):
            run_id = self.ask()
            partial = self.wait(run_id, partial=True)
            self.assertEqual(self.client.post(f'{self.path}/runs', json={'question': '重复问题'}).status_code, 409)
            self.assertEqual(self.client.post(f'{self.path}/runs/{run_id}/cancel').status_code, 200)
            stopped = self.wait(run_id)
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertEqual(stopped['content'], partial['content'])
        self.assertEqual(stopped['content'], '')
        self.assertTrue(stopped['citations'])

    def test_cross_course_access_to_conversation_run_and_citation_is_rejected(self):
        """验证对话、运行及引用接口都会校验课程归属。"""
        with patch('app.qa.manager.get_llm', return_value=FakeLLM()):
            answer = self.wait(self.ask())
        foreign = f'/api/courses/{self.other}/conversations/{self.conversation}'
        for suffix in ('', f"/citations/{answer['citations'][0]['id']}", f"/runs/{answer['run_id']}/events"):
            self.assertEqual(self.client.get(foreign + suffix).status_code, 404)
        self.assertEqual(self.client.post(foreign + '/runs', json={'question': '跨课程'}).status_code, 404)
        self.assertEqual(self.client.delete(foreign).status_code, 404)

    def test_restart_marks_unfinished_run_interrupted_without_replaying(self):
        """验证重启标记遗留运行中断并保留其已保存内容。"""
        with SessionLocal() as session:
            pending = conversations.create_run(session, self.course, self.conversation, '待恢复问题', 4, False, False)
        runs.status(pending['id'], '正在生成回答')
        runs.append(pending['id'], '已生成的部分')
        runs.recover()
        restored = self.client.get(self.path).json()['messages'][1]
        self.assertEqual(restored['status'], 'interrupted')
        self.assertEqual(restored['content'], '已生成的部分')
        self.assertIsNone(self.client.get(self.path).json()['active_run'])

    def test_delete_running_conversation_removes_history_but_keeps_documents(self):
        """验证删除运行中的对话会清除历史，同时保留课程资料。"""
        with patch('app.qa.manager.get_llm', return_value=SlowLLM()):
            run_id = self.ask()
            self.wait(run_id, partial=True)
            self.assertEqual(self.client.delete(self.path).status_code, 204)
        self.assertEqual(self.client.get(self.path).status_code, 404)
        self.assertTrue(self.client.get(f'/api/courses/{self.course}/documents').json()[0]['usable'])
        with SessionLocal() as session:
            self.assertIsNone(session.get(Run, run_id))
            self.assertFalse(list(session.scalars(select(Message).where(Message.run_id == run_id))))
            self.assertFalse(list(session.scalars(select(Citation).where(Citation.run_id == run_id))))

    def test_empty_course_explains_evidence_gap_without_model_call(self):
        """验证空课程无需调用模型即可返回资料缺口说明。"""
        other_base = f'/api/courses/{self.other}/conversations'
        conversation = self.client.post(other_base, json={}).json()['id']
        with patch('app.qa.manager.get_llm', side_effect=AssertionError('must not call')):
            self.client.post(f'{other_base}/{conversation}/runs', json={'question': '为什么？', 'semantic': False})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                detail = self.client.get(f'{other_base}/{conversation}').json()
                if not detail['active_run']:
                    break
                time.sleep(0.03)
        self.assertEqual(detail['messages'][1]['status'], 'completed')
        self.assertIn('未找到足够依据', detail['messages'][1]['content'])
        self.assertEqual(detail['messages'][1]['citations'], [])
        with patch('app.qa.manager.get_llm', side_effect=AssertionError('must not call')):
            self.client.post(f'{other_base}/{conversation}/runs', json={'question': '还有呢？', 'semantic': False})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                detail = self.client.get(f'{other_base}/{conversation}').json()
                if not detail['active_run']:
                    break
                time.sleep(0.03)
        self.assertEqual(detail['messages'][-1]['status'], 'completed')

    def test_unknown_citation_is_flagged_and_no_source_is_fabricated(self):
        """验证未知引用编号会被标记，引用列表只包含实际保存的来源。"""
        with patch('app.qa.manager.get_llm', return_value=FakeLLM('解释。[99]')):
            answer = self.wait(self.ask())
        self.assertEqual(answer['run']['trace']['evidence_check']['rejected']['invalid_evidence'], 1)
        self.assertNotIn('[99]', answer['content'])
        self.assertNotIn(99, [citation['number'] for citation in answer['citations']])

    def test_registry_reuses_numbers_and_rejects_foreign_chunks(self):
        """验证引用登记复用编号并排除外部课程片段。"""
        foreign = self.client.post(f'/api/courses/{self.other}/documents', files={'file': ('foreign.md', '线程'.encode())}).json()
        self.publish(foreign)
        with SessionLocal() as session:
            own_chunk = session.scalar(select(Chunk).where(Chunk.version_id == self.document['latest']['id']))
            foreign_chunk = session.scalar(select(Chunk).where(Chunk.version_id == foreign['latest']['id']))
            run = conversations.create_run(session, self.course, self.conversation, '注册依据', 4, False, False)
            records = [{'id': own_chunk.id, 'document_id': self.document['id']}, {'id': foreign_chunk.id, 'document_id': foreign['id']}]
        sources = runs.snapshot(run['id'], records + records)
        self.assertEqual([item['number'] for item in sources], [1])
        self.assertEqual(runs.snapshot(run['id'], records)[0]['id'], sources[0]['id'])
        runs.finish(run['id'], 'completed')

    def test_service_shutdown_preserves_partial_as_interrupted(self):
        """验证服务停止后已发布内容仍可作为中断记录读取。"""
        with patch('app.qa.manager.get_llm', return_value=SlowLLM()):
            run_id = self.ask()
            before = self.wait(run_id, partial=True)
            self.context.__exit__(None, None, None)
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        after = self.wait(run_id)
        self.assertEqual(after['status'], 'interrupted')
        self.assertEqual(after['content'], before['content'])

    def test_model_length_limit_is_not_saved_as_completed_history(self):
        """验证长度截断的模型回复不会记为已完成问答。"""
        class LimitedLLM(FakeLLM):
            """模拟因输出长度限制而结束生成的模型。"""

            async def astream(self, messages, **kwargs):
                """发送携带长度终止标志的模型片段。

                Args:
                    messages: 流式接口收到的输入消息。
                    kwargs: 调用方附加的关键字参数。

                Yields:
                    模拟截断响应的 AI 消息块。
                """
                yield AIMessageChunk(content='尚未写完的解释。[1]', response_metadata={'finish_reason': 'length'})
        with patch('app.qa.manager.get_llm', return_value=LimitedLLM()):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'failed')
        self.assertIn('长度上限', answer['run']['error'])
        self.assertEqual(answer['content'], '')
        self.assertTrue(answer['citations'])
