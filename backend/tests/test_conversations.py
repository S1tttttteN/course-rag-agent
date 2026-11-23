"""验证对话持久化、SSE、引用和后台运行状态。"""

import re
from threading import Event
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select

from agent_fakes import PausedModel, ScriptedModel, StreamingModel, call
from test_courses import app
from test_documents import FakeEmbeddings
from app.db.models import Citation, CleanupJob, Message, Run
from app.db.models import Chunk
from app.db.session import SessionLocal, engine
from app.jobs.indexing import cleanup_job, process_version
from app.services import conversations, runs


def course_model(text="进程是资源分配单位，线程是调度单位。[1]"):
    """构造先检索课程再分段输出指定内容的模型。"""
    return StreamingModel(responses=[call('search_course', {'query': '进程'}), AIMessage(content=text)])


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

    def ask(self, question='进程和线程的区别是什么？', *, web_search=False):
        """在当前测试对话中提交问答。

        Args:
            question: 用户提交的问题正文。
            web_search: 是否为本轮提供联网工具。

        Returns:
            已接受的运行 ID。
        """
        response = self.client.post(f'{self.path}/runs', json={
            'question': question, 'web_search': web_search,
        })
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def wait(self, run_id, partial=False):
        """轮询对话，直到助手消息达到预期状态。

        Args:
            run_id: 问答运行 ID。
            partial: 是否在已发布部分内容时返回。

        Returns:
            符合等待条件的助手消息。

        Raises:
            AssertionError: 运行在等待期限内未达到预期状态。
        """
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            detail = self.client.get(self.path).json()
            answer = next(message for message in detail['messages'] if message['run_id'] == run_id and message['role'] == 'assistant')
            if (partial and answer['content']) or (not partial and answer['status'] in {'completed', 'cancelled', 'failed', 'interrupted'}):
                return answer
            time.sleep(0.03)
        self.fail('run did not reach expected state')

    def test_persistent_conversation_stream_and_citation_survives_source_delete(self):
        """验证 SSE 事件可回放，删除资料后历史引用快照仍可读取。"""
        with patch('app.qa.agent.get_llm', return_value=course_model()):
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
        with patch('app.qa.agent.get_llm', return_value=course_model()):
            answer = self.wait(self.ask())
        replacement = self.client.put(f"/api/courses/{self.course}/documents/{self.document['id']}/file", files={'file': ('新版.md', '进程是程序执行实例。'.encode())}).json()
        self.publish(replacement)
        snapshot = self.client.get(f"{self.path}/citations/{answer['citations'][0]['id']}").json()
        self.assertEqual(snapshot['source_state'], 'updated')
        self.assertEqual(snapshot['version'], 1)
        self.assertIn('资源分配', snapshot['text'])

    def test_cancel_preserves_partial_answer_and_blocks_duplicate(self):
        """验证活动运行会阻止重复提交，取消时保留部分正文。"""
        with patch('app.qa.agent.get_llm', return_value=PausedModel(responses=[])):
            run_id = self.ask()
            partial = self.wait(run_id, partial=True)
            self.assertEqual(self.client.post(f'{self.path}/runs', json={'question': '重复问题'}).status_code, 409)
            self.assertEqual(self.client.post(f'{self.path}/runs/{run_id}/cancel').status_code, 200)
            stopped = self.wait(run_id)
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertEqual(stopped['content'], partial['content'])

    def test_cross_course_access_to_conversation_run_and_citation_is_rejected(self):
        """验证对话、运行及引用接口都会校验课程归属。"""
        with patch('app.qa.agent.get_llm', return_value=course_model()):
            answer = self.wait(self.ask())
        foreign = f'/api/courses/{self.other}/conversations/{self.conversation}'
        for suffix in ('', f"/citations/{answer['citations'][0]['id']}", f"/runs/{answer['run_id']}/events"):
            self.assertEqual(self.client.get(foreign + suffix).status_code, 404)
        self.assertEqual(self.client.post(foreign + '/runs', json={'question': '跨课程'}).status_code, 404)
        self.assertEqual(self.client.delete(foreign).status_code, 404)

    def test_restart_marks_unfinished_run_interrupted_without_replaying(self):
        """验证重启标记遗留运行中断并保留其已保存内容。"""
        with SessionLocal() as session:
            pending = conversations.create_run(session, self.course, self.conversation, '待恢复问题')
        runs.status(pending['id'], '正在生成回答')
        runs.append(pending['id'], '已生成的部分')
        runs.recover()
        restored = self.client.get(self.path).json()['messages'][1]
        self.assertEqual(restored['status'], 'interrupted')
        self.assertEqual(restored['content'], '已生成的部分')
        self.assertIsNone(self.client.get(self.path).json()['active_run'])

    def test_delete_running_conversation_removes_history_but_keeps_documents(self):
        """验证删除运行中的对话会清除历史，同时保留课程资料。"""
        with patch('app.qa.agent.get_llm', return_value=PausedModel(responses=[])):
            run_id = self.ask()
            self.wait(run_id, partial=True)
            self.assertEqual(self.client.delete(self.path).status_code, 204)
        self.assertEqual(self.client.get(self.path).status_code, 404)
        self.assertTrue(self.client.get(f'/api/courses/{self.course}/documents').json()[0]['usable'])
        with SessionLocal() as session:
            self.assertIsNone(session.get(Run, run_id))
            self.assertFalse(list(session.scalars(select(Message).where(Message.run_id == run_id))))
            self.assertFalse(list(session.scalars(select(Citation).where(Citation.run_id == run_id))))

    def test_unknown_citation_is_flagged_and_no_source_is_fabricated(self):
        """验证未知引用编号会被标记，引用列表只包含实际保存的来源。"""
        with patch('app.qa.agent.get_llm', return_value=course_model('解释。[99]')):
            answer = self.wait(self.ask())
        self.assertEqual(answer['run']['trace']['unknown_citations'], [99])
        self.assertNotIn(99, [citation['number'] for citation in answer['citations']])

    def test_registry_reuses_numbers_and_rejects_foreign_chunks(self):
        """验证引用登记复用编号并排除外部课程片段。"""
        foreign = self.client.post(f'/api/courses/{self.other}/documents', files={'file': ('foreign.md', '线程'.encode())}).json()
        self.publish(foreign)
        with SessionLocal() as session:
            own_chunk = session.scalar(select(Chunk).where(Chunk.version_id == self.document['latest']['id']))
            foreign_chunk = session.scalar(select(Chunk).where(Chunk.version_id == foreign['latest']['id']))
            run = conversations.create_run(session, self.course, self.conversation, '注册依据')
            records = [{'id': own_chunk.id, 'document_id': self.document['id']}, {'id': foreign_chunk.id, 'document_id': foreign['id']}]
        sources = runs.snapshot(run['id'], records + records)
        self.assertEqual([item['number'] for item in sources], [1])
        self.assertEqual(runs.snapshot(run['id'], records)[0]['id'], sources[0]['id'])
        runs.finish(run['id'], 'completed')

    def test_service_shutdown_preserves_partial_as_interrupted(self):
        """验证服务停止后已发布内容仍可作为中断记录读取。"""
        with patch('app.qa.agent.get_llm', return_value=PausedModel(responses=[])):
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
        model = ScriptedModel(responses=[
            AIMessage(content='尚未写完的解释。', response_metadata={'done_reason': 'length'}),
        ])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'failed')
        self.assertIn('长度上限', answer['run']['error'])
        with SessionLocal() as session:
            self.assertIsNone(session.get(Run, answer['run_id']).agent_checkpoint_id)
