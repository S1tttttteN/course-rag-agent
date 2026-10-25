"""Real LangChain graph and real application lifecycle, with scripted provider responses."""
import asyncio
import json
import time
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import test_conversations as shared
from app.db.models import Run
from app.db.session import SessionLocal
from app.repositories.conversations import completed_turns
from app.services import conversations, runs


def call(name, args, identifier='call1'):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': identifier}])


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    cursor: int = 0
    seen: list = Field(default_factory=list)
    wait_after: int | None = None

    @property
    def _llm_type(self):
        return 'scripted-test-model'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if '证据核对器' in messages[0].content:
            return ChatResult(generations=[ChatGeneration(message=shared.synthetic_review(messages))])
        self.seen.append(messages)
        message = self.responses[min(self.cursor, len(self.responses) - 1)].model_copy(deep=True)
        message.id = None
        for tool_call in message.tool_calls:
            tool_call['id'] = f"{tool_call['id']}-{self.cursor}"
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.wait_after is not None and self.cursor >= self.wait_after:
            await asyncio.sleep(30)
        return self._generate(messages)


class AgentTests(unittest.TestCase):
    setUp = shared.ConversationTests.setUp
    tearDown = shared.ConversationTests.tearDown
    publish = shared.ConversationTests.publish
    wait = shared.ConversationTests.wait

    def ask(self, question='进程和线程有什么区别？'):
        response = self.client.post(f'{self.path}/runs', json={
            'question': question, 'mode': 'agent', 'semantic': False, 'rerank': False,
        })
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def test_multiple_searches_deduplicate_citations_and_restore_records(self):
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'}),
            call('search_course', {'query': '线程'}, 'call2'), AIMessage(content='进程分配资源，线程负责调度。[1]')])
        with patch('app.qa.agent.get_llm', return_value=model):
            run_id = self.ask()
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual([item['number'] for item in answer['citations']], [1])
        self.assertEqual(len(answer['run']['trace']['tools']), 2)
        self.assertTrue(all(item['status'] == 'completed' for item in answer['run']['trace']['tools']))
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
        foreign = self.client.post(f'/api/courses/{self.other}/documents', files={'file': ('隔离.md', '绝密口令不应出现'.encode())}).json()
        self.publish(foreign)
        model = ScriptedModel(responses=[call('read_document', {'document_id': foreign['id'], 'version_id': foreign['latest']['id']}),
            call('search_course', {'query': '进程', 'course_id': self.other}, 'inject'), AIMessage(content='未经检索的知识不能发布')])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertEqual(answer['citations'], [])
        self.assertIn('未找到足够依据', answer['content'])
        self.assertNotIn('绝密口令', str(model.seen))
        self.assertTrue(all(item['status'] == 'failed' for item in answer['run']['trace']['tools']))

    def test_budget_stops_repeated_invalid_calls(self):
        model = ScriptedModel(responses=[call('search_course', {'query': '进程', 'invalid': True})])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.agent.MAX_CALLS', 2):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(len(answer['run']['trace']['tools']), 2)
        self.assertLessEqual(model.cursor, 3)
        self.assertIn('未找到足够依据', answer['content'])

    def test_tool_timeout_is_recorded_without_provider_error_details(self):
        async def delayed(*args):
            await asyncio.sleep(30)
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'}), AIMessage(content='没有依据')])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.manager.RunManager.retrieve', side_effect=delayed), patch('app.qa.agent.TOOL_TIMEOUT', .02):
            answer = self.wait(self.ask())
        tool = answer['run']['trace']['tools'][0]
        self.assertEqual(tool['status'], 'failed')
        self.assertIn('超时', tool['summary'])

    def test_cancel_retains_sources_and_excludes_incomplete_history(self):
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
            self.assertEqual(completed_turns(session, self.conversation, 'next'), [])

    def test_recovery_marks_unfinished_tool_interrupted_without_replay(self):
        with SessionLocal() as session:
            run = conversations.create_run(session, self.course, self.conversation, '重启验证', 4, False, False, 'agent')
        runs.tool_update(run['id'], {'id': 'unfinished', 'status': 'running', 'name': 'search_course'})
        self.context.__exit__(None, None, None)
        with patch('app.qa.agent.get_llm', side_effect=AssertionError('must not replay')):
            self.context = shared.TestClient(shared.app)
            self.client = self.context.__enter__()
            answer = self.wait(run['id'])
        self.assertEqual(answer['status'], 'interrupted')
        self.assertEqual(answer['run']['trace']['tools'][0]['status'], 'interrupted')

    def test_no_tool_answer_is_rejected_and_usage_is_reported(self):
        model = ScriptedModel(responses=[AIMessage(content='不受资料支持的解释', usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})])
        with patch('app.qa.agent.get_llm', return_value=model):
            answer = self.wait(self.ask())
        self.assertNotIn('不受资料支持的解释', answer['content'])
        self.assertIn('未找到足够依据', answer['content'])
        self.assertEqual(answer['run']['trace']['usage']['input_tokens'], 10)
        self.assertEqual(answer['run']['trace']['usage']['model_calls'], 1)
        self.assertGreaterEqual(answer['run']['trace']['elapsed_seconds'], 0)

    def test_invalid_mode_is_rejected(self):
        response = self.client.post(f'{self.path}/runs', json={'question': '测试', 'mode': 'web'})
        self.assertEqual(response.status_code, 422)

    def test_budget_with_evidence_finishes_without_more_tools(self):
        model = ScriptedModel(responses=[call('search_course', {'query': '进程'}),
            call('search_course', {'query': '线程'}, 'second'), AIMessage(content='进程分配资源。[1]')])
        with patch('app.qa.agent.get_llm', return_value=model), patch('app.qa.agent.MAX_CALLS', 1):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'completed')
        self.assertEqual(len(answer['run']['trace']['tools']), 1)
        self.assertIn('查阅上限', answer['content'])
        self.assertIn('进程分配资源', answer['content'])

    def test_empty_course_avoids_paid_model_call(self):
        base = f'/api/courses/{self.other}/conversations'
        conversation = self.client.post(base, json={}).json()['id']
        with patch('app.qa.agent.get_llm', side_effect=AssertionError('must not call')):
            self.client.post(f'{base}/{conversation}/runs', json={'question': '课程是什么？', 'mode': 'agent'})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                detail = self.client.get(f'{base}/{conversation}').json()
                if not detail['active_run']:
                    break
                time.sleep(.03)
        self.assertEqual(detail['messages'][-1]['status'], 'completed')
        self.assertIn('未找到足够依据', detail['messages'][-1]['content'])
