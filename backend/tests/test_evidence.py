import asyncio
import json
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import test_conversations as shared
from app.qa.evidence import Review, render, review_options, parse_review, ReviewFormatError

SOURCE = {'number': 1, 'source': '进程.md', 'text': '进程是资源分配单位。线程是调度单位。'}


def claim(text='进程是资源分配单位。', number=1, quote='进程是资源分配单位', verdict='supported', scope='passage'):
    return {'text': text, 'verdict': verdict, 'scope': scope, 'evidence': [{'number': number, 'quote': quote}]}


def review(claims, insufficient=False):
    return {'claims': claims, 'insufficient': insufficient, 'include_catalog': False}


class EvidenceRenderTests(unittest.TestCase):
    def test_evaluation_recognizes_partial_evidence_gap_wording(self):
        from evaluation.run import checks
        result = checks({'expect_gap': True}, {'content': '本轮取得的片段仅支持以上结论，其余内容未取得足够依据。',
            'status': 'completed', 'citations': [], 'run': {'trace': {}}})
        self.assertTrue(result['automatic']['gap_when_expected'])
    def test_bad_claim_is_quarantined_without_repairing_its_evidence(self):
        bad = claim('未经确认的结论')
        bad['evidence'] *= 3
        parsed, rejected = parse_review(json.dumps(review([claim(), bad]), ensure_ascii=False))
        self.assertEqual(rejected, 1)
        self.assertEqual(len(parsed.claims), 1)
        self.assertEqual(len(parsed.claims[0].evidence), 1)
        with self.assertRaises(ReviewFormatError):
            parse_review(json.dumps(review([bad]), ensure_ascii=False))
    def test_provider_options_are_limited_to_verified_endpoint(self):
        self.assertEqual(review_options(SimpleNamespace(openai_api_base='https://other.example/v1')), {})
        options = review_options(SimpleNamespace(openai_api_base='https://api.deepseek.com'))
        self.assertEqual(options['response_format']['type'], 'json_object')
        self.assertEqual(options['extra_body']['thinking']['type'], 'disabled')
    def test_valid_claim_keeps_original_number(self):
        result, summary = render(Review.model_validate(review([claim()])), [SOURCE])
        self.assertIn('进程是资源分配单位。 [1]', result)
        self.assertEqual(summary['kept'], 1)

    def test_unquoted_inferences_and_collection_absence_are_removed(self):
        candidates = [claim(), claim('线程开销更小', verdict='unsupported', scope='inference'),
                      claim('资料库中绝无其他内容', scope='collection')]
        result, summary = render(Review.model_validate(review(candidates)), [SOURCE])
        self.assertNotIn('线程开销', result)
        self.assertNotIn('绝无', result)
        self.assertEqual(summary['rejected']['unsupported'], 1)
        self.assertEqual(summary['rejected']['scope'], 1)

    def test_unknown_number_fabricated_quote_and_embedded_citation_are_rejected(self):
        for candidate in (claim(number=99), claim(quote='资料不存在的摘录'), claim(text='说明[99]')):
            with self.subTest(candidate=candidate):
                result, summary = render(Review.model_validate(review([candidate])), [SOURCE])
                self.assertEqual(summary['kept'], 0)
                self.assertEqual(summary['rejected']['invalid_evidence'], 1)
                self.assertNotIn('[99]', result)
                self.assertIn('本轮取得的资料片段', result)

    def test_quote_must_match_its_specific_source(self):
        second = {'number': 2, 'source': '另页.md', 'text': '本页只说明文件存储方式。'}
        result, summary = render(Review.model_validate(review([claim(number=2)])), [SOURCE, second])
        self.assertEqual(summary['kept'], 0)

    def test_whitespace_variation_only_is_allowed(self):
        candidate = claim(quote='进程 是资源\n分配单位')
        _, summary = render(Review.model_validate(review([candidate])), [SOURCE])
        self.assertEqual(summary['kept'], 1)

    def test_empty_review_has_local_scope_gap_not_global_absence(self):
        result, summary = render(Review.model_validate(review([], True)), [SOURCE])
        self.assertIn('本轮取得的资料片段中', result)
        self.assertTrue(summary['insufficient'])


class ReviewerModel(shared.FakeLLM):
    def __init__(self, output, delay=False):
        super().__init__('进程是资源分配单位。[1] 线程开销更小。[1]')
        self.output, self.delay = output, delay
        self.review_started = Event()

    async def ainvoke(self, messages, **kwargs):
        if '证据核对器' in messages[0].content:
            self.review_started.set()
            if self.delay:
                await asyncio.sleep(30)
            return AIMessage(content=self.output)
        return await super().ainvoke(messages, **kwargs)


class EvidencePipelineTests(unittest.TestCase):
    setUp = shared.ConversationTests.setUp
    tearDown = shared.ConversationTests.tearDown
    publish = shared.ConversationTests.publish
    wait = shared.ConversationTests.wait
    ask = shared.ConversationTests.ask

    def test_only_supported_claim_is_published_and_restored(self):
        output = json.dumps(review([claim(), claim('线程开销更小', verdict='unsupported', scope='inference')]), ensure_ascii=False)
        with patch('app.qa.manager.get_llm', return_value=ReviewerModel(output)):
            run_id = self.ask()
            answer = self.wait(run_id)
        self.assertEqual(answer['status'], 'completed')
        self.assertNotIn('开销更小', answer['content'])
        self.assertEqual(answer['run']['trace']['evidence_check']['kept'], 1)
        stream = self.client.get(f'{self.path}/runs/{run_id}/events').text
        self.assertNotIn('开销更小', stream)
        self.assertIn('正在逐条核对回答依据', stream)

    def test_malformed_review_fails_closed_without_draft_leak(self):
        with patch('app.qa.manager.get_llm', return_value=ReviewerModel('not json')):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'failed')
        self.assertEqual(answer['content'], '')
        self.assertTrue(answer['citations'])
        self.assertEqual(answer['run']['trace']['evidence_check']['status'], 'unavailable')
        self.assertIn('未发布草稿', answer['run']['error'])

    def test_review_timeout_retains_evidence_without_unchecked_answer(self):
        with patch('app.qa.manager.get_llm', return_value=ReviewerModel('{}', delay=True)), patch('app.qa.evidence.REVIEW_TIMEOUT', .02):
            answer = self.wait(self.ask())
        self.assertEqual(answer['status'], 'failed')
        self.assertEqual(answer['content'], '')
        self.assertTrue(answer['citations'])

    def test_cancel_during_review_does_not_publish_draft_or_add_history(self):
        model = ReviewerModel('{}', delay=True)
        with patch('app.qa.manager.get_llm', return_value=model):
            run_id = self.ask()
            self.assertTrue(model.review_started.wait(5))
            pending = self.client.get(self.path).json()['messages'][-1]
            self.assertEqual(pending['content'], '')
            self.client.post(f'{self.path}/runs/{run_id}/cancel')
            stopped = self.wait(run_id)
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertEqual(stopped['content'], '')
        self.assertEqual(stopped['run']['trace']['evidence_check']['status'], 'interrupted')
        with patch('app.qa.manager.get_llm', return_value=shared.FakeLLM()):
            next_answer = self.wait(self.ask())
        self.assertEqual(next_answer['run']['trace']['history_turns'], 0)
