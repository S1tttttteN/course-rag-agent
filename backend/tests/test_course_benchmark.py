"""覆盖课程评测的证据计分、会话隔离和匿名导出。"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import test_courses  # 初始化测试专用数据目录。
from agent_fakes import StreamingModel, call
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.retrieval.search import rank_candidates
from evaluation.course_dataset import resolve_evidence
from evaluation.course_report import export_experiments, prepare_review, read_scores, render_report
from evaluation.real_course import retrieval_scores, run_qa_case, run_retrieval_case
from evaluation.run import publish_document
from test_documents import FakeEmbeddings


class CourseBenchmarkTests(unittest.TestCase):
    """覆盖需要独立于模型服务验证的评测行为。"""

    def test_equivalent_chunks_count_as_one_evidence_group(self):
        """同一组有多个命中时只计一次，首个命中决定 MRR。"""
        chunks = {"D01": [
            {"id": "p1", "text": "同一  事实"},
            {"id": "p2", "text": "同一\n事实"},
            {"id": "p3", "text": "另一事实"},
        ]}
        groups = resolve_evidence(
            [[{"document": "D01", "anchor": anchor}] for anchor in ("同一 事实", "另一事实")],
            chunks,
        )
        self.assertEqual(retrieval_scores(["x", "p1", "p2", "y", "p3"], groups), (.5, .5))
        self.assertEqual(retrieval_scores(["p1", "p3"], groups), (1, 1))
        with self.assertRaises(ValueError):
            resolve_evidence([[{"document": "D01", "anchor": "找不到"}]], chunks)

    def test_vector_only_does_not_build_keyword_index(self):
        """纯向量能返回候选，嵌入故障时不借用 BM25。"""
        chunk = Mock(id="1", text="证据", page=None, start_index=0)
        document = Mock(id="d")
        version = Mock(id="v", embedding_profile="test", filename="note.md", number=1)
        with (
            patch("app.retrieval.search.SessionLocal"),
            patch("app.retrieval.search.require_course"),
            patch("app.retrieval.search.active_chunks", return_value=[(chunk, document, version)]),
            patch("app.retrieval.search.keyword_index", side_effect=AssertionError),
            patch("app.retrieval.search.embedding_profile", return_value="test"),
            patch("app.retrieval.search.get_embeddings") as embeddings,
            patch("app.retrieval.search.index.collection") as collection,
        ):
            collection.return_value.query.return_value = {"ids": [["1"]]}
            ranked, trace = rank_candidates("course", "查询", keyword=False)
            self.assertEqual([row["id"] for row in ranked], ["1"])
            self.assertTrue(trace["semantic"])
            self.assertFalse(trace["keyword"])
            embeddings.return_value.embed_query.side_effect = RuntimeError()
            ranked, trace = rank_candidates("course", "查询", keyword=False)
            self.assertEqual(ranked, [])
            self.assertFalse(trace["semantic"])
            self.assertFalse(trace["keyword"])

    def test_rerank_uses_same_pool_and_failed_service_is_marked(self):
        """融合两组共用初排名单，重排未执行时标明降级。"""
        trace = {"keyword": True, "semantic": True, "reranked": False}
        ranked = [{"id": "gold"}]
        case = {"id": "R01", "category": "exact", "question": "查询", "evidence": []}
        with (
            patch("app.retrieval.search.rank_candidates", return_value=(ranked, trace)) as rank,
            patch("app.retrieval.search.finish_search", return_value={"results": ranked, "trace": trace}) as finish,
        ):
            rows = list(run_retrieval_case("course", case, [{"gold"}], 0))
        self.assertEqual(rank.call_count, 3)
        self.assertIs(finish.call_args_list[-1].args[2], finish.call_args_list[-2].args[2])
        self.assertEqual(rows[-2]["configuration"], "hybrid")
        self.assertFalse(rows[-2]["degraded"])
        self.assertTrue(rows[-1]["degraded"])

    def test_qa_reuses_only_the_same_scenario_conversation(self):
        """Agent 复用场景会话，固定 RAG 使用独立的内存历史。"""
        client = Mock()
        client.post.return_value.json.return_value = {"id": "agent-session"}
        turn = {"question": "为什么？", "reference": "参考", "evidence": []}
        case = {"id": "Q01", "category": "followup", "turns": [turn, turn]}
        answer = {"status": "completed", "run": {"trace": {}}}

        async def fixed(course, question, history, timeout):
            """记录基线每轮可见的历史长度。"""
            history.append(question)
            return answer, len(history)

        with (
            patch("evaluation.fixed_rag.run_fixed_turn", new=AsyncMock(side_effect=fixed)),
            patch("evaluation.run.run_turn", return_value=(answer, 1)) as run,
        ):
            rows = run_qa_case(client, "course", case, 1, 0, 2)
        self.assertEqual([row["configuration"] for row in rows], ["rag", "rag", "agent", "agent"])
        self.assertEqual([row["elapsed_seconds"] for row in rows], [1, 2, 1, 1])
        self.assertEqual([request.args[1] for request in run.call_args_list],
                         ["/api/courses/course/conversations/agent-session"] * 2)
        self.assertEqual(run.call_args.args[2], {"question": "为什么？"})
        self.assertEqual(client.post.call_count, 1)

    def test_public_fixture_agent_and_fixed_rag_complete_followup(self):
        """用公开资料跑通两组问答，并生成可重新读取的盲评包。"""
        fixture = Path(__file__).resolve().parents[1] / 'evaluation' / 'fixtures' / '进程笔记.md'
        turn = {'question': 'PCB 包含什么？', 'reference': '进程标识符', 'evidence': []}
        followup = {**turn, 'question': '它还包含哪些信息？'}
        case = {'id': 'Q01', 'category': 'followup', 'turns': [turn, followup]}
        fixed = StreamingModel(responses=[
            AIMessage(content='PCB 保存进程标识符。[1]'),
            AIMessage(content='PCB 包含哪些字段？'),
            AIMessage(content='还保存程序计数器。[1]'),
        ])
        agent = StreamingModel(responses=[
            call('search_course', {'query': 'PCB'}),
            AIMessage(content='PCB 保存进程标识符。[1]'),
            call('search_course', {'query': '程序计数器'}),
            AIMessage(content='还保存程序计数器。[1]'),
        ])

        with (
            TestClient(test_courses.app) as client,
            patch('app.jobs.indexing.get_embeddings', return_value=FakeEmbeddings()),
            patch('app.jobs.indexing.embedding_profile', return_value='test:3'),
            patch('app.retrieval.search.get_embeddings', return_value=FakeEmbeddings()),
            patch('app.retrieval.search.embedding_profile', return_value='test:3'),
            patch('evaluation.fixed_rag.get_llm', return_value=fixed),
            patch('app.qa.agent.get_llm', return_value=agent),
        ):
            course = client.post('/api/courses', json={'name': '公开评测资料'}).json()['id']
            try:
                publish_document(client, course, fixture.name, fixture.read_bytes())
                rows = run_qa_case(client, course, case, 1, 0, 10)
            finally:
                client.delete(f'/api/courses/{course}')

        self.assertEqual([row['answer']['status'] for row in rows], ['completed'] * 4)
        self.assertTrue(all(row['answer']['citations'][0]['source'] == fixture.name for row in rows))
        self.assertIn(turn['question'], fixed.seen[1][-1].content)
        self.assertIn(turn['question'], [message.content for message in agent.seen[2]])
        report = {
            'run_id': 'fixture', 'status': 'completed', 'application_version': 'test',
            'cases_sha256': 'fixture', 'manifest': {'documents': [], 'chunk_size': 500, 'chunk_overlap': 100},
            'indexed_chunks': {}, 'models': {'chat_model': 'scripted', 'embedding_model': 'test:3', 'reranker': {}},
            'parameters': {'repeats': 1, 'workers': 1, 'timing': {'rag': 'in_process', 'agent': 'http_polling'}},
            'retrieval': [], 'qa': rows,
        }
        with TemporaryDirectory() as temporary:
            prepare_review(report, Path(temporary))
            self.assertEqual(len(read_scores(Path(temporary), 'fixture')), 4)
        self.assertIn('固定 RAG 记录进程内耗时', render_report(report, {}))
        previous = {**report, 'qa': [*rows, {**rows[0], 'configuration': 'rag_rerank'}]}
        self.assertIn('| all | rag_rerank |', render_report(previous, {}))

    def test_anonymous_export_does_not_copy_private_fields(self):
        """两组样本可配对，原始问题、来源和回答不会出现在导出。"""
        base = {
            "id": "Q01", "category": "followup", "repeat": 2, "turn": 1,
            "started_at": "2026-01-01T01:00:00+00:00",
            "ended_at": "2026-01-01T01:00:01+00:00", "elapsed_seconds": 1.0,
            "degraded": False, "question": "TOP_SECRET_QUESTION", "reference": "TOP_SECRET_REFERENCE",
            "evidence": ["TOP_SECRET_QUOTE"],
            "answer": {"status": "completed", "content": "TOP_SECRET_ANSWER", "run": {"trace": {
                "usage": {"input_tokens": 5, "model_calls": 1, "calls_with_usage": 1},
            }}},
        }
        report = {
            "status": "completed", "parameters": {"smoke": False}, "cases_sha256": "digest",
            "manifest": {"documents": [{"sha256": "hash", "filename": "TOP_SECRET_NAME"}],
                         "chunk_size": 500, "chunk_overlap": 100},
            "retrieval": [], "qa": [{**base, "configuration": config} for config in ("rag", "agent")],
        }
        left, right = list(export_experiments(report, {}))
        self.assertEqual(left["dataset_id"], right["dataset_id"])
        self.assertEqual(left["results"][0]["row_id"], right["results"][0]["row_id"])
        self.assertNotIn("TOP_SECRET", json.dumps([left, right]))
        unsafe = deepcopy(report)
        unsafe["qa"][0]["elapsed_seconds"] = "TOP_SECRET"
        with self.assertRaises(ValueError):
            list(export_experiments(unsafe, {}))

    def test_anonymous_export_distinguishes_shared_retrieval_timestamps(self):
        """复用召回起点的旧报告仍可导出成两个可配对的实验。"""
        start = "2026-01-01T01:00:00+00:00"
        fusion_end = "2026-01-01T01:00:01+00:00"
        rerank_end = "2026-01-01T01:00:02+00:00"
        row = {
            "id": "R01", "category": "exact", "started_at": start,
            "elapsed_seconds": 1, "degraded": False, "recall_at_4": 1, "mrr_at_4": 1,
        }
        report = {
            "status": "completed", "parameters": {"smoke": False}, "cases_sha256": "digest",
            "manifest": {"documents": [], "chunk_size": 500, "chunk_overlap": 100},
            "retrieval": [
                {**row, "configuration": "hybrid", "ended_at": fusion_end},
                {**row, "configuration": "hybrid_rerank", "ended_at": rerank_end},
            ],
            "qa": [],
        }
        fusion, rerank = list(export_experiments(report, {}))
        self.assertEqual(fusion["results"][0]["row_id"], rerank["results"][0]["row_id"])
        self.assertEqual(rerank["experiment_start_time"], fusion_end)
        self.assertEqual(rerank["results"][0]["actual_outputs"]["elapsed_seconds"], 1)
