"""覆盖课程评测的证据计分、会话隔离和匿名导出。"""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

import test_courses  # 初始化测试专用数据目录。
from app.retrieval.search import rank_candidates
from evaluation.course_dataset import resolve_evidence
from evaluation.course_report import export_experiments
from evaluation.real_course import retrieval_scores, run_qa_case, run_retrieval_case


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
        """每种配置新建会话，追问留在同一个会话内。"""
        client = Mock()
        client.post.side_effect = [
            Mock(**{"json.return_value": {"id": f"session{index}"}}) for index in range(3)
        ]
        turn = {"question": "为什么？", "reference": "参考", "evidence": []}
        case = {"id": "Q01", "category": "followup", "turns": [turn, turn]}
        answer = {"status": "completed", "run": {"trace": {}}}
        with patch("evaluation.run.run_turn", return_value=(answer, 1)) as run:
            run_qa_case(client, "course", case, 1, 0, 2)
        paths = [call.args[1] for call in run.call_args_list]
        self.assertEqual(paths[::2], paths[1::2])
        self.assertEqual(len(set(paths)), 3)

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
