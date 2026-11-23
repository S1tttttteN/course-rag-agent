"""运行隔离的真实课程检索和问答实验。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic
from uuid import uuid4

BACKEND = Path(__file__).resolve().parents[1]
if __package__ is None:
    sys.path.insert(0, str(BACKEND))

from evaluation.run import QA_CONFIGS, scenario_turns
from evaluation.course_dataset import (
    freeze,
    load_cases,
    read_json,
    resolve_evidence,
    snapshot_file,
    write_json,
)


ROOT = BACKEND.parent
WORK = ROOT / ".cache" / "os-benchmark"
RETRIEVAL_TOP_K = 4


def utc_now():
    """返回供实验上传使用的 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def disable_tracing():
    """在导入应用和 SDK 之前为本进程关闭 LangSmith 自动追踪。"""
    for key in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ[key] = "false"


def check_dataset(retrieval, qa, chunks):
    """在任何生成模型调用之前验证所有标注能定位到冻结原文。"""
    evidence = {}
    items = [(case["id"], case) for case in retrieval]
    items.extend(
        (f"{case['id']}/{index}", turn)
        for case in qa for index, turn in enumerate(case["turns"], 1)
    )
    for identifier, item in items:
        try:
            evidence[identifier] = resolve_evidence(item["evidence"], chunks)
        except ValueError as error:
            raise ValueError(f"{identifier}：{error}") from error
    return evidence


def inspect_snapshot(snapshot, manifest):
    """用应用的分块器在本地验证冻结资料，无需调用模型或嵌入服务。"""
    from app.retrieval.loader import load_and_split

    chunks = {}
    for document in manifest["documents"]:
        key = document["key"]
        parts = load_and_split(
            snapshot_file(snapshot, document), document["filename"],
            manifest["chunk_size"], manifest["chunk_overlap"],
        )
        chunks[key] = [
            {"id": f"{key}/{index}", "text": part.page_content, "page": part.metadata.get("page")}
            for index, part in enumerate(parts)
        ]
    return chunks


def publish_snapshot(client, snapshot, manifest):
    """将冻结原件发布到一次性评测库并读取实际入库的片段。"""
    from app.db.session import SessionLocal
    from app.repositories.documents import active_chunks
    from evaluation.run import publish_document

    response = client.post("/api/courses", json={"name": "离线课程评测"})
    response.raise_for_status()
    course = response.json()["id"]
    keys = {}
    for document in manifest["documents"]:
        data = snapshot_file(snapshot, document).read_bytes()
        result = publish_document(client, course, document["filename"], data)
        keys[result["active_version_id"]] = document["key"]
    chunks = {key: [] for key in keys.values()}
    with SessionLocal() as session:
        for chunk, _, version in active_chunks(session, course):
            chunks[keys[version.id]].append({"id": chunk.id, "text": chunk.text, "page": chunk.page})
    if sum(map(len, chunks.values())) <= 20:
        raise ValueError("课程片段数必须超过单路召回上限 20。")
    return course, chunks


def retrieval_scores(result_ids, groups):
    """计算必需证据组的 Recall@4 与首个有效片段的 MRR@4。"""
    top = result_ids[:RETRIEVAL_TOP_K]
    hits = set(top)
    first = next(
        (rank for rank, identifier in enumerate(top, 1) if any(identifier in group for group in groups)),
        None,
    )
    return sum(bool(hits & group) for group in groups) / len(groups), 1 / first if first else 0.0


def run_retrieval_case(course, case, groups, case_index):
    """实跑三路召回，并让融合与重排组共享同一排序前候选。"""
    from app.retrieval.search import finish_search, rank_candidates

    configurations = (("bm25", False, True), ("vector", True, False), ("hybrid", True, True))
    query = case["question"]
    for config, semantic, keyword in configurations[case_index % 3:] + configurations[:case_index % 3]:
        start_time = utc_now()
        started = monotonic()
        ranked, trace = rank_candidates(course, query, semantic=semantic, keyword=keyword)
        recall_seconds = monotonic() - started
        for rerank in ((False, True) if config == "hybrid" else (False,)):
            phase_start = utc_now() if rerank else start_time
            finishing = monotonic()
            result = finish_search(course, query, ranked, trace, RETRIEVAL_TOP_K, rerank)
            elapsed = round(recall_seconds + monotonic() - finishing, 4)
            stopped = utc_now()
            actual = result["trace"]
            degraded = (
                (semantic and not actual["semantic"])
                or (keyword and not actual["keyword"])
                or (rerank and bool(ranked) and not actual["reranked"])
            )
            recall, mrr = retrieval_scores([row["id"] for row in result["results"]], groups)
            yield {
                "id": case["id"], "category": case["category"],
                "configuration": "hybrid_rerank" if rerank else config,
                "question": query, "evidence": case["evidence"],
                "evidence_chunk_ids": [sorted(group) for group in groups],
                "started_at": phase_start, "ended_at": stopped, "elapsed_seconds": elapsed,
                "recall_at_4": recall, "mrr_at_4": mrr, "degraded": degraded, **result,
            }


def run_qa_case(client, course, case, repeat, offset, timeout):
    """用相同检索策略和独立上下文运行两组问答。"""
    from app.retrieval.reranker import is_available

    rerank = is_available()
    modes = QA_CONFIGS[offset:] + QA_CONFIGS[:offset]
    turns = case["turns"]
    rows = []
    for mode, index, answer, elapsed, started, ended in scenario_turns(
        client, course, [turn["question"] for turn in turns], modes, timeout,
    ):
        trace = answer.get("run", {}).get("trace", {})
        if mode == "agent":
            tools = [tool for tool in trace.get("tools", []) if tool["name"] == "search_course"]
            degraded = any(
                tool["status"] != "completed" or retrieval_degraded(tool.get("retrieval", {}), rerank)
                for tool in tools
            ) if tools else None
        else:
            degraded = retrieval_degraded(trace, rerank) if "semantic" in trace else None
        rows.append({
            "id": case["id"], "category": case["category"], "configuration": mode,
            "repeat": repeat, "turn": index + 1, **turns[index], "answer": answer,
            "degraded": degraded, "started_at": started, "ended_at": ended, "elapsed_seconds": elapsed,
        })
    return rows


def retrieval_degraded(trace, rerank):
    """根据执行记录识别有候选时的向量或重排降级。"""
    return bool(trace.get("candidates")) and (
        not trace.get("semantic") or (rerank and not trace.get("reranked"))
    )


def run_experiment(args, manifest, retrieval, qa, temporary):
    """发布资料一次，成对运行各组并持续保存本地原始报告。"""
    os.environ["COURSE_AGENT_DATA_DIR"] = temporary
    os.environ["CHUNK_SIZE"] = str(manifest["chunk_size"])
    os.environ["CHUNK_OVERLAP"] = str(manifest["chunk_overlap"])
    from fastapi.testclient import TestClient
    from langsmith import tracing_context
    from app.core.models import RUN_TIMEOUT, TOP_K, embedding_profile, get_active_models
    from app.main import app
    from app.retrieval.reranker import status as reranker_status

    if embedding_profile() != manifest["embedding_profile"]:
        raise ValueError("当前嵌入配置与冻结清单不一致。")
    app.state.indexing_enabled = False
    if args.smoke:
        retrieval = retrieval[:1]
        qa = [next(case for case in qa if case["category"] == "followup")]
    repeats = 1 if args.smoke else args.repeats
    reranker = reranker_status()
    report = {
        "status": "running", "run_id": uuid4().hex, "application_version": app.version,
        "manifest": manifest,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "models": {**get_active_models(), "reranker": reranker},
        "parameters": {
            "repeats": repeats, "workers": args.workers, "top_k": TOP_K, "smoke": args.smoke,
            "retrieval_top_k": RETRIEVAL_TOP_K, "web_search": False,
            "qa": {
                mode: {"semantic": True, "rerank": reranker["enabled"]} for mode in QA_CONFIGS
            },
            "timing": {"rag": "in_process", "agent": "http_polling"},
        },
        "retrieval": [], "qa": [],
    }
    with tracing_context(enabled=False), TestClient(app) as client:
        course, chunks = publish_snapshot(client, args.snapshot, manifest)
        gold = check_dataset(retrieval, qa, chunks)
        report["indexed_chunks"] = {key: len(values) for key, values in chunks.items()}
        write_json(args.output, report)
        for index, case in enumerate(retrieval):
            report["retrieval"].extend(run_retrieval_case(course, case, gold[case["id"]], index))
            write_json(args.output, report)
            print(f"retrieval {index + 1}/{len(retrieval)}", flush=True)

        jobs = [
            (case, repeat, (index + repeat - 1) % len(QA_CONFIGS))
            for repeat in range(1, repeats + 1) for index, case in enumerate(qa)
        ]

        def evaluate(job):
            """为一个场景及重复序号运行所有问答配置。"""
            return run_qa_case(client, course, *job, RUN_TIMEOUT)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, rows in enumerate(pool.map(evaluate, jobs), 1):
                report["qa"].extend(rows)
                write_json(args.output, report)
                completed = sum(row["answer"]["status"] == "completed" for row in rows)
                print(f"qa {index}/{len(jobs)} ({completed}/{len(rows)} turns completed)", flush=True)
    report["status"] = "completed"
    write_json(args.output, report)


def main():
    """提供快照、标注校验和实际评测三个入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run"))
    parser.add_argument("--course", default="操作系统")
    parser.add_argument("--snapshot", type=Path, default=WORK / "snapshot")
    parser.add_argument("--cases", type=Path, default=WORK / "cases.json")
    parser.add_argument("--output", type=Path, default=WORK / "raw.json")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    disable_tracing()
    if args.command == "freeze":
        manifest = freeze(args.course, args.snapshot)
        print(f"Frozen documents: {len(manifest['documents'])}")
        return
    manifest = read_json(args.snapshot / "manifest.json")
    retrieval, qa = load_cases(args.cases)
    chunks = inspect_snapshot(args.snapshot, manifest)
    check_dataset(retrieval, qa, chunks)
    print(f"Validated {len(retrieval)} retrieval questions and {len(qa)} QA cases.")
    if args.command == "run":
        with tempfile.TemporaryDirectory(prefix="real-course-eval-") as temporary:
            run_experiment(args, manifest, retrieval, qa, temporary)


if __name__ == "__main__":
    main()
