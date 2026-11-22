"""生成盲评包、本地对照报告和 LangSmith 匿名实验。"""

import argparse
from collections import defaultdict
from datetime import datetime
import math
from pathlib import Path
import random
import re
from statistics import mean, median
import sys
from uuid import NAMESPACE_URL, uuid4, uuid5

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.course_dataset import QA_CATEGORIES, RETRIEVAL_CATEGORIES, read_json, write_json
from evaluation.real_course import QA_CONFIGS, ROOT, WORK


RETRIEVAL_CONFIGS = ("bm25", "vector", "hybrid", "hybrid_rerank")
SCORE_FIELDS = ("completeness", "citation_support", "gap_correct")


def row_key(row):
    """返回同一配置下唯一的评测记录键。"""
    return (row["configuration"], row["id"], row.get("repeat", 1), row.get("turn", 1))


def prepare_review(report, folder):
    """打乱已完成回答并生成匿名评分表和独立的本地对照键。"""
    review_file = folder / "review.json"
    key_file = folder / "review-key.json"
    if review_file.exists() or key_file.exists():
        raise ValueError("盲评文件已存在。请使用新目录，以保留已填评分。")
    items = []
    mapping = {}
    questions = defaultdict(list)
    for row in report["qa"]:
        scenario = row_key(row)[:3]
        previous = questions[scenario]
        if row["answer"]["status"] == "completed":
            blind = uuid4().hex
            mapping[blind] = row_key(row)
            items.append({
                "review_id": blind, "previous_turns": previous.copy(),
                "question": row["question"], "reference": row["reference"],
                "evidence": row["evidence"], "answer": row["answer"]["content"],
                "citations": row["answer"]["citations"],
                "completeness": None, "citation_support": None, "gap_correct": None,
                "gap": row.get("gap"), "notes": "",
            })
        previous.append({"question": row["question"], "answer": row["answer"].get("content", "")})
    random.SystemRandom().shuffle(items)
    write_json(key_file, {"run_id": report["run_id"], "keys": mapping})
    write_json(review_file, items)


def read_scores(folder, run_id):
    """读取盲评打分并验证分值与本地对照键。"""
    if folder is None:
        return {}
    mapping = read_json(folder / "review-key.json")
    if mapping["run_id"] != run_id:
        raise ValueError("盲评结果属于另一份原始评测报告。")
    keys = mapping["keys"]
    scores = {}
    for item in read_json(folder / "review.json"):
        values = {key: item[key] for key in SCORE_FIELDS if item[key] is not None}
        for key, value in values.items():
            valid = type(value) is int and value in ((0, 1) if key == "gap_correct" else (0, 1, 2))
            if not valid or (key == "gap_correct" and not item["gap"]):
                raise ValueError(f"无效评分：{item['review_id']} / {key}")
        scores[tuple(keys[item["review_id"]])] = values
    return scores


def metric(values, aggregate=mean):
    """呈现已采集的数值和样本量，缺失时显示待评。"""
    values = [float(value) for value in values if value is not None]
    return f"{aggregate(values):.3f}（{len(values)}）" if values else "—"


def grouped(rows, categories, configs):
    """逐题型提供各配置的记录，并附带总体行。"""
    for category in ("all", *categories):
        group = {config: [] for config in configs}
        for row in rows:
            if category == "all" or row["category"] == category:
                group[row["configuration"]].append(row)
        if any(group.values()):
            yield category, group


def qa_usage(row):
    """获取模型实际返回的用量字段，缺失时保留为空。"""
    usage = row["answer"].get("run", {}).get("trace", {}).get("usage", {})
    if usage.get("calls_with_usage") != usage.get("model_calls"):
        return {"model_calls": usage.get("model_calls")}
    return usage


def render_report(report, scores):
    """按题型生成两张对照表及实际观察到的可复核记录。"""
    documents = report["manifest"]["documents"]
    lines = [
        "# 操作系统课程评测", "",
        f"- 状态：{report['status']}。资料 {len(documents)} 份，评测索引 {sum(report['indexed_chunks'].values())} 个片段。",
        f"- 应用版本：{report['application_version']}。题库 SHA-256：`{report['cases_sha256']}`。",
        f"- 分块：{report['manifest']['chunk_size']} / {report['manifest']['chunk_overlap']}。"
        f" 重复：{report['parameters']['repeats']}。问答并发：{report['parameters']['workers']}。",
        f"- 对话模型：{report['models']['chat_model']}；嵌入模型：{report['models']['embedding_model']}；"
        f"重排模型：{report['models']['reranker'].get('model', '未指定')}。",
        "- 资料版本与 SHA-256 见原始报告 `manifest.documents`。每个分数后括号为有效样本数。", "",
        "## 检索对照", "",
        "| 题型 | 配置 | 有效 / 总数 | 降级数 | Recall@4 | MRR@4 | 耗时中位数（秒） |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category, group in grouped(report["retrieval"], RETRIEVAL_CATEGORIES, RETRIEVAL_CONFIGS):
        paired = set.intersection(*({row["id"] for row in rows if not row["degraded"]} for rows in group.values()))
        for config, rows in group.items():
            valid = [row for row in rows if not row["degraded"]]
            comparable = [row for row in valid if row["id"] in paired]
            lines.append(
                f"| {category} | {config} | {len(valid)} / {len(rows)} | {len(rows) - len(valid)} | "
                f"{metric(row['recall_at_4'] for row in comparable)} | {metric(row['mrr_at_4'] for row in comparable)} | "
                f"{metric((row['elapsed_seconds'] for row in comparable), median)} |"
            )
    lines.extend([
        "", "检索对照使用四组均正常执行的题。重排耗时包括相同候选池的召回时间。", "",
        "## 问答对照", "",
        "| 题型 | 配置 | 完成 / 总数 | 超时 | 检索降级 / 有记录 | 完整性 /2 | 引用支撑 /2 | 缺口通过率 | 耗时中位数（秒） | 平均模型调用 | 平均输入 / 输出 Token |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    qa_configs = tuple(config for config, _, _ in QA_CONFIGS)
    for category, group in grouped(report["qa"], QA_CATEGORIES, qa_configs):
        for config, rows in group.items():
            completed = [row for row in rows if row["answer"]["status"] == "completed"]
            timeout = sum(
                row["answer"]["status"] == "timeout"
                or "超时" in (row["answer"].get("run", {}).get("error") or "")
                for row in rows
            )
            judged = [scores.get(row_key(row), {}) for row in completed]
            tokens = [qa_usage(row) for row in completed]
            successful = [row for row in completed if row["degraded"] is not True]
            good_judged = [scores.get(row_key(row), {}) for row in successful]
            lines.append(
                f"| {category} | {config} | {len(completed)} / {len(rows)} | {timeout} | "
                f"{sum(row['degraded'] is True for row in rows)} / {sum(row['degraded'] is not None for row in rows)} | "
                f"{metric(score.get('completeness') for score in good_judged)} | "
                f"{metric(score.get('citation_support') for score in good_judged)} | "
                f"{metric(score.get('gap_correct') for score in judged)} | "
                f"{metric((row['elapsed_seconds'] for row in successful), median)} | "
                f"{metric(value.get('model_calls') for value in tokens)} | "
                f"{metric(value.get('input_tokens') for value in tokens)} / {metric(value.get('output_tokens') for value in tokens)} |"
            )
    lines.extend(["", "未填写的盲评分数保持为空。补齐同题同轮评分后可做配对质量对照。", ""])
    for left, right in (("rag", "agent"), ("rag", "rag_rerank")):
        left_scores = {key[1:]: value for key, value in scores.items() if key[0] == left}
        right_scores = {key[1:]: value for key, value in scores.items() if key[0] == right}
        comparable = {
            row_key(row) for row in report["qa"]
            if row["answer"]["status"] == "completed" and row["degraded"] is not True
        }
        shared = [
            key for key in left_scores.keys() & right_scores.keys()
            if (left, *key) in comparable and (right, *key) in comparable
        ]
        deltas = {
            field: metric(
                right_scores[key][field] - left_scores[key][field]
                for key in shared if field in left_scores[key] and field in right_scores[key]
            )
            for field in SCORE_FIELDS[:2]
        }
        lines.append(
            f"- {right} 相对 {left} 的配对评分变化。完整性 {deltas['completeness']}，"
            f"引用支撑 {deltas['citation_support']}。"
        )
    lines.extend(["", "## 复核记录", ""])
    lines.extend(case_examples(report))
    return "\n".join(lines) + "\n"


def case_examples(report):
    """按观测差异选出可在本地原始报告核对的三个场景。"""
    retrieval = defaultdict(dict)
    for row in report["retrieval"]:
        if not row["degraded"]:
            retrieval[row["id"]][row["configuration"]] = row
    pairs = [
        (rows["bm25"], rows["hybrid_rerank"])
        for rows in retrieval.values() if {"bm25", "hybrid_rerank"} <= rows.keys()
    ]
    if pairs:
        base, reranked = max(pairs, key=lambda pair: pair[1]["recall_at_4"] - pair[0]["recall_at_4"])
        yield f"- 检索差异 · `{base['id']}`，BM25 和融合重排的 Recall@4 分别为 {base['recall_at_4']:.2f} 和 {reranked['recall_at_4']:.2f}。"
    agent = []
    for row in report["qa"]:
        if row["configuration"] != "agent" or row["answer"]["status"] != "completed":
            continue
        tools = [
            tool["name"] for tool in row["answer"]["run"]["trace"].get("tools", [])
            if tool["status"] == "completed"
        ]
        if tools:
            agent.append((row, tools))
    if agent:
        row, tools = max(agent, key=lambda pair: ("read_document" in pair[1], len(pair[1])))
        yield f"- Agent 工具行为 · `{row['id']}`，重复 {row['repeat']}、第 {row['turn']} 轮调用了 {', '.join(tools)}。可与固定 RAG 同轮引用和盲评对照。"
    if pairs:
        base, reranked = min(pairs, key=lambda pair: pair[1]["recall_at_4"] - pair[0]["recall_at_4"])
        yield f"- 收益下界 · `{base['id']}`，BM25 和融合重排的 Recall@4 分别为 {base['recall_at_4']:.2f} 和 {reranked['recall_at_4']:.2f}。"


def export_experiments(report, scores):
    """从原始报告重建允许出站的字段，生成同题对齐的 LangSmith 请求。"""
    if report["status"] != "completed" or report["parameters"]["smoke"]:
        raise ValueError("仅导出已完成的正式评测。")
    manifest = report["manifest"]
    material = "|".join(document["sha256"] for document in manifest["documents"])
    protocol = f"{report['cases_sha256']}:{material}:{manifest['chunk_size']}:{manifest['chunk_overlap']}"
    hybrid_timing = {
        row["id"]: (row["started_at"], row["ended_at"])
        for row in report["retrieval"] if row["configuration"] == "hybrid"
    }
    for kind, categories, configs in (
        ("retrieval", RETRIEVAL_CATEGORIES, RETRIEVAL_CONFIGS),
        ("qa", QA_CATEGORIES, tuple(config for config, _, _ in QA_CONFIGS)),
    ):
        dataset = uuid5(NAMESPACE_URL, f"course-evaluation:{kind}:{protocol}")
        for config in configs:
            results = [
                anonymous_row(row, scores, dataset, categories, kind)
                for row in report[kind] if row["configuration"] == config
            ]
            if config == "hybrid_rerank":
                for result in results:
                    original = hybrid_timing.get(result["inputs"]["case_id"])
                    # 旧报告的两组共享召回起点。用初排结束时间定位重排阶段，
                    # 避免 LangSmith 将同题同起点视为重复样本版本。
                    if original and result["start_time"] == datetime.fromisoformat(original[0]).isoformat():
                        result["start_time"] = datetime.fromisoformat(original[1]).isoformat()
            if not results:
                continue
            start = min(row["start_time"] for row in results)
            end = max(row["end_time"] for row in results)
            yield {
                "dataset_id": str(dataset),
                "dataset_name": f"course-{kind}-{str(dataset)[:8]}",
                "experiment_name": f"{config}-{start.replace(':', '')}",
                "experiment_start_time": start, "experiment_end_time": end,
                "results": results,
            }


def anonymous_row(row, scores, dataset, categories, kind):
    """将单条记录转换为匿名样本，只复制经类型校验的指标。"""
    config, case, repeat, turn = row_key(row)
    prefix = "R" if kind == "retrieval" else "Q"
    if not re.fullmatch(prefix + r"[0-9]{2,4}", case) or row["category"] not in categories:
        raise ValueError("LangSmith 匿名导出要求使用规范题号和题型。")
    if any(type(value) is not int or not 1 <= value <= 100 for value in (repeat, turn)):
        raise ValueError("重复序号和轮次必须为有效整数。")
    identity = {"case_id": case, "category": row["category"], "repeat": repeat, "turn": turn}
    metrics = {"elapsed_seconds": row["elapsed_seconds"], "degraded": row["degraded"]}
    if kind == "retrieval":
        if not row["degraded"]:
            metrics.update(recall_at_4=row["recall_at_4"], mrr_at_4=row["mrr_at_4"])
    else:
        metrics["completed"] = row["answer"]["status"] == "completed"
        usage = qa_usage(row)
        for key in ("input_tokens", "output_tokens", "model_calls"):
            metrics[key] = usage.get(key)
        for key in SCORE_FIELDS:
            metrics[key] = scores.get(row_key(row), {}).get(key)
    numeric = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if type(value) not in (int, float, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"指标不是有效数值：{key}")
        numeric[key] = round(float(value), 4)
    return {
        "row_id": str(uuid5(dataset, f"{case}:{repeat}:{turn}")),
        "inputs": identity, "actual_outputs": {"configuration": config, **numeric},
        "evaluation_scores": [{"key": key, "score": value} for key, value in numeric.items()],
        "start_time": datetime.fromisoformat(row["started_at"]).isoformat(),
        "end_time": datetime.fromisoformat(row["ended_at"]).isoformat(),
        "run_name": config,
    }


def main():
    """输出盲评文件、可读对照表或显式上传匿名实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review", "report", "export", "upload"))
    parser.add_argument("--raw", type=Path, default=WORK / "raw.json")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = read_json(args.raw)
    if args.command == "review":
        prepare_review(report, args.reviews or WORK / "blind")
        return
    scores = read_scores(args.reviews, report["run_id"])
    if args.command == "report":
        output = args.output or WORK / "report.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_report(report, scores), encoding="utf-8")
        return
    experiments = list(export_experiments(report, scores))
    if args.command == "export":
        write_json(args.output or WORK / "langsmith-anonymous.json", experiments)
        return

    from dotenv import load_dotenv
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError

    load_dotenv(ROOT / ".env")
    client = Client(auto_batch_tracing=False)
    for experiment in experiments:
        name = experiment["experiment_name"]
        try:
            project = client.read_project(project_name=name, include_stats=True)
        except LangSmithNotFoundError:
            project = None
        if project is not None:
            count = project.run_count
            if count != len(experiment["results"]):
                raise RuntimeError(f"LangSmith 中同名实验的运行数不一致：{name}（{count}）")
            print(f"Already uploaded {name}", flush=True)
            continue
        client.request_with_retries("POST", "/datasets/upload-experiment", json=experiment)
        print(f"Uploaded {name}", flush=True)


if __name__ == "__main__":
    main()
