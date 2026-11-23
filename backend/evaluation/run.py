"""Run a reproducible, real-provider comparison in disposable storage.

Usage: backend/.venv/Scripts/python.exe backend/evaluation/run.py
Requires requirements-dev.txt for FastAPI's TestClient. No live server required.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from threading import Event
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QA_CONFIGS = ('rag', 'agent')


def publish_document(client, course, filename, data):
    """通过现有入库流程发布评测资料并检查索引状态。"""
    from app.jobs.indexing import process_version

    url = f"/api/courses/{course}/documents"
    response = client.post(url, files={"file": (filename, data)})
    response.raise_for_status()
    document = response.json()
    process_version(document["latest"]["id"], Event())
    response = client.get(url)
    response.raise_for_status()
    published = next(item for item in response.json() if item["id"] == document["id"])
    if not published["usable"]:
        raise RuntimeError("索引失败，请检查资料格式与嵌入服务：" + filename)
    return published


def run_turn(client, base, body, run_timeout):
    """提交一轮问答，返回最终消息和从提交到结束的耗时。"""
    started = time.monotonic()
    response = client.post(base + "/runs", json=body)
    response.raise_for_status()
    run_id = response.json()["id"]
    deadline = started + run_timeout + 30
    while time.monotonic() <= deadline:
        response = client.get(base)
        response.raise_for_status()
        answer = next(
            message for message in response.json()["messages"]
            if message["run_id"] == run_id and message["role"] == "assistant"
        )
        if answer["status"] in ("completed", "failed", "cancelled", "interrupted"):
            return answer, round(time.monotonic() - started, 3)
        time.sleep(.2)
    client.post(base + f"/runs/{run_id}/cancel").raise_for_status()
    raise TimeoutError("Evaluation run exceeded the configured run budget")


def scenario_turns(client, course, questions, modes, timeout):
    """运行同一场景的两种配置，每组独立保存已完成问答。

    Args:
        client: 临时评测应用的 HTTP 客户端。
        course: 临时课程 ID。
        questions: 按对话顺序排列的问题。
        modes: 基线和 Agent 的执行顺序。
        timeout: 单轮问答超时时间。

    Yields:
        配置、轮次下标、回答记录、耗时及起止时间。失败后的追问标为跳过。
    """
    from evaluation.fixed_rag import run_fixed_turn

    for mode in modes:
        history = []
        base = f"/api/courses/{course}/conversations"
        if mode == "agent":
            response = client.post(base, json={})
            response.raise_for_status()
            base = f"{base}/{response.json()['id']}"
        previous_completed = True
        for index, question in enumerate(questions):
            started_at = datetime.now(timezone.utc).isoformat()
            if not previous_completed:
                answer, elapsed = {"status": "skipped"}, None
            elif mode == "agent":
                started = time.monotonic()
                try:
                    answer, elapsed = run_turn(client, base, {"question": question}, timeout)
                except TimeoutError:
                    answer, elapsed = {"status": "timeout"}, round(time.monotonic() - started, 3)
            else:
                answer, elapsed = asyncio.run(run_fixed_turn(course, question, history, timeout))
            previous_completed = answer["status"] == "completed"
            yield mode, index, answer, elapsed, started_at, datetime.now(timezone.utc).isoformat()


def checks(case, answer):
    """对一轮回答运行结构和词组覆盖检查。

    Args:
        case: 包含预期词组、禁出词和缺口要求的用例。
        answer: 包含正文、引用及运行摘要的助手消息。

    Returns:
        自动检查项、词组覆盖率和留待填写的人工评分字段。
    """
    content = answer.get('content', '')
    citations = answer.get('citations', [])
    groups = case.get('expected_groups', [])
    coverage = [any(term.lower() in content.lower() for term in group) for group in groups]
    gap = bool(re.search(r'未找到|未取得足够依据|没有.{0,15}(提供|依据|资料)|未.{0,15}(提供|记载|提及)|资料.{0,12}(不足|未包含|未提供)', content))
    structural = {
        'completed': answer['status'] == 'completed',
        'citation_numbers_registered': not answer.get('run', {}).get('trace', {}).get('unknown_citations'),
        'no_forbidden_marker': all(term not in content for term in case.get('forbidden', [])),
        'gap_when_expected': gap if case.get('expect_gap') else None,
        'has_citation_for_expected_facts': bool(citations) if groups else None,
    }
    return {'automatic': structural, 'term_group_coverage': coverage,
            'coverage_fraction': sum(coverage) / len(coverage) if coverage else None,
            'manual': {'supported_by_evidence_0_to_2': None, 'completeness_0_to_2': None, 'notes': ''}}


def run(args, temporary):
    # Set before importing any app modules. Never use the developer's course database.
    """使用独立临时数据库运行真实模型的 RAG 与 Agent 对照评测。

    Args:
        args: 已解析的用例、重复次数、检索选项和输出路径。
        temporary: 评测专用的数据目录。

    Raises:
        ValueError: 指定了未知用例 ID。
        RuntimeError: 评测资料无法索引或不满足分页条件。
        TimeoutError: 某轮问答超过评测时限。
    """
    os.environ['COURSE_AGENT_DATA_DIR'] = temporary
    os.environ['CHUNK_SIZE'] = '250'
    os.environ['CHUNK_OVERLAP'] = '40'
    sys.path.insert(0, str(ROOT / 'backend'))
    from fastapi.testclient import TestClient
    from app.main import app
    from app.qa.agent import SYSTEM as AGENT_SYSTEM
    from app.core.models import RUN_TIMEOUT, TOP_K, get_active_models
    from app.db.models import Document
    from app.db.session import SessionLocal
    from app.retrieval.reranker import status as reranker_status
    app.state.indexing_enabled = False
    cases = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
    if args.cases:
        requested = set(args.cases.split(','))
        unknown = requested - {case['id'] for case in cases}
        if unknown:
            raise ValueError('Unknown case IDs: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case['id'] in requested]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'schema_version': 3,
        'application_version': app.version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'fixture_only': True,
        'models': {**get_active_models(), 'reranker': reranker_status()},
        'packages': {
            name: importlib.metadata.version(name)
            for name in ('langchain', 'langchain-openai', 'langgraph', 'chromadb')
        },
        'parameters': {
            'top_k': TOP_K, 'chunk_size': 250, 'chunk_overlap': 40,
            'web_search': False, 'semantic': True, 'repeats': args.repeats,
        },
        'protocol': {
            'cases_sha256': hashlib.sha256((HERE / 'cases.json').read_bytes()).hexdigest(),
            'agent_prompt_sha256': hashlib.sha256(AGENT_SYSTEM.encode()).hexdigest(),
            'timing': {'rag': 'in_process', 'agent': 'http_polling'},
        },
        'documents': [], 'results': [], 'status': 'running',
        'limitations': ['自动词组覆盖仅衡量预期词组是否出现。', '回答质量等待人工证据评分。'],
    }

    def save():
        """将当前进度和结果写入 JSON 报告。"""
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    with TestClient(app) as client:
        course = client.post('/api/courses', json={'name': '隔离评估：操作系统实验'}).json()['id']
        other = client.post('/api/courses', json={'name': '另一门隔离课程'}).json()['id']
        for path in sorted((HERE / 'fixtures').glob('*.md')):
            data = path.read_bytes()
            published = publish_document(client, course, path.name, data)
            if path.name == '实验手册.md' and published['chunk_count'] <= 4:
                raise RuntimeError('Long-document fixture must exceed one read_document call (4 chunks)')
            report['documents'].append({'name': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                                        'version_id': published['active_version_id'], 'chunks': published['chunk_count'],
                                        'embedding_profile': published['latest']['embedding_profile']})
        publish_document(client, other, '私有口令.md', '另一门课程的虚构私有核验口令是ORCHID-908。'.encode())
        save()
        for repeat in range(args.repeats):
            for case_index, case in enumerate(cases):
                # Alternate order to reduce consistent warm-cache / provider-load bias.
                modes = QA_CONFIGS if (repeat + case_index) % 2 == 0 else QA_CONFIGS[::-1]
                for mode, turn_index, answer, elapsed, _, _ in scenario_turns(
                    client, course, case['turns'], modes, RUN_TIMEOUT,
                ):
                    with SessionLocal() as session:
                        scoped = all(
                            (source := session.get(Document, item['document_id'])) is not None
                            and source.course_id == course
                            for item in answer.get('citations', [])
                        )
                    evaluation = checks(case if turn_index == len(case['turns']) - 1 else {}, answer)
                    evaluation['automatic']['citations_belong_to_course'] = scoped
                    tools = answer.get('run', {}).get('trace', {}).get('tools', [])
                    report['results'].append({
                        'case_id': case['id'], 'category': case['category'], 'repeat': repeat + 1,
                        'mode': mode, 'turn': turn_index + 1, 'question': case['turns'][turn_index],
                        'elapsed_seconds': elapsed, 'tool_calls': len(tools),
                        'answer': answer, 'evaluation': evaluation,
                    })
                    save()
                    print(f"{case['id']} / {mode} / turn {turn_index + 1}: {answer['status']}", flush=True)
        report['status'] = 'completed'
        save()
    print('Saved report: ' + str(args.output), flush=True)


def main():
    """解析命令行选项并在临时数据目录中启动对照评测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', help='Comma-separated case IDs; default all eight cases')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--output', type=Path, default=ROOT / '.cache' / 'evaluation.json')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='course-agent-eval-') as temporary:
        run(args, temporary)


if __name__ == '__main__':
    main()
