"""Run a reproducible, real-provider comparison in disposable storage.

Usage: backend/.venv/Scripts/python.exe backend/evaluation/run.py
Requires requirements-dev.txt for FastAPI's TestClient. No live server required.
"""
import argparse
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


def checks(case, answer):
    content = answer['content']
    citations = answer['citations']
    groups = case.get('expected_groups', [])
    coverage = [any(term.lower() in content.lower() for term in group) for group in groups]
    gap = bool(re.search(r'未找到|未取得足够依据|没有.{0,15}(提供|依据|资料)|未.{0,15}(提供|记载|提及)|资料.{0,12}(不足|未包含|未提供)', content))
    structural = {
        'completed': answer['status'] == 'completed',
        'citation_numbers_registered': not answer['run']['trace'].get('unknown_citations'),
        'no_forbidden_marker': all(term not in content for term in case.get('forbidden', [])),
        'gap_when_expected': gap if case.get('expect_gap') else None,
        'has_citation_for_expected_facts': bool(citations) if groups else None,
    }
    return {'automatic': structural, 'term_group_coverage': coverage,
            'coverage_fraction': sum(coverage) / len(coverage) if coverage else None,
            'manual': {'supported_by_evidence_0_to_2': None, 'completeness_0_to_2': None, 'notes': ''}}


def run(args, temporary):
    # Set before importing any app modules. Never use the developer's course database.
    os.environ['COURSE_AGENT_DATA_DIR'] = temporary
    os.environ['CHUNK_SIZE'] = '250'
    os.environ['CHUNK_OVERLAP'] = '40'
    sys.path.insert(0, str(ROOT / 'backend'))
    from fastapi.testclient import TestClient
    from app.main import app
    from app.qa.evidence import SYSTEM as REVIEW_SYSTEM, REVIEW_TIMEOUT, review_options
    from app.qa.agent import SYSTEM as AGENT_SYSTEM
    from app.core.models import get_active_models, get_llm, RERANK_PROVIDER, RERANK_MODEL, RERANK_PROVIDERS, ANSWER_REVIEW, WEB_SEARCH_ENABLED, RUN_TIMEOUT
    from app.db.session import SessionLocal
    from app.jobs.indexing import process_version
    app.state.indexing_enabled = False
    cases = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
    if args.cases:
        requested = set(args.cases.split(','))
        unknown = requested - {case['id'] for case in cases}
        if unknown:
            raise ValueError('Unknown case IDs: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case['id'] in requested]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 2, 'application_version': app.version, 'created_at': datetime.now(timezone.utc).isoformat(),
              'fixture_only': True, 'models': {**get_active_models(), 'rerank_provider': RERANK_PROVIDER,
                  'rerank_model': RERANK_MODEL or RERANK_PROVIDERS.get(RERANK_PROVIDER, {}).get('default_model', '')},
              'packages': {name: importlib.metadata.version(name) for name in ('langchain', 'langchain-openai', 'langgraph', 'chromadb')},
              'parameters': {'top_k': 4, 'chunk_size': 250, 'chunk_overlap': 40, 'evidence_review': ANSWER_REVIEW, 'web_search': WEB_SEARCH_ENABLED,
                  'semantic': not args.keyword_only, 'rerank': not args.no_rerank, 'repeats': args.repeats},
              'protocol': {'cases_sha256': hashlib.sha256((HERE / 'cases.json').read_bytes()).hexdigest(),
                  'agent_prompt_sha256': hashlib.sha256(AGENT_SYSTEM.encode()).hexdigest(),
                  'review_prompt_sha256': hashlib.sha256(REVIEW_SYSTEM.encode()).hexdigest(), 'review_timeout_seconds': REVIEW_TIMEOUT,
                  'review_request_options': review_options(get_llm())},
              'documents': [], 'results': [], 'status': 'running',
              'limitations': ['自动词组覆盖不是事实正确率。', '尚未填写人工证据评分，不能据此宣称Agent优于RAG。', '用量仅记录服务商返回数据，不估计费用。']}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    with TestClient(app) as client:
        course = client.post('/api/courses', json={'name': '隔离评估：操作系统实验'}).json()['id']
        other = client.post('/api/courses', json={'name': '另一门隔离课程'}).json()['id']
        for path in sorted((HERE / 'fixtures').glob('*.md')):
            data = path.read_bytes()
            response = client.post(f'/api/courses/{course}/documents', files={'file': (path.name, data)})
            response.raise_for_status()
            document = response.json()
            process_version(document['latest']['id'], Event())
            state = client.get(f'/api/courses/{course}/documents').json()
            published = next(item for item in state if item['id'] == document['id'])
            if not published['usable']:
                raise RuntimeError('Fixture indexing failed: ' + path.name + '; check embedding configuration.')
            if path.name == '实验手册.md' and published['chunk_count'] <= 4:
                raise RuntimeError('Long-document fixture must exceed one read_document call (4 chunks)')
            report['documents'].append({'name': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                                        'version_id': published['active_version_id'], 'chunks': published['chunk_count'],
                                        'embedding_profile': published['latest']['embedding_profile']})
        foreign = client.post(f'/api/courses/{other}/documents', files={'file': ('私有口令.md', '另一门课程的虚构私有核验口令是ORCHID-908。'.encode())}).json()
        process_version(foreign['latest']['id'], Event())
        foreign_state = client.get(f'/api/courses/{other}/documents').json()[0]
        if not foreign_state['usable']:
            raise RuntimeError('Foreign-course isolation fixture could not be indexed')
        save()
        for repeat in range(args.repeats):
            for case_index, case in enumerate(cases):
                # Alternate order to reduce consistent warm-cache / provider-load bias.
                modes = ('rag', 'agent') if (repeat + case_index) % 2 == 0 else ('agent', 'rag')
                for mode in modes:
                    conversation = client.post(f'/api/courses/{course}/conversations', json={}).json()['id']
                    base = f'/api/courses/{course}/conversations/{conversation}'
                    for turn_index, question in enumerate(case['turns']):
                        started = time.monotonic()
                        response = client.post(base + '/runs', json={'question': question, 'mode': mode, 'top_k': 4,
                            'semantic': not args.keyword_only, 'rerank': not args.no_rerank})
                        response.raise_for_status()
                        run_id = response.json()['id']
                        deadline = started + RUN_TIMEOUT + 30
                        while True:
                            detail = client.get(base).json()
                            answer = next(message for message in detail['messages'] if message['run_id'] == run_id and message['role'] == 'assistant')
                            if answer['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                                break
                            if time.monotonic() > deadline:
                                client.post(base + f'/runs/{run_id}/cancel').raise_for_status()
                                raise TimeoutError('Evaluation run exceeded the configured run budget')
                            time.sleep(.2)
                        with SessionLocal() as session:
                            from app.db.models import Document
                            scoped = all(
                                (bool(item.get('url')) if item.get('kind') in {'web_snippet', 'web_page'} else
                                 ((source := session.get(Document, item['document_id'])) is not None and source.course_id == course))
                                for item in answer['citations'])
                        evaluation = checks(case if turn_index == len(case['turns']) - 1 else {}, answer)
                        evaluation['automatic']['citations_belong_to_course'] = scoped
                        item = {'case_id': case['id'], 'category': case['category'], 'repeat': repeat + 1, 'mode': mode,
                                'turn': turn_index + 1, 'question': question, 'elapsed_seconds': round(time.monotonic() - started, 3),
                                'tool_calls': len(answer['run']['trace'].get('tools', [])), 'answer': answer, 'evaluation': evaluation}
                        report['results'].append(item)
                        save()
                        print(f"{case['id']} / {mode} / turn {turn_index + 1}: {answer['status']} ({item['elapsed_seconds']}s, tools={item['tool_calls']})", flush=True)
        report['status'] = 'completed'
        save()
    print('Saved report: ' + str(args.output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', help='Comma-separated case IDs; default all eight cases')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--keyword-only', action='store_true', help='Disable embedding queries; initial indexing still uses configured embeddings')
    parser.add_argument('--no-rerank', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache' / 'evaluation.json')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='course-agent-eval-') as temporary:
        run(args, temporary)


if __name__ == '__main__':
    main()
