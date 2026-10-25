"""Render a comparison report without claiming keyword checks are correctness scores."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    lines = ['# RAG / Agent 对照记录', '', f"数据时间：{report['created_at']}；运行状态：{report['status']}。", '',
             '使用仓库内虚构示例资料和独立临时数据库；结果不代表真实课程全集或其他模型表现。', '',
             '## 环境', '', '```json', json.dumps({'models': report['models'], 'parameters': report['parameters'], 'packages': report['packages']}, ensure_ascii=False, indent=2), '```', '',
             '## 汇总', '', '| 模式 | 完成轮数 | 平均耗时 | 中位耗时 | 平均工具次数 | 模型调用总数 | 已报告输入 / 输出 token |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('rag', 'agent'):
        rows = [row for row in report['results'] if row['mode'] == mode]
        if not rows:
            continue
        usages = [row['answer']['run']['trace'].get('usage', {}) for row in rows]
        known = [usage for usage in usages if usage.get('calls_with_usage')]
        tokens = f"{sum(usage['input_tokens'] for usage in known)} / {sum(usage['output_tokens'] for usage in known)}" if known else '未返回'
        if not all(usage.get('complete') for usage in usages):
            tokens += '（不完整）'
        lines.append(f"| {mode} | {sum(row['answer']['status'] == 'completed' for row in rows)}/{len(rows)} | {mean(row['elapsed_seconds'] for row in rows):.2f}s | {median(row['elapsed_seconds'] for row in rows):.2f}s | {mean(row['tool_calls'] for row in rows):.2f} | {sum(usage.get('model_calls', 0) for usage in usages)} | {tokens} |")
    lines += ['', '工具次数只统计显式 Agent 工具，RAG 的固定检索不计为工具调用。耗时包含本地轮询延迟；token为服务商返回值，未折算费用。', '']
    if report.get('parameters', {}).get('evidence_review'):
        lines += ['## 证据核对', '', '| 模式 | 完成核对轮数 | 核对未完成 | 候选条目 / 保留条目 | 拒绝条目 | 平均核对耗时 |', '|---|---:|---:|---:|---:|---:|']
        for mode in ('rag', 'agent'):
            reviews = [row['answer']['run']['trace'].get('evidence_check', {}) for row in report['results'] if row['mode'] == mode]
            checked = [review for review in reviews if review.get('status') == 'checked']
            unavailable = sum(review.get('status') == 'unavailable' for review in reviews)
            duration = f"{mean(review['duration_ms'] for review in checked) / 1000:.2f}s" if checked else '不适用'
            lines.append(f"| {mode} | {len(checked)} | {unavailable} | {sum(review.get('claims', 0) for review in checked)} / {sum(review.get('kept', 0) for review in checked)} | {sum(sum(review.get('rejected', {}).values()) for review in checked)} | {duration} |")
        lines += ['', '代码生成的纯目录或无依据答复不调用核对器。候选条目是核对器选取的关键结论，不是草稿全部句子；拒绝计数也不等于捕获的所有错误数量。', '']
    lines += [
              '## 逐轮自动检查', '', '| 用例 / 轮次 | 模式 | 状态 | 自动结构检查 | 预期词组覆盖 | 工具次数 | 耗时 |', '|---|---|---|---|---:|---:|---:|']
    for row in report['results']:
        flags = row['evaluation']['automatic']
        failed = [key for key, value in flags.items() if value is False]
        coverage = row['evaluation']['coverage_fraction']
        lines.append(f"| {row['case_id']} / {row['turn']} | {row['mode']} | {row['answer']['status']} | {', '.join(failed) if failed else '通过'} | {f'{coverage:.0%}' if coverage is not None else '不适用'} | {row['tool_calls']} | {row['elapsed_seconds']:.2f}s |")
    lines += ['', '## 阅读结果时的边界', '',
              '- 自动结构检查仅覆盖完成状态、引用编号、课程归属、指定禁出词及预期缺口提示。',
              '- 词组覆盖只说明答案出现了预设词，不证明事实正确、引用充分或完成用户要求的全部步骤。',
              '- 原始JSON包含每轮答案、引用全文、工具执行摘要和人工评分栏；先核对每项事实与引用，再填写评分。',
              '- 单次运行不支持统计显著性判断。可用 --repeats 3 交替顺序重复运行。',
              '- 真实模型对照不替代取消、重启、资料替换等确定性自动验收。', '',
              f"原始数据：[{args.report.name}]({args.report.name})。"]
    output = args.output or args.report.with_suffix('.md')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(output))


if __name__ == '__main__':
    main()
