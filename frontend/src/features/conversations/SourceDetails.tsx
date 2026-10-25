import type { Citation } from '../../types/conversation';

export function SourceDetails({citation}: {citation: Citation}) {
  const external = citation.kind !== 'course';
  return <>
    <p className="citation-source-title">{citation.source}</p>
    <p className="subtle-note">{external ? `${citation.kind === 'web_snippet' ? '网络搜索摘要' : '网页正文摘录'} · ${new Date(citation.retrieved_at).toLocaleString()}` : `课程资料 · 版本 ${citation.version}${citation.page !== null ? ` / 第 ${citation.page} 页` : ''}`}</p>
    {external && /^https?:\/\//i.test(citation.url) && <p className="subtle-note"><a href={citation.url} target="_blank" rel="noopener noreferrer">打开外部来源 ↗</a></p>}
    <p className={`snapshot-state ${citation.source_state}`}>{external ? citation.kind === 'web_snippet' ? '以下为本次网络搜索返回的摘要，不代表已阅读网页正文。' : '以下为本次读取网页时保存的正文摘录，不代表网页全文；网页内容可能更新。' : citation.source_state === 'deleted' ? '原资料已删除，保留回答时的原文快照。' : citation.source_state === 'updated' ? '资料已有新版本，以下为回答时使用的旧版快照。' : '以下为本次回答使用的课程原文快照。'}</p>
    <div className="citation-snapshot">{citation.text}</div>
  </>;
}
