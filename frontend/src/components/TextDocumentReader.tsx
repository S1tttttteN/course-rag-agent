import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import type { Components } from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math-extended';
import 'katex/dist/katex.min.css';
import { documentsApi } from '../api/documents';

export interface TextCitation {
  text: string;
  startIndex: number;
}

interface SourceRange {
  start: number;
  end: number;
}

interface SourceNode {
  type: string;
  tagName?: string;
  children?: SourceNode[];
  properties?: Record<string, unknown>;
  position?: { start: { offset?: number }; end: { offset?: number } };
}

const locatableBlocks = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'li', 'pre', 'tr']);
const katexOptions = { trust: false, maxSize: 24, maxExpand: 1000 };

function markdownReadingBlocks({ recordPositions }: { recordPositions: boolean }) {
  function annotate(node: SourceNode) {
    if (node.type === 'element' && node.tagName && node.properties) {
      if (recordPositions && locatableBlocks.has(node.tagName)) {
        node.properties.dataSourceStart = node.position?.start.offset;
        node.properties.dataSourceEnd = node.position?.end.offset;
      }
      const code = node.tagName === 'pre' ? node.children?.[0] : undefined;
      const classes = code?.properties?.className;
      if (code?.properties && Array.isArray(classes) && classes.includes('language-math')) {
        // Keep the source-positioned wrapper when KaTeX replaces its inner math node.
        node.tagName = 'div';
        if (!classes.includes('math-display')) code.properties.className = [...classes, 'math-display'];
      }
    }
    node.children?.forEach(annotate);
  }
  return annotate;
}

const markdownComponents: Components = {
  img({ alt }) { return <span>{alt || '图片'}</span>; },
  a({ href, children }) {
    return href && /^https?:\/\//i.test(href)
      ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
      : <span>{children}</span>;
  },
  input({ checked }) { return <input type="checkbox" checked={checked ?? false} disabled />; },
};

function citationRange(source: string, citation?: TextCitation): SourceRange | null {
  if (!citation?.text.trim()) return null;
  const { text, startIndex } = citation;
  if (Number.isInteger(startIndex) && startIndex >= 0 && startIndex < source.length) {
    // Python stores Unicode code-point offsets. The browser uses UTF-16 offsets.
    let codePoints = 0;
    let offset = 0;
    for (const char of source) {
      if (codePoints === startIndex) break;
      offset += char.length;
      codePoints++;
    }
    if (codePoints === startIndex && source.startsWith(text, offset)) {
      return { start: offset, end: offset + text.length };
    }
  }
  const first = source.indexOf(text);
  return first >= 0 && source.indexOf(text, first + 1) === -1
    ? { start: first, end: first + text.length }
    : null;
}

function markdownTarget(frame: HTMLElement, range: SourceRange) {
  let target: HTMLElement | null = null;
  let firstOffset = Infinity;
  let smallestBlock = Infinity;
  for (const block of frame.querySelectorAll<HTMLElement>('[data-source-start][data-source-end]')) {
    const start = Number(block.dataset.sourceStart);
    const end = Number(block.dataset.sourceEnd);
    const overlap = Math.max(start, range.start);
    if (overlap >= Math.min(end, range.end) || !block.textContent?.trim() || !block.getClientRects().length) continue;
    if (overlap < firstOffset || (overlap === firstOffset && end - start < smallestBlock)) {
      target = block;
      firstOffset = overlap;
      smallestBlock = end - start;
    }
  }
  return target;
}

interface Props {
  courseId: string;
  documentId: string;
  versionId: string;
  format: 'txt' | 'markdown';
  citation?: TextCitation;
}

export default function TextDocumentReader({ courseId, documentId, versionId, format, citation }: Props) {
  const [source, setSource] = useState<{ text: string; range: SourceRange | null } | null>(null);
  const [error, setError] = useState('');
  const [located, setLocated] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const frame = useRef<HTMLDivElement>(null);
  const targetLine = useRef<HTMLSpanElement>(null);
  const hasCitation = citation !== undefined;
  const quote = citation?.text;
  const startIndex = citation?.startIndex;

  useEffect(() => {
    const controller = new AbortController();
    setError('');
    setSource(null);
    void documentsApi.text(courseId, documentId, versionId, controller.signal).then(({ text }) => {
      if (controller.signal.aborted) return;
      const range = quote !== undefined && startIndex !== undefined ? citationRange(text, { text: quote, startIndex }) : null;
      setSource({ text, range });
    }).catch(failure => {
      if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : '无法读取原文。');
    });
    return () => controller.abort();
  }, [courseId, documentId, versionId, quote, startIndex, attempt]);

  const range = source?.range ?? null;
  const lineStart = source && range && format === 'txt'
    ? (range.start === 0 ? 0 : source.text.lastIndexOf('\n', range.start - 1) + 1)
    : null;
  const markdown = useMemo(() => source && format === 'markdown'
    ? <div className="chat-markdown"><Markdown skipHtml remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[markdownReadingBlocks, { recordPositions: Boolean(source.range) }], [rehypeKatex, katexOptions]]}
        components={markdownComponents}>{source.text}</Markdown></div>
    : null, [source, format]);

  useLayoutEffect(() => {
    const container = frame.current;
    if (!container) return;
    container.scrollTop = 0;
    const target = range && (format === 'txt' ? targetLine.current : markdownTarget(container, range));
    setLocated(Boolean(target));
    if (!target) return;
    target.classList.add('text-source-target');
    container.scrollTop = Math.max(0, target.getBoundingClientRect().top - container.getBoundingClientRect().top - 12);
    return () => target.classList.remove('text-source-target');
  }, [range, format]);

  return <div className="text-reader">
    {error && <div className="error-banner" role="alert">
      <p>{error}</p>
      <button type="button" className="text-button" onClick={() => setAttempt(value => value + 1)}>重新读取原文</button>
    </div>}
    {!source && !error && <p className="subtle-note" role="status">正在读取原文…</p>}
    {source && hasCitation && !located && <p className="subtle-note" role="status">未能定位到具体位置。</p>}
    <div className="text-reader-frame" ref={frame} role="region" aria-label="资料正文" tabIndex={0} hidden={!source}>
      {markdown}
      {source && format === 'txt' && <pre className="text-reader-plain">{range && lineStart !== null ? <>
          {source.text.slice(0, lineStart)}<span ref={targetLine}>{source.text.slice(lineStart, range.start)}<mark>{source.text.slice(range.start, range.end)}</mark></span>{source.text.slice(range.end)}
        </> : source.text}</pre>}
    </div>
  </div>;
}
