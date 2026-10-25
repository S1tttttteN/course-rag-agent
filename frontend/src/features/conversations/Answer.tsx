import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Citation } from '../../types/conversation';

interface MarkdownNode { type: string; value?: string; url?: string; children?: MarkdownNode[] }

// Transform text nodes only, preserving code blocks, inline code and existing links.
function citationLinks() {
  return (tree: MarkdownNode) => {
    function walk(node: MarkdownNode) {
      if (!node.children || ['link', 'code', 'inlineCode'].includes(node.type)) return;
      node.children = node.children.flatMap(child => {
        if (child.type !== 'text' || !child.value) { walk(child); return [child]; }
        const parts: MarkdownNode[] = [];
        let offset = 0;
        for (const match of child.value.matchAll(/\[(\d{1,4})\]/g)) {
          if (match.index > offset) parts.push({type: 'text', value: child.value.slice(offset, match.index)});
          parts.push({type: 'link', url: `#source-${match[1]}`, children: [{type: 'text', value: match[0]}]});
          offset = match.index + match[0].length;
        }
        if (offset < child.value.length) parts.push({type: 'text', value: child.value.slice(offset)});
        return parts.length ? parts : [child];
      });
    }
    walk(tree);
  };
}

export function Answer({ text, citations, onCitation }: {text: string; citations: Citation[]; onCitation: (citation: Citation) => void}) {
  return <div className="chat-markdown"><Markdown skipHtml remarkPlugins={[remarkGfm, citationLinks]} components={{
    img({alt}) { return <span>{alt || '图片内容请查阅原资料'}</span>; },
    a({href, children}) {
      const match = href?.match(/^#source-(\d+)$/);
      if (match) {
        const citation = citations.find(item => item.number === Number(match[1]));
        return citation ? <button className="inline-citation" aria-label={`查看引用 ${citation.number}`} onClick={() => onCitation(citation)}>{children}</button> : <span>{children}</span>;
      }
      return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
    },
  }}>{text}</Markdown></div>;
}
