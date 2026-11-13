import Markdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { createContext, useContext } from 'react';
import type { MouseEvent } from 'react';
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

interface CitationContextValue {
  citations: Citation[];
  selectedCitationId?: string;
  onCitation: (citation: Citation, event: MouseEvent<HTMLButtonElement>) => void;
}

const CitationContext = createContext<CitationContextValue | null>(null);

const markdownComponents: Components = {
  img({ alt }) { return <span>{alt || '图片内容请查阅原资料'}</span>; },
  a: function CitationLink({ href, children }) {
    const context = useContext(CitationContext);
    const match = href?.match(/^#source-(\d+)$/);
    if (!match) return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
    const citation = context?.citations.find(item => item.number === Number(match[1]));
    if (!citation || !context) return <span>{children}</span>;
    return <button type="button" className="inline-citation"
      aria-label={`查看引用 ${citation.number}`} aria-expanded={context.selectedCitationId === citation.id}
      onClick={event => context.onCitation(citation, event)}>{children}</button>;
  },
};

export function Answer({ text, ...citationContext }: CitationContextValue & { text: string }) {
  return <CitationContext.Provider value={citationContext}>
    <div className="chat-markdown">
      <Markdown skipHtml remarkPlugins={[remarkGfm, citationLinks]} components={markdownComponents}>{text}</Markdown>
    </div>
  </CitationContext.Provider>;
}
