import { useEffect, useRef, useState } from 'react';
import { documentsApi } from '../api/documents';

const framePolicy = "default-src 'none'; style-src 'unsafe-inline'; img-src blob:; base-uri 'none'; form-action 'none'";
const readerStyles = `
  html { color-scheme: light; }
  body { margin: 0; padding: 16px; font: 15px/1.65 system-ui, sans-serif; color: #19263a; background: white; }
  img { max-width: 100%; height: auto; }
  pre { overflow: auto; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #d9e2ed; padding: 5px 8px; vertical-align: top; }
  a { color: #2454a6; }
`;
const wordStyles = `
  html { color-scheme: light; }
  body { margin: 0; color: #19263a; background: white; }
  body > section.docx { max-width: 100%; margin: 0 auto; overflow: visible; }
  .docx-anchored-figures { max-width: 100%; margin-block: 4px; overflow: auto; }
  .docx-anchored-figures > div > svg { overflow: visible; }
  @media (max-width: 640px) {
    body > section.docx { padding: 24px clamp(12px, 5vw, 32px) !important; }
  }
`;
const bodyAttributes = ['id', 'class', 'style', 'dir', 'lang'];
const readingBlocks = 'p, li, h1, h2, h3, h4, h5, h6, pre, dd, dt, figcaption, blockquote, div, section, article';

interface Props {
  courseId: string;
  documentId: string;
  versionId: string;
  format: 'html' | 'docx';
  quote?: string;
  onClose: () => void;
}

interface RichPreview {
  html: string;
  css: string;
}

function findQuote(doc: Document, quote: string) {
  const excerpt = quote.replace(/\s/gu, '');
  if (!excerpt) return null;

  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
  const range = doc.createRange();
  const styles = new WeakMap<Element, boolean>();
  const parts: { element: HTMLElement; end: number; text: string }[] = [];
  let length = 0;
  for (let text = walker.nextNode(); text; text = walker.nextNode()) {
    const value = text.textContent?.replace(/\s/gu, '');
    const parent = text.parentElement;
    if (!value || !parent) continue;
    let visible = styles.get(parent);
    if (visible === undefined) {
      const style = doc.defaultView!.getComputedStyle(parent);
      visible = style.visibility === 'visible' && style.opacity !== '0';
      for (let ancestor = parent.parentElement; visible && ancestor; ancestor = ancestor.parentElement) {
        visible = doc.defaultView!.getComputedStyle(ancestor).opacity !== '0';
      }
      styles.set(parent, visible);
    }
    if (!visible) continue;
    range.selectNodeContents(text);
    if (!Array.from(range.getClientRects()).some(rect => rect.width > 0 && rect.height > 0)) continue;
    length += value.length;
    parts.push({ element: parent.closest('tr') ?? parent.closest<HTMLElement>(readingBlocks) ?? parent, end: length, text: value });
  }

  const visibleText = parts.map(part => part.text).join('');
  const start = visibleText.indexOf(excerpt);
  if (start < 0 || visibleText.indexOf(excerpt, start + 1) !== -1) return null;
  return parts.find(part => part.end > start)?.element ?? null;
}

function previewLink(href: string): string | null {
  if (href.startsWith('#')) return href;
  try {
    const url = new URL(href);
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : null;
  } catch {
    return null;
  }
}

function followPreviewLink(event: MouseEvent, doc: Document) {
  const target = event.target as Element | null;
  const link = target?.closest?.<HTMLAnchorElement>('a[href]');
  if (!link) return;
  event.preventDefault();
  const href = previewLink(link.getAttribute('href') ?? '');
  if (!href) return;

  if (href.startsWith('#')) {
    if (href === '#') {
      doc.defaultView?.scrollTo({ top: 0 });
      return;
    }
    let id = href.slice(1);
    try { id = decodeURIComponent(id); } catch { /* Try the literal fragment. */ }
    const anchor = doc.getElementById(id) ?? Array.from(doc.getElementsByName(id)).find(element => element.tagName === 'A');
    anchor?.scrollIntoView({ block: 'start' });
    return;
  }

  window.open(href, '_blank', 'noopener,noreferrer');
}

function installRasterImage(image: Element, attribute: string, dataUrl: string, imageUrls: string[]) {
  const data = /^data:(image\/(?:png|jpeg|gif|webp));base64,([\w+/=]+)$/u.exec(dataUrl);
  if (!data) return;
  const [, type, payload] = data;
  try {
    const header = atob(payload.slice(0, 16));
    const valid = (type === 'image/png' && header.startsWith('\x89PNG\r\n\x1a\n'))
      || (type === 'image/jpeg' && header.startsWith('\xff\xd8\xff'))
      || (type === 'image/gif' && /^GIF8[79]a/u.test(header))
      || (type === 'image/webp' && header.startsWith('RIFF') && header.slice(8, 12) === 'WEBP');
    if (!valid) return;
    const bytes = Uint8Array.from(atob(payload), char => char.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type }));
    imageUrls.push(url);
    image.setAttribute(attribute, url);
  } catch (error) {
    if (!(error instanceof DOMException && error.name === 'InvalidCharacterError')) throw error;
    // Invalid embedded base64 only affects that image.
  }
}

function placeWordTextBoxes(doc: Document) {
  const anchors = new Set<HTMLParagraphElement>();
  for (const box of doc.querySelectorAll('svg foreignObject')) {
    const svg = box.closest<SVGSVGElement>('svg');
    if (svg?.style.position !== 'absolute') continue;
    const paragraph = svg.closest('p');
    if (paragraph) anchors.add(paragraph);
  }

  for (const paragraph of anchors) {
    // Reserve space in the reading flow for all parts of the diagram at this anchor.
    const shapes = [...paragraph.querySelectorAll('svg')]
      .filter(svg => svg.style.position === 'absolute' && svg.closest('p') === paragraph)
      .map(svg => ({ svg, bounds: svg.getBoundingClientRect() }));
    let left = Infinity;
    let top = Infinity;
    let right = -Infinity;
    let bottom = -Infinity;
    for (const { bounds } of shapes) {
      left = Math.min(left, bounds.left);
      top = Math.min(top, bounds.top);
      right = Math.max(right, bounds.right);
      bottom = Math.max(bottom, bounds.bottom);
    }

    const scroller = doc.createElement('div');
    scroller.className = 'docx-anchored-figures';
    const layout = doc.createElement('div');
    layout.style.position = 'relative';
    layout.style.width = `${Math.ceil(right - left) + 2}px`;
    layout.style.height = `${Math.ceil(bottom - top) + 2}px`;
    for (const { svg, bounds } of shapes) {
      svg.style.inset = `${bounds.top - top + 1}px auto auto ${bounds.left - left + 1}px`;
      svg.style.margin = '0';
      svg.style.width = `${bounds.width}px`;
      svg.style.height = `${bounds.height}px`;
      layout.appendChild(svg);
    }
    scroller.appendChild(layout);
    paragraph.after(scroller);
  }
}

async function renderWord(doc: Document, source: Blob, signal: AbortSignal, imageUrls: string[]) {
  const { parseAsync, renderDocument } = await import('docx-preview');
  if (signal.aborted) return;
  // Never render external fonts or package parts containing alternative HTML.
  const options = { breakPages: false, ignoreHeight: true, ignoreFonts: true, inWrapper: false,
    renderAltChunks: false, useBase64URL: true };
  const word = await parseAsync(source, options);
  if (signal.aborted) return;
  const nodes = await renderDocument(word, options);
  if (signal.aborted) return;
  for (const node of nodes) {
    if (node instanceof Element) {
      for (const link of node.querySelectorAll('a[href]')) {
        const href = previewLink(link.getAttribute('href') ?? '');
        if (href) link.setAttribute('href', href);
        else link.removeAttribute('href');
      }
      for (const image of node.querySelectorAll('img[src], svg image[href]')) {
        const attribute = image.localName === 'img' ? 'src' : 'href';
        const value = image.getAttribute(attribute) ?? '';
        image.removeAttribute(attribute);
        installRasterImage(image, attribute, value, imageUrls);
      }
      // Legacy Word shapes may render a text box under <image>, whose children are invisible.
      for (const image of node.querySelectorAll('svg image')) {
        const boxes = [...image.children].filter(child => child.localName === 'foreignObject');
        if (!boxes.length) continue;
        image.after(...boxes);
        if (!image.hasAttribute('href')) {
          const shape = doc.createElementNS(image.namespaceURI, 'rect');
          for (const name of ['width', 'height', 'stroke-width']) {
            const value = image.getAttribute(name);
            if (value !== null) shape.setAttribute(name, value);
          }
          for (const name of ['fill', 'stroke']) {
            // A theme color can carry a Word palette index after its RGB value.
            const color = /^#[\da-f]{6}(?=\s|$)/iu.exec(image.getAttribute(name) ?? '');
            shape.setAttribute(name, color?.[0] ?? 'none');
          }
          image.replaceWith(shape);
        }
      }
    }
    (node.nodeName === 'STYLE' ? doc.head : doc.body).appendChild(node);
  }
  placeWordTextBoxes(doc);
}

export default function RichDocumentReader({ courseId, documentId, versionId, format, quote, onClose }: Props) {
  const [preview, setPreview] = useState<RichPreview | Blob | null>(null);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [located, setLocated] = useState<boolean | null>(null);
  const [attempt, setAttempt] = useState(0);
  const iframe = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const request = new AbortController();
    setError('');
    setReady(false);
    setPreview(null);
    setLocated(null);
    const load = format === 'docx' ? documentsApi.docx : documentsApi.richPreview;
    void load(courseId, documentId, versionId, request.signal).then(result => {
      if (!request.signal.aborted) setPreview(result);
    }).catch((failure: unknown) => {
      if (!request.signal.aborted) setError(failure instanceof Error ? failure.message : '无法读取资料原文。');
    });
    return () => request.abort();
  }, [courseId, documentId, versionId, format, attempt]);

  useEffect(() => {
    const frame = iframe.current;
    if (!frame || !preview) return;
    const lifetime = new AbortController();
    const imageUrls: string[] = [];
    let prepared = false;

    async function prepareFrame() {
      const doc = frame?.contentDocument;
      const style = doc?.head.querySelector('style');
      if (!doc || !style || !preview || prepared) return;
      prepared = true;

      doc.addEventListener('click', event => followPreviewLink(event, doc), { signal: lifetime.signal });
      doc.addEventListener('auxclick', event => {
        if (event.button === 1) followPreviewLink(event, doc);
      }, { signal: lifetime.signal });
      doc.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
          event.preventDefault();
          onClose();
        }
      }, { signal: lifetime.signal });

      if (preview instanceof Blob) {
        style.textContent = wordStyles;
        await renderWord(doc, preview, lifetime.signal, imageUrls);
      } else {
        const bodyWrapper = doc.body.firstElementChild;
        if (!bodyWrapper) throw new Error('资料正文缺失。');
        for (const attr of bodyAttributes) {
          const value = bodyWrapper.getAttribute(attr);
          if (value !== null) doc.body.setAttribute(attr, value);
        }
        const content = doc.createRange();
        content.selectNodeContents(bodyWrapper);
        doc.body.replaceChildren(content.extractContents());
        style.textContent = readerStyles + '\n' + preview.css;

        for (const image of doc.querySelectorAll<HTMLImageElement>('img[data-preview-image]')) {
          installRasterImage(image, 'src', image.dataset.previewImage ?? '', imageUrls);
          image.removeAttribute('data-preview-image');
        }
      }
      if (lifetime.signal.aborted) return;
      if (quote !== undefined) {
        // Wait for embedded images to claim their space before locating the excerpt.
        await Promise.allSettled([...doc.images].map(image => image.decode()));
        if (lifetime.signal.aborted) return;
        const target = findQuote(doc, quote);
        if (target) {
          target.style.setProperty('outline', '2px solid #bd7c19', 'important');
          target.style.setProperty('background-color', '#fff4d2', 'important');
          target.scrollIntoView({ block: 'center' });
        }
        setLocated(Boolean(target));
      }
      setReady(true);
    }

    frame.addEventListener('load', () => {
      void prepareFrame().catch(() => {
        if (!lifetime.signal.aborted) setError('无法显示资料原文。');
      });
    }, { signal: lifetime.signal });
    // The frame receives only allowlisted markup and has a policy before any document content.
    frame.srcdoc = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta http-equiv="Content-Security-Policy" content="${framePolicy}"><style></style>
      </head><body>${preview instanceof Blob ? '' : preview.html}</body></html>`;
    return () => {
      lifetime.abort();
      imageUrls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [preview, quote, onClose]);

  return <div className="text-reader">
    {error ? <div className="error-banner" role="alert">
      <p>{error}</p>
      <button type="button" className="text-button" onClick={() => setAttempt(value => value + 1)}>重新读取原文</button>
    </div> : !ready && <p className="subtle-note" role="status">正在读取原文…</p>}
    {!error && located === false && <p className="subtle-note" role="status">未能定位到具体位置。</p>}
    {!error && preview && <iframe ref={iframe} className="rich-reader-frame" title="资料正文" sandbox="allow-same-origin" />}
  </div>;
}
