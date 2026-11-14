import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent, ReactNode } from 'react';
import { GripHorizontal, Maximize, Minimize, X } from 'lucide-react';
import { Dialog } from './Dialog';
import { PdfReader } from './PdfReader';
import type { PdfReadingPosition } from './PdfReader';
import { compactReadingQuery, useMediaQuery } from './useMediaQuery';

interface Props {
  url: string;
  filename: string;
  version: number;
  initialPage?: number;
  fromCitation?: boolean;
  onClose: () => void;
}

interface Bounds {
  left: number;
  top: number;
  width: number;
  height: number;
}

const minWidth = 380;
const minHeight = 240;
const resizeEdges = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'] as const;
type ResizeEdge = typeof resizeEdges[number];

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(value, max));
}

function withinViewport(bounds: Bounds): Bounds {
  const { clientWidth, clientHeight } = document.documentElement;
  const width = clamp(bounds.width, Math.min(minWidth, clientWidth), clientWidth);
  const height = clamp(bounds.height, Math.min(minHeight, clientHeight), clientHeight);
  return {
    left: clamp(bounds.left, 0, clientWidth - width),
    top: clamp(bounds.top, 0, clientHeight - height),
    width,
    height,
  };
}

function resizeAxis(start: number, size: number, delta: number, minSize: number, limit: number, fromStart: boolean) {
  if (fromStart) {
    const end = start + size;
    const nextStart = clamp(start + delta, 0, Math.max(0, end - minSize));
    return [nextStart, end - nextStart];
  }
  return [start, clamp(size + delta, Math.min(minSize, limit - start), limit - start)];
}

function resizeBounds(bounds: Bounds, edge: ResizeEdge, dx: number, dy: number): Bounds {
  const { clientWidth, clientHeight } = document.documentElement;
  let { left, top, width, height } = withinViewport(bounds);
  if (edge.includes('w') || edge.includes('e')) {
    [left, width] = resizeAxis(left, width, dx, minWidth, clientWidth, edge.includes('w'));
  }
  if (edge.includes('n') || edge.includes('s')) {
    [top, height] = resizeAxis(top, height, dy, minHeight, clientHeight, edge.includes('n'));
  }
  return { left, top, width, height };
}

function FloatingPdfWindow({ filename, version, onClose, children }: Pick<Props, 'filename' | 'version' | 'onClose'> & { children: ReactNode }) {
  const panel = useRef<HTMLElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const drag = useRef<{ id: number; x: number; y: number; bounds: Bounds; edge?: ResizeEdge } | null>(null);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  const [maximized, setMaximized] = useState(false);
  const [moving, setMoving] = useState(false);

  useEffect(() => {
    closeButton.current?.focus({ preventScroll: true });
    function fitWindow() {
      setBounds(previous => previous && withinViewport(previous));
    }
    window.addEventListener('resize', fitWindow);
    return () => window.removeEventListener('resize', fitWindow);
  }, []);

  function startDrag(event: PointerEvent<HTMLDivElement>, edge?: ResizeEdge) {
    const rect = panel.current?.getBoundingClientRect();
    if (!rect || maximized || drag.current || !event.isPrimary || event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, bounds: rect, edge };
    if (!edge) setMoving(true);
  }

  function moveDrag(event: PointerEvent<HTMLDivElement>) {
    const activeDrag = drag.current;
    if (activeDrag?.id !== event.pointerId) return;
    const dx = event.clientX - activeDrag.x;
    const dy = event.clientY - activeDrag.y;
    const { left, top, width, height } = activeDrag.bounds;
    setBounds(activeDrag.edge ? resizeBounds(activeDrag.bounds, activeDrag.edge, dx, dy)
      : withinViewport({ left: left + dx, top: top + dy, width, height }));
  }

  function stopDrag(event: PointerEvent<HTMLDivElement>) {
    if (drag.current?.id === event.pointerId) {
      drag.current = null;
      setMoving(false);
    }
  }

  function toggleMaximized() {
    const rect = panel.current?.getBoundingClientRect();
    if (!maximized && rect) setBounds(withinViewport(rect));
    setMaximized(!maximized);
  }

  function moveWithKeyboard(event: KeyboardEvent<HTMLDivElement>) {
    const element = panel.current;
    if (!element || maximized) return;
    const step = 24;
    const bounds = element.getBoundingClientRect();
    let { left, top } = bounds;
    switch (event.key) {
      case 'ArrowLeft': left -= step; break;
      case 'ArrowRight': left += step; break;
      case 'ArrowUp': top -= step; break;
      case 'ArrowDown': top += step; break;
      default: return;
    }
    event.preventDefault();
    setBounds(withinViewport({ left, top, width: bounds.width, height: bounds.height }));
  }

  return <section role="dialog" aria-label="PDF 原文" ref={panel}
    className={maximized ? 'pdf-window is-maximized' : 'pdf-window'}
    style={!maximized && bounds ? { ...bounds, right: 'auto' } : undefined}
    onKeyDown={event => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        event.stopPropagation();
        onClose();
      }
    }}>
    <header className="pdf-window-heading">
      <div role="group" tabIndex={maximized ? -1 : 0} aria-label={maximized ? 'PDF 文件' : '移动 PDF 窗口，使用方向键'}
        className={moving ? 'pdf-window-handle is-moving' : 'pdf-window-handle'}
        onPointerDown={event => startDrag(event)} onPointerMove={moveDrag} onLostPointerCapture={stopDrag}
        onKeyDown={moveWithKeyboard}>
        {!maximized && <GripHorizontal size={18} aria-hidden="true" />}
        <h2 title={filename}>{filename}</h2>
        <span className="subtle-note">版本 {version}</span>
      </div>
      <button type="button" className="icon-button" aria-label={maximized ? '还原窗口' : '最大化窗口'}
        title={maximized ? '还原窗口' : '最大化窗口'} onClick={toggleMaximized}>
        {maximized ? <Minimize size={17} /> : <Maximize size={17} />}
      </button>
      <button type="button" className="icon-button" aria-label="关闭 PDF 原文" ref={closeButton} onClick={onClose}><X size={18} /></button>
    </header>
    {children}
    {!maximized && resizeEdges.map(edge => <div key={edge} className="pdf-window-resize" data-edge={edge}
      aria-hidden="true" onPointerDown={event => startDrag(event, edge)} onPointerMove={moveDrag}
      onLostPointerCapture={stopDrag} />)}
  </section>;
}

export function PdfReadingWindow({ url, filename, version, initialPage, fromCitation, onClose }: Props) {
  const compact = useMediaQuery(compactReadingQuery);
  const readingPosition = useRef<PdfReadingPosition>(null);
  const reader = <PdfReader key={url} url={url} initialPage={initialPage} readingPosition={readingPosition} />;

  if (compact) {
    return <Dialog title="PDF 原文" variant="sheet" busy={false} onClose={onClose}>
      <div className="reading-panel-body pdf-sheet-body">
        {fromCitation && <button type="button" className="text-button" onClick={onClose}>返回摘录</button>}
        <h3 className="citation-source-title">{filename}</h3>
        <p className="subtle-note">版本 {version}</p>
        {reader}
      </div>
    </Dialog>;
  }

  return <FloatingPdfWindow filename={filename} version={version} onClose={onClose}>
    {reader}
  </FloatingPdfWindow>;
}
