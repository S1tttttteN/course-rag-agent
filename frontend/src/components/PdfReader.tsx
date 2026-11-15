import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { ChevronLeft, ChevronRight, Minus, Plus } from 'lucide-react';
import type { PDFDocumentLoadingTask, ResponseException } from 'pdfjs-dist';
import type { PDFViewer } from 'pdfjs-dist/web/pdf_viewer.mjs';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

interface PageLocation {
  pageNumber: number;
  left: number;
  top: number;
}

export interface PdfReadingPosition {
  url: string;
  initialPage: number;
  location: PageLocation;
  scale: string;
}

interface Props {
  url: string;
  initialPage?: number;
  readingPosition: RefObject<PdfReadingPosition | null>;
}

const fitWidth = 'page-width';

function restoreLocation(viewer: PDFViewer, location: PageLocation) {
  viewer.scrollPageIntoView({
    pageNumber: location.pageNumber,
    destArray: [null, { name: 'XYZ' }, location.left, location.top, null],
    allowNegativeOffset: true,
  });
}

function updateBottomSpace(viewer: PDFViewer) {
  if (!viewer.pagesCount) return;
  const lastPage = viewer.getPageView(viewer.pagesCount - 1).div;
  // Leave enough space to align the last page to the top when jumping.
  const space = lastPage.isConnected ? Math.max(12, viewer.container.clientHeight - lastPage.offsetHeight + 2) : 12;
  viewer.container.style.setProperty('--pdfViewer-padding-bottom', space + 'px');
}

export function PdfReader({ url, initialPage = 1, readingPosition }: Props) {
  const [pages, setPages] = useState(0);
  const [page, setPage] = useState(1);
  const [pageInput, setPageInput] = useState<string | null>(null);
  const [scale, setScale] = useState<string>(fitWidth);
  const [pageMode, setPageMode] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [unreadablePage, setUnreadablePage] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);
  const frame = useRef<HTMLDivElement>(null);
  const surface = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<PDFViewer | null>(null);
  const navigation = useRef(0);

  useEffect(() => {
    const container = frame.current!;
    const pageSurface = surface.current!;
    const lifetime = new AbortController();
    let task: PDFDocumentLoadingTask | undefined;
    let viewer: PDFViewer | undefined;
    let observer: ResizeObserver | undefined;
    setPages(0);
    setNotice('');
    setError('');
    setUnreadablePage(null);
    setPageInput(null);

    void (async () => {
      try {
        // The packaged viewer reads PDF.js from globalThis during module initialization.
        const pdfjs = await import('pdfjs-dist');
        if (lifetime.signal.aborted) return;
        const { EventBus, PDFViewer, ScrollMode } = await import('pdfjs-dist/web/pdf_viewer.mjs');
        if (lifetime.signal.aborted) return;

        pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
        const assets = import.meta.env.BASE_URL + 'pdf-assets/';
        task = pdfjs.getDocument({
          url,
          cMapUrl: assets + 'cmaps/',
          standardFontDataUrl: assets + 'standard_fonts/',
          iccUrl: assets + 'iccs/',
          wasmUrl: assets + 'wasm/',
          disableAutoFetch: true,
          disableWorkerRendering: true,
        });
        const pdf = await task.promise;
        if (lifetime.signal.aborted) return;
        const validPage = Number.isInteger(initialPage) && initialPage >= 1 && initialPage <= pdf.numPages;
        const saved = readingPosition.current;
        const resume = saved?.url === url && saved.initialPage === initialPage ? saved : null;
        const start = resume?.location.pageNumber ?? (validPage ? initialPage : 1);
        const [, startPage] = await Promise.all([pdf.getPage(1), pdf.getPage(start)]);
        if (lifetime.signal.aborted) return;

        const eventBus = new EventBus();
        const options = {
          container,
          viewer: pageSurface,
          eventBus,
          abortSignal: lifetime.signal,
          textLayerMode: 0,
          annotationMode: pdfjs.AnnotationMode.DISABLE,
          annotationEditorMode: pdfjs.AnnotationEditorType.DISABLE,
          enableAutoLinking: false,
          enableSelectionRendering: false,
          supportsPinchToZoom: false,
          maxCanvasPixels: 4_000_000,
          maxCanvasDim: 4096,
          enableDetailCanvas: false,
        };
        const pdfViewer = new PDFViewer(options);
        viewer = pdfViewer;
        const listeners = { signal: lifetime.signal };
        let lastPage = start;
        let ready = false;
        let width = container.clientWidth;
        eventBus.on('updateviewarea', ({ location }: { location: PageLocation }) => {
          if (!ready) return;
          readingPosition.current = { url, initialPage, location, scale: pdfViewer.currentScaleValue };
          if (lastPage !== location.pageNumber) {
            lastPage = location.pageNumber;
            setPage(lastPage);
            setPageInput(null);
          }
        }, listeners);
        eventBus.on('scalechanging', () => {
          setScale(pdfViewer.currentScaleValue);
          updateBottomSpace(pdfViewer);
        }, listeners);
        eventBus.on('pagerendered', ({ pageNumber, error }: { pageNumber: number; error?: unknown }) => {
          updateBottomSpace(pdfViewer);
          setUnreadablePage(failedPage => error ? pageNumber : failedPage === pageNumber ? null : failedPage);
        }, listeners);
        eventBus.on('pagesinit', () => {
          if (start > 1) pdfViewer.getPageView(start - 1).setPdfPage(startPage);
          pdfViewer.scrollPageIntoView({ pageNumber: start });
          pdfViewer.currentScaleValue = resume?.scale ?? fitWidth;
          updateBottomSpace(pdfViewer);
          if (resume) restoreLocation(pdfViewer, resume.location);
          else pdfViewer.scrollPageIntoView({ pageNumber: start });
          viewerRef.current = pdfViewer;
          setPage(start);
          setPages(pdf.numPages);
          setPageMode(pdfViewer.scrollMode === ScrollMode.PAGE);
          setNotice(validPage ? '' : '引用页码超出范围，已打开第 1 页。');
          ready = true;
        }, listeners);
        pdfViewer.setDocument(pdf);
        void (pdfViewer.pagesPromise as Promise<unknown>).catch(() => {
          if (!lifetime.signal.aborted) setError('无法准备 PDF 页面，请重试。');
        });

        observer = new ResizeObserver(() => {
          if (!ready) return;
          const nextWidth = container.clientWidth;
          const anchor = readingPosition.current?.location;
          if (nextWidth && nextWidth !== width && pdfViewer.currentScaleValue === fitWidth) {
            pdfViewer.currentScaleValue = fitWidth;
            updateBottomSpace(pdfViewer);
            if (anchor) restoreLocation(pdfViewer, anchor);
          } else {
            updateBottomSpace(pdfViewer);
          }
          width = nextWidth;
          pdfViewer.update();
        });
        observer.observe(container);
      } catch (failure) {
        if (lifetime.signal.aborted) return;
        const error = failure as ResponseException | null;
        if (error?.status === 404) setError('PDF 原文件不可用。');
        else if (error?.name === 'PasswordException') setError('需要密码的 PDF 无法在此预览。');
        else setError('无法读取 PDF，请确认文件可打开后重试。');
      }
    })();

    return () => {
      lifetime.abort();
      observer?.disconnect();
      viewerRef.current = null;
      navigation.current++;
      viewer?.setDocument(null);
      void viewer?.l10n?.destroy();
      void task?.destroy().catch(() => {});
    };
  }, [url, initialPage, attempt, readingPosition]);

  async function goToPage(next: number) {
    const viewer = viewerRef.current;
    const pdf = viewer?.pdfDocument;
    if (!viewer || !pdf) return;
    const request = ++navigation.current;
    setPageInput(null);
    if (!Number.isInteger(next) || next < 1 || next > pages) {
      setNotice(`请输入 1 至 ${pages} 的页码。`);
      return;
    }
    setNotice('');
    try {
      const view = viewer.getPageView(next - 1);
      if (!view.pdfPage) {
        const target = await pdf.getPage(next);
        if (request !== navigation.current) return;
        if (!view.pdfPage) view.setPdfPage(target);
      }
      updateBottomSpace(viewer);
      viewer.scrollPageIntoView({ pageNumber: next });
      viewer.update();
    } catch {
      if (request === navigation.current) setNotice(`无法跳转到第 ${next} 页。`);
    }
  }

  return <div className="pdf-reader">
    {pages > 0 && <div className="pdf-toolbar">
      <form className="pdf-paging" onSubmit={event => {
        event.preventDefault();
        void goToPage(pageInput === null ? page : Number(pageInput));
      }}>
        <button type="button" className="icon-button" aria-label="上一页" disabled={page === 1} onClick={() => void goToPage(page - 1)}><ChevronLeft size={17} /></button>
        <input type="text" inputMode="numeric" aria-label="PDF 页码" value={pageInput ?? String(page)}
          onChange={event => setPageInput(event.target.value)} />
        <span>/ {pages}</span>
        <button type="submit" className="text-button">跳转</button>
        <button type="button" className="icon-button" aria-label="下一页" disabled={page === pages} onClick={() => void goToPage(page + 1)}><ChevronRight size={17} /></button>
      </form>
      <div className="pdf-zoom" role="group" aria-label="PDF 缩放">
        <button type="button" className="icon-button" aria-label="缩小 PDF" onClick={() => viewerRef.current?.decreaseScale()}><Minus size={16} /></button>
        <button type="button" className="text-button" aria-label="适合宽度" onClick={() => {
          if (viewerRef.current) viewerRef.current.currentScaleValue = fitWidth;
        }}>{scale === fitWidth ? '适合宽度' : `${Math.round(Number(scale) * 100)}%`}</button>
        <button type="button" className="icon-button" aria-label="放大 PDF" onClick={() => viewerRef.current?.increaseScale()}><Plus size={16} /></button>
      </div>
      {pageMode && <p className="subtle-note pdf-mode-note" role="status">文档页数较多，已切换到逐页阅读。</p>}
    </div>}
    {notice && <p className="subtle-note" role="status">{notice}</p>}
    {error ? <div className="error-banner" role="alert">
      <p>{error}</p><button type="button" className="text-button" onClick={() => setAttempt(value => value + 1)}>重新打开 PDF</button>
    </div> : !pages && <p className="subtle-note" role="status">正在读取 PDF…</p>}
    {unreadablePage === page && <p className="error" role="status">无法显示这一页，请跳转到其他页。</p>}
    <div className="pdf-scroll-area">
      <div className="pdf-frame" role="region" aria-label="PDF 页面" tabIndex={0} ref={frame}>
        <div className="pdfViewer" ref={surface} />
      </div>
    </div>
  </div>;
}
