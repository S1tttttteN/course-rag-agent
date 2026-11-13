import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import { CheckCircle2, ChevronLeft, ChevronRight, Clock3, FileText, FolderOpen, LoaderCircle, RefreshCw, Search, Trash2, Upload, XCircle } from 'lucide-react';
import { documentsApi } from '../../api/documents';
import { Dialog } from '../../components/Dialog';
import { ReadingPanel } from '../../components/ReadingPanel';
import type { CourseDocument, LibraryConfig, Preview, SearchResult } from '../../types/document';

function sizeLabel(size: number) { return size < 1024 * 1024 ? `${Math.max(1, Math.round(size / 1024))} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`; }

export function DocumentLibrary({ courseId }: { courseId: string }) {
  const [documents, setDocuments] = useState<CourseDocument[]>([]);
  const [config, setConfig] = useState<LibraryConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [actionError, setActionError] = useState('');
  const [busy, setBusy] = useState(false);
  const [uploadLabel, setUploadLabel] = useState('');
  const [notice, setNotice] = useState('');
  const [deleting, setDeleting] = useState<CourseDocument | null>(null);
  const [replacement, setReplacement] = useState<CourseDocument | null>(null);
  const [selection, setSelection] = useState<{ id: string; offset: number } | null>(null);
  const [preview, setPreview] = useState<{ key: string; data?: Preview; error?: string } | null>(null);
  const [previewAttempt, setPreviewAttempt] = useState(0);
  const [query, setQuery] = useState('');
  const [semantic, setSemantic] = useState(true);
  const [rerank, setRerank] = useState(false);
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<SearchResult | null>(null);
  const [searchError, setSearchError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const replaceInput = useRef<HTMLInputElement>(null);
  const mounted = useRef(true);
  const listRequest = useRef(0);
  const previewTrigger = useRef<HTMLElement | null>(null);
  const selectedDocument = documents.find(document => document.id === selection?.id && document.usable);
  const selectedId = selectedDocument?.id;
  const selectedVersion = selectedDocument?.active_version_number;
  const offset = selection?.offset ?? 0;
  const previewKey = selectedId ? `${selectedId}:${selectedVersion}:${offset}:${previewAttempt}` : '';
  const currentPreview = preview?.key === previewKey ? preview : null;
  const readyCount = documents.filter(document => document.usable).length;
  const hasPending = documents.some(document => ['queued', 'processing'].includes(document.latest.status));
  const revision = documents.map(document => `${document.id}:${document.active_version_id}`).join(',');
  const currentRevision = useRef(revision);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      listRequest.current++;
    };
  }, []);
  const reload = useCallback(async () => {
    const request = ++listRequest.current;
    try {
      const [items, settings] = await Promise.all([documentsApi.list(courseId), documentsApi.config()]);
      if (request === listRequest.current) { setDocuments(items); setConfig(settings); setError(''); }
    } catch (error) {
      if (request === listRequest.current) setError(error instanceof Error ? error.message : '资料加载失败。');
    } finally { if (request === listRequest.current) setLoading(false); }
  }, [courseId]);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => {
    const timer = setInterval(() => { void reload(); }, hasPending ? 1500 : 7000);
    return () => clearInterval(timer);
  }, [reload, hasPending]);
  useEffect(() => { currentRevision.current = revision; setResults(null); setSearchError(''); }, [revision]);
  useEffect(() => {
    if (selection && !selectedId) setSelection(null);
  }, [selectedId, selection]);
  useEffect(() => {
    if (!selectedId) return;
    let ignore = false;
    void documentsApi.preview(courseId, selectedId, offset).then(data => {
      if (ignore) return;
      if (data.version !== selectedVersion) {
        setPreview({ key: previewKey, error: '资料已更新，正在同步可用版本。' });
        void reload();
      } else if (!data.chunks.length && offset > 0) {
        setSelection({ id: selectedId, offset: 0 });
      } else {
        setPreview({ key: previewKey, data });
      }
    }).catch(error => {
      if (!ignore) setPreview({ key: previewKey, error: error instanceof Error ? error.message : '预览失败。' });
    });
    return () => { ignore = true; };
  }, [courseId, offset, previewKey, reload, selectedId, selectedVersion]);

  async function uploadFiles(files: File[], replacing?: CourseDocument) {
    if (!files.length || busy) return;
    setBusy(true); setActionError(''); setNotice('');
    const failures: string[] = [];
    let uploaded = 0;
    for (let index = 0; index < files.length; index++) {
      const file = files[index];
      setUploadLabel(`正在上传 ${index + 1}/${files.length}：${file.name}`);
      if (file.size > 20 * 1024 * 1024 || file.size === 0) {
        failures.push(`${file.name}：文件为空或超过 20 MB。`); continue;
      }
      try { await documentsApi.upload(courseId, file, replacing?.id); uploaded++; }
      catch (error) { failures.push(`${file.name}：${error instanceof Error ? error.message : '上传失败'}`); }
    }
    if (mounted.current) {
      setBusy(false); setUploadLabel(''); setReplacement(null);
      setActionError(failures.join('\n'));
      if (uploaded) setNotice(`已上传 ${uploaded} 份资料。处理进度见下方列表。`);
      await reload();
    }
  }
  async function fileSelected(event: ChangeEvent<HTMLInputElement>, replacing?: CourseDocument) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    await uploadFiles(files, replacing);
  }
  async function retry(document: CourseDocument) {
    setBusy(true); setActionError('');
    try { await documentsApi.retry(courseId, document.id); await reload(); }
    catch (error) { setActionError(error instanceof Error ? error.message : '重试失败。'); }
    finally { setBusy(false); }
  }
  async function remove() {
    if (!deleting) return;
    setBusy(true); setActionError('');
    try {
      await documentsApi.delete(courseId, deleting.id);
      setDocuments(items => items.filter(item => item.id !== deleting.id));
      setDeleting(null);
      setResults(null);
      await reload();
      setNotice('资料已从课程中移除，原文件与索引将在后台清理。');
    }
    catch (error) { setActionError(error instanceof Error ? error.message : '删除失败。'); }
    finally { setBusy(false); }
  }
  function closePreview() {
    setSelection(null);
    setPreview(null);
    const trigger = previewTrigger.current;
    requestAnimationFrame(() => trigger?.focus({ preventScroll: true }));
  }
  async function find(event: FormEvent) {
    event.preventDefault();
    if (!query.trim() || searching) return;
    setSearching(true); setSearchError(''); setResults(null);
    const requestRevision = revision;
    try {
      const data = await documentsApi.search(courseId, query.trim(), semantic, rerank);
      if (mounted.current && currentRevision.current === requestRevision) setResults(data);
      else if (mounted.current) setSearchError('资料已更新，请重新检索以查看最新版本。');
    }
    catch (error) { if (mounted.current) setSearchError(error instanceof Error ? error.message : '搜索失败。'); }
    finally { if (mounted.current) setSearching(false); }
  }

  return <section className="document-library" aria-labelledby="library-title">
    <div className="library-workspace">
    <div className="library-content">
    <div className="page-heading library-heading"><div><h1 id="library-title">课程资料</h1><p>{readyCount} / {documents.length} 份可用</p></div><button type="button" className="button primary" onClick={() => input.current?.click()} disabled={busy}><Upload size={16} />{busy ? '正在操作…' : '上传资料'}</button></div>
    <input className="file-input" ref={input} type="file" multiple accept=".txt,.md,.html,.htm,.pdf,.docx" aria-label="选择课程资料" onChange={event => void fileSelected(event)} />
    <input className="file-input" ref={replaceInput} type="file" accept=".txt,.md,.html,.htm,.pdf,.docx" aria-label="选择替换文件" onChange={event => void fileSelected(event, replacement ?? undefined)} />
    <div className="upload-zone" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); if (!busy) void uploadFiles(Array.from(event.dataTransfer.files)); }}>
      <Upload size={24} strokeWidth={1.5} /><p>拖入资料，或<button className="text-button" onClick={() => input.current?.click()} disabled={busy}>选择文件</button></p><span>支持 TXT、Markdown、HTML、DOCX 和可提取文字的 PDF，每份不超过 20 MB。上传后自动处理；暂不支持扫描 PDF 文字识别和 PPT。</span>
    </div>
    <div className="embedding-note"><span className={`local-dot ${config && !config.configured ? 'warning-dot' : ''}`} /><span>{config ? `嵌入模型：${config.provider || '未配置'} / ${config.model || '未选择'}` : '正在读取嵌入配置…'}</span></div>
    {config && !config.configured && <p className="error" role="alert">{config.note} 配置后重启后端，再重试失败资料。</p>}
    {config && config.cleanup_pending > 0 && <p className="subtle-note">有 {config.cleanup_pending} 项文件或索引清理正在后台执行，已移除资料不会参与检索。</p>}
    {uploadLabel && <p className="operation-note" role="status"><LoaderCircle className="spin" size={15} />{uploadLabel}</p>}
    {notice && <p className="success-note" role="status">{notice}</p>}
    {actionError && !deleting && <p className="error preserve-lines" role="alert">{actionError}</p>}
    {error && <div className="error-banner" role="alert"><p>{error}</p><button className="button secondary" onClick={() => void reload()}>重新加载</button></div>}
    {loading ? <p className="operation-note"><LoaderCircle className="spin" size={16} />正在读取资料…</p> : documents.length === 0 ? <div className="library-empty"><FolderOpen size={28} strokeWidth={1.3} /><p>课程资料库还是空的。上传第一份笔记，开始积累依据。</p></div> : <div className="document-list">
      {documents.map(document => {
        const pending = ['queued', 'processing'].includes(document.latest.status);
        const failed = document.latest.status === 'failed';
        return <article className={`document-row${selectedId === document.id ? ' is-selected' : ''}`} key={document.id}>
          <div className="document-file-icon"><FileText size={23} strokeWidth={1.4} /><span>{document.latest.filename.split('.').pop()?.toUpperCase()}</span></div>
          <div className="document-body"><h3>{document.name}</h3><p className="document-meta">{sizeLabel(document.latest.size)} <span>版本 {document.latest.number}</span>{document.usable && <span>{document.chunk_count} 个可检索片段</span>}</p>
            <p className={`document-status ${failed ? 'failed' : pending ? 'pending' : 'ready'}`}>{document.latest.status === 'processing' ? <LoaderCircle size={13} className="spin" /> : document.latest.status === 'queued' ? <Clock3 size={13} /> : failed ? <XCircle size={13} /> : <CheckCircle2 size={13} />}{document.latest.step}{document.usable && document.latest.id !== document.active_version_id && <span>继续使用版本 {document.active_version_number}</span>}</p>
            {failed && <p className="document-failure">{document.latest.error}</p>}
          </div>
          <div className="document-actions">
            <button type="button" className="text-button" disabled={!document.usable || busy}
              aria-label={`查看原文 ${document.name}`} aria-expanded={selectedId === document.id}
              onClick={event => {
                previewTrigger.current = event.currentTarget;
                setSelection({ id: document.id, offset: 0 });
              }}>原文</button>
            {failed && <button className="text-button" disabled={busy} onClick={() => void retry(document)}><RefreshCw size={13} />重试</button>}
            <button className="text-button" disabled={pending || busy} onClick={() => { setReplacement(document); replaceInput.current?.click(); }}>替换</button>
            <button className="icon-button danger" aria-label={`删除资料${document.name}`} disabled={busy} onClick={() => { setDeleting(document); setActionError(''); }}><Trash2 size={15} /></button>
          </div>
        </article>;
      })}
    </div>}

    <section className="retrieval-trial" aria-labelledby="search-title"><div className="trial-heading"><h2 id="search-title"><Search size={19} />检索试用</h2><span>查看原文召回，不生成回答</span></div><p>输入关键词或问题，核对这门课的资料能找到什么。</p>
      <form onSubmit={find} className="search-form"><div className="query-row"><input aria-label="资料检索问题" placeholder="例如：进程和线程有什么区别？" maxLength={1000} value={query} onChange={event => setQuery(event.target.value)} disabled={searching} /><button className="button primary" type="submit" disabled={!readyCount || !query.trim() || searching}>{searching ? <LoaderCircle size={16} className="spin" /> : <Search size={16} />}{searching ? '正在检索…' : '检索原文'}</button></div><div className="search-options"><label><input type="checkbox" checked={semantic} onChange={event => setSemantic(event.target.checked)} />结合语义检索</label><label><input type="checkbox" checked={rerank} onChange={event => setRerank(event.target.checked)} />相关度重排</label><span>语义和重排均关闭时，检索仅在本地执行。</span></div></form>
      {searchError && <p className="error" role="alert">{searchError}</p>}
      {results && <div className="search-results" aria-live="polite"><p className="search-summary">找到 {results.results.length} 个片段{results.trace.semantic ? '，已结合语义与关键词检索' : '，使用关键词检索'}{results.trace.reranked && '，已重排'}。</p>{results.trace.note && <p className="subtle-note">{results.trace.note}</p>}{results.results.map((result, position) => <article key={result.id}><h3><span className="citation-number">[{position + 1}]</span>{result.source}<span className="result-location">版本 {result.version}{result.page !== null && ` / 第 ${result.page} 页`}</span></h3><p>{result.text}</p></article>)}{results.results.length === 0 && <p className="subtle-note">没有找到相关片段。试试资料中的关键词，或添加相关文件。</p>}</div>}
    </section>
    </div>
    {selectedDocument ? <ReadingPanel title="资料原文" onClose={closePreview}>
      <h3 className="citation-source-title">{currentPreview?.data?.source ?? selectedDocument.name}</h3>
      <p className="subtle-note">版本 {selectedDocument.active_version_number} · 共 {selectedDocument.chunk_count} 个片段</p>
      {!currentPreview && <p className="operation-note" role="status"><LoaderCircle size={15} className="spin" />正在读取原文…</p>}
      {currentPreview?.error && <div className="error-banner" role="alert">
        <p>{currentPreview.error}</p>
        <button type="button" className="text-button" onClick={() => setPreviewAttempt(attempt => attempt + 1)}>重新读取原文</button>
      </div>}
      {currentPreview?.data && <>
        <div className="preview-chunks">{currentPreview.data.chunks.map(chunk => <article key={chunk.id}>
          <h3>片段 {chunk.ordinal}{chunk.page !== null && ` · 第 ${chunk.page} 页`}</h3>
          <p>{chunk.text}</p>
        </article>)}</div>
        <div className="preview-pagination">
          <button type="button" className="button secondary" disabled={offset === 0} onClick={() => setSelection({ id: selectedDocument.id, offset: Math.max(0, offset - 5) })}><ChevronLeft size={15} />上一组</button>
          <button type="button" className="button secondary" disabled={offset + currentPreview.data.chunks.length >= currentPreview.data.total} onClick={() => setSelection({ id: selectedDocument.id, offset: offset + 5 })}>下一组<ChevronRight size={15} /></button>
        </div>
      </>}
    </ReadingPanel> : <aside className="reading-panel reading-placeholder" aria-label="资料原文">
      <header className="reading-panel-heading"><h2>资料原文</h2></header>
      <p className="reading-panel-body subtle-note">选择一份可用资料的「原文」查看片段。</p>
    </aside>}
    </div>

    {deleting && <Dialog title="删除这份资料？" busy={busy} onClose={() => setDeleting(null)}><p className="dialog-description">「{deleting.name}」及其所有版本会从当前课程移除，之后不再参与检索。后台将清理原文件和向量索引，历史回答中的引用快照仍会保留。此操作无法撤销。</p>{actionError && <p className="error" role="alert">{actionError}</p>}<div className="dialog-actions"><button className="button secondary" disabled={busy} onClick={() => setDeleting(null)}>保留资料</button><button className="button destructive" disabled={busy} onClick={() => void remove()}>{busy ? '正在删除…' : '确认删除资料'}</button></div></Dialog>}
  </section>;
}
