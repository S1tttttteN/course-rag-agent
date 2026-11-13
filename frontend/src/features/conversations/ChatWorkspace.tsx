import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Link, NavLink, useMatch, useNavigate } from 'react-router-dom';
import { ArrowUp, BookOpen, FileText, LoaderCircle, MessageSquare, Pencil, Plus, Settings2, Square, Trash2 } from 'lucide-react';
import { conversationsApi } from '../../api/conversations';
import { Dialog } from '../../components/Dialog';
import { ReadingPanel } from '../../components/ReadingPanel';
import type { ChatConfig, ChatMessage, Citation, Conversation, ConversationDetail, Run, ToolExecution } from '../../types/conversation';
import { Answer } from './Answer';
import { SourceDetails } from './SourceDetails';
import { rememberConversation } from './recentConversation';

const answerPreferencesKey = 'course-study-room.answer-preferences';
const defaultAnswerPreferences = { web_search: false, answer_review: false };
type AnswerPreferences = typeof defaultAnswerPreferences;

function readAnswerPreferences(): AnswerPreferences {
  try {
    const saved = JSON.parse(localStorage.getItem(answerPreferencesKey) ?? 'null');
    return {
      web_search: saved?.web_search === true,
      answer_review: saved?.answer_review === true,
    };
  } catch {
    return defaultAnswerPreferences;
  }
}

export function ChatWorkspace({ courseId, courseName, navigationTarget, onNavigate, library }: {
  courseId: string;
  courseName: string;
  navigationTarget: HTMLElement | null;
  onNavigate: () => void;
  library?: ReactNode;
}) {
  const conversationId = useMatch('/courses/:courseId/conversations/:conversationId')?.params.conversationId;
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [config, setConfig] = useState<ChatConfig | null>(null);
  const [listError, setListError] = useState('');
  const [configError, setConfigError] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [actionError, setActionError] = useState('');
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState('');
  const [topK, setTopK] = useState(4);
  const [semantic, setSemantic] = useState(true);
  const [rerank, setRerank] = useState(true);
  const [mode, setMode] = useState<'rag' | 'agent'>('agent');
  const [answerPreferences, setAnswerPreferences] = useState(readAnswerPreferences);
  const [connectionNote, setConnectionNote] = useState('');
  const [deleting, setDeleting] = useState<Conversation | null>(null);
  const [renaming, setRenaming] = useState<Conversation | null>(null);
  const [title, setTitle] = useState('');
  const [selection, setSelection] = useState<{ runId: string; citation: Citation } | null>(null);
  const [citationError, setCitationError] = useState('');
  const thread = useRef<HTMLDivElement>(null);
  const autoScroll = useRef(true);
  const alive = useRef(true);
  const routeRef = useRef(conversationId);
  const citationRequest = useRef(0);
  const citationTrigger = useRef<HTMLElement | null>(null);
  const currentDetail = detail?.id === conversationId ? detail : null;
  const showingLibrary = library !== undefined;
  const active = currentDetail?.active_run;
  const selectedMessage = currentDetail?.messages.find(message => message.role === 'assistant' && message.run_id === selection?.runId);
  const settingsDisabled = busy || !!active;
  const webSearch = mode === 'agent' && answerPreferences.web_search;
  const answerReview = active?.trace.answer_review ?? answerPreferences.answer_review;

  function changeAnswerPreference(key: keyof AnswerPreferences, checked: boolean) {
    const next = { ...answerPreferences, [key]: checked };
    setAnswerPreferences(next);
    try {
      localStorage.setItem(answerPreferencesKey, JSON.stringify(next));
    } catch {
      // Keep the choice for this page if browser storage is unavailable.
    }
  }

  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const refreshList = useCallback(async () => {
    const items = await conversationsApi.list(courseId);
    if (alive.current) {
      setConversations(items);
      setListError('');
    }
  }, [courseId]);
  const refreshDetail = useCallback(async () => {
    if (!conversationId) return;
    const result = await conversationsApi.detail(courseId, conversationId);
    if (alive.current && routeRef.current === conversationId) {
      setDetail(result);
      rememberConversation(courseId, conversationId);
    }
  }, [courseId, conversationId]);
  const reloadList = useCallback(() => {
    void refreshList().catch(error => {
      if (alive.current) setListError(error instanceof Error ? error.message : '无法读取对话记录。');
    });
  }, [refreshList]);
  useEffect(reloadList, [reloadList]);
  useEffect(() => {
    if (showingLibrary || config) return;
    let ignore = false;
    setConfigError('');
    void conversationsApi.config().then(value => {
      if (ignore) return;
      setConfig(value);
      setTopK(value.top_k);
      setRerank(value.rerank_available);
    }).catch(error => {
      if (!ignore) setConfigError(error instanceof Error ? error.message : '无法读取回答设置。');
    });
    return () => { ignore = true; };
  }, [showingLibrary, config]);
  useEffect(() => {
    routeRef.current = conversationId;
    setDetail(null); setError(''); setActionError(''); setDraft(''); setConnectionNote('');
    setSelection(null); citationRequest.current++; autoScroll.current = true;
    citationTrigger.current = null;
    if (!conversationId) { setLoading(false); return; }
    setLoading(true);
    void refreshDetail().catch(error => {
      if (alive.current && routeRef.current === conversationId) setError(error instanceof Error ? error.message : '无法读取这段对话。');
    }).finally(() => { if (alive.current && routeRef.current === conversationId) setLoading(false); });
  }, [conversationId, refreshDetail]);

  const runId = active?.id;
  useEffect(() => {
    if (!runId || !conversationId) return;
    const source = conversationsApi.events(courseId, conversationId, runId, active!.last_event_id);
    let disposed = false;
    source.onopen = () => { if (!disposed) setConnectionNote(''); };
    source.addEventListener('status', event => {
      const data = JSON.parse((event as MessageEvent).data) as Pick<Run, 'status' | 'phase'>;
      if (!disposed) setDetail(previous => previous?.id === conversationId ? {...previous, active_run: previous.active_run ? {...previous.active_run, ...data} : null} : previous);
    });
    source.addEventListener('delta', event => {
      const data = JSON.parse((event as MessageEvent).data) as {text: string};
      if (!disposed) setDetail(previous => previous?.id === conversationId ? {...previous, messages: previous.messages.map(message => message.run_id === runId && message.role === 'assistant' ? {...message, content: message.content + data.text, status: 'running'} : message)} : previous);
    });
    source.addEventListener('citations', event => {
      const data = JSON.parse((event as MessageEvent).data) as {citations: Citation[]};
      if (!disposed) setDetail(previous => previous?.id === conversationId ? {...previous, messages: previous.messages.map(message => message.run_id === runId && message.role === 'assistant' ? {...message, citations: data.citations} : message)} : previous);
    });
    source.addEventListener('tool', event => {
      const data = JSON.parse((event as MessageEvent).data) as ToolExecution;
      if (!disposed) setDetail(previous => previous?.id === conversationId ? {...previous, messages: previous.messages.map(message => {
        if (message.run_id !== runId || message.role !== 'assistant' || !message.run) return message;
        const tools = message.run.trace.tools ?? [];
        const updated = tools.some(tool => tool.id === data.id) ? tools.map(tool => tool.id === data.id ? data : tool) : [...tools, data];
        return {...message, run: {...message.run, trace: {...message.run.trace, tools: updated}}};
      })} : previous);
    });
    source.addEventListener('done', () => {
      source.close();
      if (!disposed) void Promise.all([refreshDetail(), refreshList()]).catch(error => setActionError(error instanceof Error ? error.message : '回答状态同步失败，请刷新页面。'));
    });
    source.onerror = () => { if (!disposed) setConnectionNote('实时连接暂时中断，正在重新连接；刷新页面也可恢复已保存内容。'); };
    return () => { disposed = true; source.close(); };
    // The cursor is the coherent body snapshot at attachment; do not reopen for each delta.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, courseId, conversationId, refreshDetail, refreshList]);

  useEffect(() => {
    if (autoScroll.current && thread.current) thread.current.scrollTop = thread.current.scrollHeight;
  }, [detail?.messages, active?.phase]);

  async function createConversation() {
    if (busy) return;
    onNavigate();
    setBusy(true); setActionError('');
    try {
      const item = await conversationsApi.create(courseId);
      if (alive.current) {
        setConversations(previous => [item, ...previous]);
        navigate(`/courses/${courseId}/conversations/${item.id}`);
      }
    } catch (error) { setActionError(error instanceof Error ? error.message : '创建失败。'); }
    finally { if (alive.current) setBusy(false); }
  }
  async function send(question = draft) {
    if (busy || active || !question.trim() || loading) return;
    const origin = routeRef.current;
    setBusy(true); setActionError(''); autoScroll.current = true;
    try {
      let id = conversationId;
      if (!id) {
        const item = await conversationsApi.create(courseId);
        id = item.id;
      }
      await conversationsApi.ask(courseId, id, {
        question: question.trim(),
        top_k: topK,
        semantic,
        rerank,
        mode,
        web_search: webSearch,
        answer_review: answerPreferences.answer_review,
      });
      if (alive.current && routeRef.current === origin) {
        setDraft('');
        await refreshList();
        if (conversationId) await refreshDetail();
        else navigate(`/courses/${courseId}/conversations/${id}`);
      }
    } catch (error) { if (alive.current) setActionError(error instanceof Error ? error.message : '发送失败，请重试。'); }
    finally { if (alive.current) setBusy(false); }
  }
  async function cancel() {
    if (!active || !conversationId) return;
    try { await conversationsApi.cancel(courseId, conversationId, active.id); }
    catch (error) { setActionError(error instanceof Error ? error.message : '停止失败，请重试。'); }
  }
  async function deleteConversation() {
    if (!deleting) return;
    setBusy(true); setActionError('');
    try {
      await conversationsApi.delete(courseId, deleting.id);
      if (deleting.id === conversationId) navigate(`/courses/${courseId}`);
      setDeleting(null); await refreshList();
    } catch (error) { setActionError(error instanceof Error ? error.message : '删除失败。'); }
    finally { setBusy(false); }
  }
  async function rename(event: FormEvent) {
    event.preventDefault();
    if (!renaming || !title.trim()) return;
    setBusy(true); setActionError('');
    try {
      const updated = await conversationsApi.rename(courseId, renaming.id, title.trim());
      setDetail(previous => previous?.id === updated.id ? { ...previous, title: updated.title } : previous);
      setRenaming(null); await refreshList();
    } catch (error) { setActionError(error instanceof Error ? error.message : '修改失败。'); }
    finally { setBusy(false); }
  }
  async function openCitation(message: ChatMessage, item: Citation, trigger?: HTMLElement) {
    if (!conversationId) return;
    const request = ++citationRequest.current;
    if (trigger) citationTrigger.current = trigger;
    setSelection({ runId: message.run_id, citation: item });
    setCitationError('');
    try {
      const result = await conversationsApi.citation(courseId, conversationId, item.id);
      if (alive.current && request === citationRequest.current && routeRef.current === conversationId) {
        setSelection({ runId: message.run_id, citation: result });
      }
    } catch (error) { if (alive.current && request === citationRequest.current) setCitationError(error instanceof Error ? error.message : '来源状态暂时无法更新。'); }
  }

  function closeCitation() {
    citationRequest.current++;
    setSelection(null);
    const trigger = citationTrigger.current;
    requestAnimationFrame(() => trigger?.focus({ preventScroll: true }));
  }

  const prompts = courseName.includes('操作系统') ? ['PCB 包含哪些信息？', '进程与线程有哪些区别？', '解释进程的状态转换。'] : ['梳理资料中的核心概念。', '比较资料中容易混淆的概念。', '这门课的资料主要讲了什么？'];
  return <>
    {navigationTarget && createPortal(
      <nav className="conversation-list" aria-label="本课程对话">
        <div className="rail-heading">
          <h2>对话记录</h2>
          <button type="button" className="icon-button" onClick={() => void createConversation()} disabled={busy} aria-label="新建对话"><Plus size={17} /></button>
        </div>
        {conversations.map(item => <div className={`conversation-entry${item.id === conversationId ? ' is-selected' : ''}`} key={item.id}>
          <NavLink className="nav-row" to={`/courses/${courseId}/conversations/${item.id}`} onClick={onNavigate}>
            <MessageSquare size={15} /><span>{item.title}</span>
          </NavLink>
          <div className="conversation-actions">
            <button type="button" className="icon-button" aria-label={`重命名对话${item.title}`} disabled={busy}
              onClick={() => { setRenaming(item); setTitle(item.title); setActionError(''); }}><Pencil size={13} /></button>
            <button type="button" className="icon-button danger" aria-label={`删除对话${item.title}`} disabled={busy}
              onClick={() => { setDeleting(item); setActionError(''); }}><Trash2 size={13} /></button>
          </div>
        </div>)}
        {listError ? <div className="rail-error">
          <p className="error" role="alert">{listError}</p>
          <button type="button" className="text-button" onClick={reloadList}>重新加载对话</button>
        </div> : conversations.length === 0 && <p className="subtle-note">还没有对话。</p>}
      </nav>,
      navigationTarget,
    )}
    {library ?? <div className={`chat-workspace${selectedMessage ? ' has-reading-panel' : ''}`}>
    <section className="chat-thread" aria-labelledby="conversation-title">
      <header className="thread-heading">
        <h1 id="conversation-title">{currentDetail?.title ?? '学习对话'}</h1>
      </header>
      <div className="thread-notices">
        {configError && <p className="error" role="alert">{configError}</p>}
        {config && !config.configured && <p className="error" role="alert">{config.note}</p>}
        {error && <div className="error-banner" role="alert"><p>{error}</p><button className="button secondary" onClick={() => { setError(''); void refreshDetail().catch(error => setError(error.message)); }}>重新加载</button></div>}
      </div>
      <div className="thread-messages" ref={thread} aria-label="对话消息" onScroll={() => { if (thread.current) autoScroll.current = thread.current.scrollHeight - thread.current.scrollTop - thread.current.clientHeight < 90; }}>
        {loading ? <p className="operation-note" role="status"><LoaderCircle size={17} className="spin" />正在恢复对话…</p> : !currentDetail?.messages.length && !error ? <div className="chat-empty">
          <BookOpen size={23} /><h2>从一个课程问题开始</h2><p>回答中的引用可以打开原文摘录。</p>
          <div className="chat-suggestions">{prompts.map(prompt => <button type="button" className="button secondary" key={prompt} disabled={busy} onClick={() => void send(prompt)}>{prompt}</button>)}</div>
          <Link to={`/courses/${courseId}/documents`} className="text-button"><FileText size={15} />查看课程资料</Link>
        </div> : currentDetail?.messages.map(message => <article className={`chat-message ${message.role}`} key={message.id}><div className="message-author">{message.role === 'user' ? '你的问题' : <><BookOpen size={14} />课程助手</>}</div>
          {message.role === 'user' ? <p className="user-text">{message.content}</p> : <>
            {message.run && <span className="answer-mode">{message.run.mode === 'agent' ? 'Agent · 自主查阅' : 'RAG · 固定检索'}</span>}
            {!!message.run?.trace.tools?.length && <details className="tool-records" open={active?.id === message.run_id}><summary>查阅记录 · {message.run.trace.tools.length} / {message.run.trace.tool_limit ?? 8}</summary><ol>{message.run.trace.tools.map(tool => <li key={tool.id} className={`tool-record ${tool.status}`}><div><strong>{tool.number}. {tool.label}</strong><span>{tool.status === 'running' ? '执行中' : tool.status === 'completed' ? '完成' : tool.status === 'limited' ? '已达上限' : tool.status === 'interrupted' ? '已中断' : '未成功'}{tool.duration_ms !== null && ` · ${(tool.duration_ms / 1000).toFixed(1)}s`}</span></div>{tool.input.query && <p>{tool.input.query}</p>}{tool.name === 'read_document' && <p>片段偏移 {tool.input.offset ?? '0'} · 读取 {tool.input.limit ?? '3'} 段</p>}<small>{tool.summary}</small></li>)}</ol></details>}
            {message.content ? <Answer text={message.content} citations={message.citations}
              selectedCitationId={selection?.runId === message.run_id ? selection.citation.id : undefined}
              onCitation={(item, event) => void openCitation(message, item, event.currentTarget)} />
              : <p className="message-placeholder">{active?.id === message.run_id ? active.phase : '本次未生成正文。'}</p>}
            {message.citations.length > 0 && <button type="button" className="button secondary message-sources"
              aria-expanded={selection?.runId === message.run_id}
              onClick={event => void openCitation(message, message.citations[0], event.currentTarget)}>
              <FileText size={15} />参考来源 <span>{message.citations.length}</span>
            </button>}
            {message.run?.trace.unknown_citations?.length ? <p className="message-warning">部分引用编号未匹配，请结合可用的参考来源核对。</p> : null}
            {message.run?.trace.note && <p className="subtle-note">{message.run.trace.note}</p>}
            {message.run?.trace.evidence_check?.status === 'checked' && <p className="subtle-note">已核对结论与原文摘录，保留 {message.run.trace.evidence_check.kept} 条；核对仍可能遗漏，请结合引用判断。</p>}
            {message.run?.trace.rewrite_note && <p className="subtle-note">{message.run.trace.rewrite_note}</p>}
            {['failed', 'cancelled', 'interrupted'].includes(message.status) && <p className="message-warning">{message.status === 'cancelled' ? '已停止。部分内容已保存，不作为后续完整问答依据。' : message.run?.error || '回答未完成，请重新提问。'}</p>}
          </>}
        </article>)}
      </div>
      <div className="composer-area">{active && <p className="run-status" role="status"><LoaderCircle size={13} className="spin" />{active.phase}</p>}{connectionNote && <p className="subtle-note" role="status">{connectionNote}</p>}{actionError && !deleting && !renaming && <p className="error" role="alert">{actionError}</p>}
        <form onSubmit={event => { event.preventDefault(); void send(); }} className="chat-composer"><textarea aria-label="课程问题" placeholder="问一个课程问题，或继续追问…" maxLength={4000} rows={2} value={draft} onChange={event => setDraft(event.target.value)} disabled={busy || !!active || loading || !!error} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } }} />{active ? <button type="button" className="send-button stop-button" onClick={() => void cancel()} disabled={active.status === 'cancelling'} aria-label="停止回答"><Square size={16} fill="currentColor" /></button> : <button type="submit" className="send-button" disabled={!draft.trim() || busy || loading || !!error} aria-label="发送问题">{busy ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={20} />}</button>}</form>
        <div className="composer-foot">
          <div className="mode-picker" role="group" aria-label="回答模式">
            <button type="button" aria-pressed={mode === 'agent'} disabled={settingsDisabled} onClick={() => setMode('agent')}>Agent</button>
            <button type="button" aria-pressed={mode === 'rag'} disabled={settingsDisabled} onClick={() => setMode('rag')}>固定 RAG</button>
          </div>
          <span className="composer-status">{mode === 'rag' ? '仅课程资料' : webSearch ? '可按需联网' : '联网关闭'} · {answerReview ? '正文核对后显示' : '回答实时显示'}</span>
          <details className="chat-settings">
            <summary><Settings2 size={15} />回答设置</summary>
            <div>
              <p>{config ? `${config.provider || '未配置'} / ${config.model || '未选择'}` : '读取模型配置…'}</p>
              <label>每次检索片段数 {mode === 'agent' ? Math.min(topK, 4) : topK}<input type="range" min="1" max={mode === 'agent' ? '4' : '10'} value={mode === 'agent' ? Math.min(topK, 4) : topK} disabled={settingsDisabled} onChange={event => setTopK(Number(event.target.value))} /></label>
              <label><input type="checkbox" checked={semantic} disabled={settingsDisabled} onChange={event => setSemantic(event.target.checked)} />结合语义检索</label>
              <label><input type="checkbox" checked={rerank} disabled={settingsDisabled || !config?.rerank_available} onChange={event => setRerank(event.target.checked)} />相关度重排</label>
              <label>
                <input type="checkbox" checked={webSearch} disabled={settingsDisabled || mode === 'rag'} aria-describedby="web-search-note" onChange={event => changeAnswerPreference('web_search', event.target.checked)} />
                允许 Agent 联网
              </label>
              <p id="web-search-note">{mode === 'rag' ? '仅 Agent 模式可用。' : '搜索词可能包含问题或资料信息，并发送给搜索服务。'}</p>
              <label>
                <input type="checkbox" checked={answerPreferences.answer_review} disabled={settingsDisabled} onChange={event => changeAnswerPreference('answer_review', event.target.checked)} />
                生成后核对证据
              </label>
            </div>
          </details>
        </div>
      </div>
    </section>
    {selection && selectedMessage && <ReadingPanel title="本轮引用" onClose={closeCitation}>
      <ul className="citation-list">
        {selectedMessage.citations.map(item => <li key={item.id}>
          <button type="button" className="nav-row" aria-current={item.id === selection.citation.id ? 'true' : undefined}
            onClick={() => void openCitation(selectedMessage, item)}>
            <span className="citation-number">[{item.number}]</span><span>{item.source}</span>
          </button>
        </li>)}
      </ul>
      <section className="source-excerpt" aria-labelledby="excerpt-title">
        <h3 id="excerpt-title">原文摘录</h3>
        <SourceDetails citation={selection.citation} />
        {citationError && <p className="error" role="alert">{citationError}</p>}
      </section>
    </ReadingPanel>}
    </div>}
    {showingLibrary && actionError && !deleting && !renaming && <p className="toast" role="alert">{actionError}</p>}
    {deleting && <Dialog title="删除这段对话？" busy={busy} onClose={() => setDeleting(null)}><p className="dialog-description">「{deleting.title}」的消息与引用快照会一并删除，正在生成的回答也会停止。课程资料不受影响，此操作无法撤销。</p>{actionError && <p className="error" role="alert">{actionError}</p>}<div className="dialog-actions"><button className="button secondary" disabled={busy} onClick={() => setDeleting(null)}>保留对话</button><button className="button destructive" disabled={busy} onClick={() => void deleteConversation()}>确认删除对话</button></div></Dialog>}
    {renaming && <Dialog title="重命名对话" busy={busy} onClose={() => setRenaming(null)}><form onSubmit={event => void rename(event)}><label htmlFor="conversation-name">对话名称</label><input id="conversation-name" maxLength={80} required value={title} disabled={busy} onChange={event => setTitle(event.target.value)} />{actionError && <p className="error" role="alert">{actionError}</p>}<div className="dialog-actions"><button className="button secondary" type="button" onClick={() => setRenaming(null)} disabled={busy}>取消</button><button className="button primary" disabled={busy || !title.trim()} type="submit">保存名称</button></div></form></Dialog>}
  </>;
}
