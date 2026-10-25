import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { Link, NavLink, useNavigate, useParams } from 'react-router-dom';
import { ArrowUp, BookOpen, FileText, LoaderCircle, MessageSquare, Pencil, Plus, Square, Trash2 } from 'lucide-react';
import { conversationsApi } from '../../api/conversations';
import { Dialog } from '../../components/Dialog';
import type { ChatConfig, Citation, Conversation, ConversationDetail, Run, ToolExecution } from '../../types/conversation';
import { Answer } from './Answer';
import { SourceDetails } from './SourceDetails';

export function ChatWorkspace({ courseId, courseName }: {courseId: string; courseName: string}) {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [config, setConfig] = useState<ChatConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [actionError, setActionError] = useState('');
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState('');
  const [topK, setTopK] = useState(4);
  const [semantic, setSemantic] = useState(true);
  const [rerank, setRerank] = useState(true);
  const [mode, setMode] = useState<'rag' | 'agent'>('agent');
  const [connectionNote, setConnectionNote] = useState('');
  const [deleting, setDeleting] = useState<Conversation | null>(null);
  const [renaming, setRenaming] = useState<Conversation | null>(null);
  const [title, setTitle] = useState('');
  const [citation, setCitation] = useState<Citation | null>(null);
  const [citationError, setCitationError] = useState('');
  const streamRef = useRef<EventSource | null>(null);
  const thread = useRef<HTMLDivElement>(null);
  const autoScroll = useRef(true);
  const alive = useRef(true);
  const routeRef = useRef(conversationId);
  const citationRequest = useRef(0);
  const active = detail && detail.id === conversationId ? detail.active_run : null;

  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const refreshList = useCallback(async () => {
    const items = await conversationsApi.list(courseId);
    if (alive.current) setConversations(items);
  }, [courseId]);
  const refreshDetail = useCallback(async () => {
    if (!conversationId) return;
    const result = await conversationsApi.detail(courseId, conversationId);
    if (alive.current && routeRef.current === conversationId) setDetail(result);
  }, [courseId, conversationId]);
  useEffect(() => {
    void Promise.all([refreshList(), conversationsApi.config().then(value => {
      if (alive.current) { setConfig(value); setTopK(value.top_k); setRerank(value.rerank_available); }
    })]).catch(error => { if (alive.current) setError(error instanceof Error ? error.message : '对话加载失败。'); });
  }, [refreshList]);
  useEffect(() => {
    routeRef.current = conversationId;
    setDetail(null); setError(''); setActionError(''); setDraft(''); setConnectionNote('');
    setCitation(null); citationRequest.current++; autoScroll.current = true;
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
    streamRef.current = source;
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
    return () => { disposed = true; source.close(); if (streamRef.current === source) streamRef.current = null; };
    // The cursor is the coherent body snapshot at attachment; do not reopen for each delta.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, courseId, conversationId, refreshDetail, refreshList]);

  useEffect(() => {
    if (autoScroll.current && thread.current) thread.current.scrollTop = thread.current.scrollHeight;
  }, [detail?.messages, active?.phase]);

  async function createConversation() {
    if (busy) return;
    setBusy(true); setActionError('');
    try {
      const item = await conversationsApi.create(courseId);
      if (alive.current) { await refreshList(); navigate(`/courses/${courseId}/conversations/${item.id}`); }
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
      await conversationsApi.ask(courseId, id, question.trim(), topK, semantic, rerank, mode);
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
      await conversationsApi.rename(courseId, renaming.id, title.trim());
      setRenaming(null); await refreshList(); await refreshDetail();
    } catch (error) { setActionError(error instanceof Error ? error.message : '修改失败。'); }
    finally { setBusy(false); }
  }
  async function openCitation(item: Citation) {
    if (!conversationId) return;
    const request = ++citationRequest.current;
    setCitation(item); setCitationError('');
    try {
      const result = await conversationsApi.citation(courseId, conversationId, item.id);
      if (alive.current && request === citationRequest.current) setCitation(result);
    } catch (error) { if (alive.current && request === citationRequest.current) setCitationError(error instanceof Error ? error.message : '来源状态暂时无法更新。'); }
  }

  const prompts = courseName.includes('操作系统') ? ['PCB 包含哪些信息？', '进程与线程有哪些区别？', '解释进程的状态转换。'] : ['梳理资料中的核心概念。', '比较资料中容易混淆的概念。', '这门课的资料主要讲了什么？'];
  return <div className="chat-workspace">
    <aside className="conversation-list" aria-label="本课程对话"><div className="conversation-list-heading"><h2>学习对话</h2><button className="icon-button" onClick={() => void createConversation()} disabled={busy} aria-label="新建对话"><Plus size={18} /></button></div><p>每个问题，都可以接着聊。</p><button className="button secondary new-conversation" onClick={() => void createConversation()} disabled={busy}><Plus size={15} />开启新对话</button>
      <nav>{conversations.map(item => <div className="conversation-entry" key={item.id}><NavLink to={`/courses/${courseId}/conversations/${item.id}`}><MessageSquare size={14} /><span>{item.title}</span></NavLink><div><button className="icon-button" aria-label={`重命名对话${item.title}`} disabled={busy} onClick={() => { setRenaming(item); setTitle(item.title); setActionError(''); }}><Pencil size={12} /></button><button className="icon-button danger" aria-label={`删除对话${item.title}`} disabled={busy} onClick={() => { setDeleting(item); setActionError(''); }}><Trash2 size={12} /></button></div></div>)}</nav>{conversations.length === 0 && <p className="conversation-empty">还没有对话。<br />从右侧的一个问题开始。</p>}
    </aside>
    <section className="chat-thread"><header className="thread-heading"><div><h2>{detail?.title ?? '与课程资料对话'}</h2><p>围绕「{courseName}」学习，{mode === 'rag' ? '先检索课程资料，再生成回答。' : config ? config.web_search ? '按需查阅课程资料、搜索网络和阅读网页。' : '按需搜索和阅读课程资料。' : '正在读取查阅配置…'}</p></div><Link to={`/courses/${courseId}/documents`} className="text-button"><FileText size={14} />管理资料</Link></header>
      {config && !config.configured && <p className="error" role="alert">{config.note}</p>}
      {error && <div className="error-banner" role="alert"><p>{error}</p><button className="button secondary" onClick={() => { setError(''); void refreshDetail().catch(error => setError(error.message)); }}>重新加载</button></div>}
      <div className="thread-messages" ref={thread} aria-label="对话消息" onScroll={() => { if (thread.current) autoScroll.current = thread.current.scrollHeight - thread.current.scrollTop - thread.current.clientHeight < 90; }}>
        {loading ? <p className="operation-note"><LoaderCircle size={17} className="spin" />正在恢复对话…</p> : !detail?.messages.length && !error ? <div className="chat-empty"><span className="chat-empty-symbol">[n]</span><h3>有问题，就回到笔记。</h3><p>理解一个概念，理清一组区别，<br />查阅所得的来源会随回答保存。</p><div className="chat-suggestions">{prompts.map(prompt => <button key={prompt} disabled={busy} onClick={() => void send(prompt)}>{prompt}</button>)}</div></div> : detail?.messages.map(message => <article className={`chat-message ${message.role}`} key={message.id}><div className="message-author">{message.role === 'user' ? '你' : <><BookOpen size={14} />课程助手</>}</div>
          {message.role === 'user' ? <p className="user-text">{message.content}</p> : <>
            {message.run && <span className="answer-mode">{message.run.mode === 'agent' ? 'Agent · 自主查阅' : 'RAG · 固定检索'}</span>}
            {!!message.run?.trace.tools?.length && <details className="tool-records" open={active?.id === message.run_id}><summary>查阅记录 · {message.run.trace.tools.length} / {message.run.trace.tool_limit ?? 8}</summary><ol>{message.run.trace.tools.map(tool => <li key={tool.id} className={`tool-record ${tool.status}`}><div><strong>{tool.number}. {tool.label}</strong><span>{tool.status === 'running' ? '执行中' : tool.status === 'completed' ? '完成' : tool.status === 'limited' ? '已达上限' : tool.status === 'interrupted' ? '已中断' : '未成功'}{tool.duration_ms !== null && ` · ${(tool.duration_ms / 1000).toFixed(1)}s`}</span></div>{tool.input.query && <p>{tool.input.query}</p>}{tool.name === 'read_document' && <p>片段偏移 {tool.input.offset ?? '0'} · 读取 {tool.input.limit ?? '3'} 段</p>}<small>{tool.summary}</small></li>)}</ol></details>}
            {message.content ? <Answer text={message.content} citations={message.citations} onCitation={item => void openCitation(item)} /> : <p className="message-placeholder">{active?.id === message.run_id ? active.phase : '本次未生成正文。'}</p>}
            {message.citations.length > 0 && <details className="message-sources"><summary>参考来源 · {message.citations.length}</summary>{message.citations.map(item => <button key={item.id} onClick={() => void openCitation(item)}><span>[{item.number}]</span>{item.source}<small>{item.kind === 'course' ? `课程资料 / 版本 ${item.version}${item.page !== null ? ` / 第 ${item.page} 页` : ''}` : item.kind === 'web_snippet' ? '网络搜索摘要' : '网页正文摘录'}</small></button>)}</details>}
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
        <div className="mode-picker" role="group" aria-label="回答模式"><button type="button" aria-pressed={mode === 'agent'} disabled={!!active || busy} onClick={() => setMode('agent')}>Agent · 自主查阅</button><button type="button" aria-pressed={mode === 'rag'} disabled={!!active || busy} onClick={() => setMode('rag')}>RAG · 固定检索</button><span>{mode === 'agent' ? config ? config.web_search ? '按需查阅课程资料、搜索网络和阅读网页。' : '按需搜索和阅读课程资料，联网已关闭。' : '正在读取查阅配置…' : '检索一次课程资料后回答，不主动联网。'}</span></div>
        <div className="composer-foot"><span>{config?.answer_review ? '正文核对后显示' : '回答实时显示'}；有引用时可点击 [n] 核对来源。</span><details className="chat-settings"><summary>回答设置</summary><div><p>{config ? `${config.provider || '未配置'} / ${config.model || '未选择'}` : '读取模型配置…'}</p><label>每次检索片段数 {mode === 'agent' ? Math.min(topK, 4) : topK}<input type="range" min="1" max={mode === 'agent' ? '4' : '10'} value={mode === 'agent' ? Math.min(topK, 4) : topK} disabled={!!active} onChange={event => setTopK(Number(event.target.value))} /></label><label><input type="checkbox" checked={semantic} disabled={!!active} onChange={event => setSemantic(event.target.checked)} />结合语义检索</label><label><input type="checkbox" checked={rerank} disabled={!!active || !config?.rerank_available} onChange={event => setRerank(event.target.checked)} />相关度重排</label></div></details></div>
      </div>
    </section>
    {deleting && <Dialog title="删除这段对话？" busy={busy} onClose={() => setDeleting(null)}><p className="dialog-description">「{deleting.title}」的消息与引用快照会一并删除，正在生成的回答也会停止。课程资料不受影响，此操作无法撤销。</p>{actionError && <p className="error" role="alert">{actionError}</p>}<div className="dialog-actions"><button className="button secondary" disabled={busy} onClick={() => setDeleting(null)}>保留对话</button><button className="button destructive" disabled={busy} onClick={() => void deleteConversation()}>确认删除对话</button></div></Dialog>}
    {renaming && <Dialog title="重命名对话" busy={busy} onClose={() => setRenaming(null)}><form onSubmit={event => void rename(event)}><label htmlFor="conversation-title">对话名称</label><input id="conversation-title" maxLength={80} required value={title} disabled={busy} onChange={event => setTitle(event.target.value)} />{actionError && <p className="error" role="alert">{actionError}</p>}<div className="dialog-actions"><button className="button secondary" type="button" onClick={() => setRenaming(null)} disabled={busy}>取消</button><button className="button primary" disabled={busy || !title.trim()} type="submit">保存名称</button></div></form></Dialog>}
    {citation && <Dialog title={`引用 [${citation.number}]`} busy={false} onClose={() => { citationRequest.current++; setCitation(null); }}><SourceDetails citation={citation} />{citationError && <p className="error" role="alert">{citationError}</p>}</Dialog>}
  </div>;
}
