import { useCallback, useEffect, useState } from 'react';
import { Link, NavLink, Route, Routes, useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowUpRight, BookOpen, Check, FileText, FolderOpen, GraduationCap, LoaderCircle, MessageSquare, Pencil, Plus, RefreshCw, Search, Trash2 } from 'lucide-react';
import { coursesApi } from '../api/courses';
import { Dialog } from '../components/Dialog';
import { CourseEditor } from '../features/courses/CourseEditor';
import { DocumentLibrary } from '../features/documents/DocumentLibrary';
import { ChatWorkspace } from '../features/conversations/ChatWorkspace';
import type { Course, CourseInput } from '../types/course';

function dateLabel(value: string) {
  return new Intl.DateTimeFormat('zh-CN', { month: 'long', day: 'numeric' }).format(new Date(value));
}

export function App() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [editor, setEditor] = useState<{ course: Course | null } | null>(null);
  const [deleting, setDeleting] = useState<Course | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const [notice, setNotice] = useState('');
  const [search, setSearch] = useState('');
  const navigate = useNavigate();

  const reload = useCallback(async () => {
    setLoading(true);
    setError('');
    try { setCourses(await coursesApi.list()); }
    catch (error) { setError(error instanceof Error ? error.message : '课程加载失败。'); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 4000);
    return () => clearTimeout(timer);
  }, [notice]);

  async function save(input: CourseInput) {
    const editing = editor?.course;
    const saved = editing ? await coursesApi.update(editing.id, input) : await coursesApi.create(input);
    setCourses(previous => [saved, ...previous.filter(course => course.id !== saved.id)]);
    setEditor(null);
    setNotice(editing ? '课程已更新' : '课程已创建');
    if (!editing) navigate(`/courses/${saved.id}`);
  }
  async function remove() {
    if (!deleting || deleteBusy) return;
    setDeleteBusy(true);
    setDeleteError('');
    try {
      await coursesApi.delete(deleting.id);
      setCourses(previous => previous.filter(course => course.id !== deleting.id));
      setDeleting(null);
      navigate('/');
      setNotice('课程已删除');
    } catch (error) { setDeleteError(error instanceof Error ? error.message : '删除失败，请重试。'); }
    finally { setDeleteBusy(false); }
  }

  return <div className="app-shell">
    <a href="#main" className="skip-link">跳到主要内容</a>
    <aside className="sidebar">
      <Link to="/" className="brand"><span className="brand-symbol">[n]</span><span><strong>课程研习室</strong><small>带着问题，回到笔记。</small></span></Link>
      <NavLink to="/" end className="nav-home"><BookOpen size={18} />我的课程<span>{courses.length}</span></NavLink>
      <div className="sidebar-section"><span>课程目录</span><button className="icon-button" onClick={() => setEditor({ course: null })} aria-label="新建课程"><Plus size={17} /></button></div>
      <nav className="course-nav" aria-label="课程目录">
        {courses.map(course => <NavLink key={course.id} to={`/courses/${course.id}`}><span className="course-dot" /><span>{course.name}</span></NavLink>)}
        {!loading && courses.length === 0 && <p className="sidebar-empty">还没有课程。<br />从一门你想学懂的课开始。</p>}
      </nav>
      <div className="sidebar-foot"><span className="local-dot" />本地学习空间<p>课程与记录保存在本机数据目录。</p></div>
    </aside>

    <div className="main-shell">
      <header className="topbar"><span><GraduationCap size={17} />学习工作台</span><span className="version-label">课程研习室 <span className="phase-dot" /> 本地实验版</span></header>
      <main id="main">
        {error && <div className="error-banner" role="alert"><div><strong>暂时无法加载课程</strong><p>{error}</p></div><button className="button secondary" onClick={() => void reload()}><RefreshCw size={15} />重试</button></div>}
        {loading ? <div className="loading" role="status"><LoaderCircle size={22} className="spin" />正在打开学习空间…</div> : <Routes>
          <Route path="/" element={<>
            <section className="welcome">
              <div><p className="welcome-note">你的笔记，值得再读一遍。</p><h1>把课程收好，<br />把问题想透。</h1><p className="welcome-description">按课程整理学习资料，在回答中追溯参考来源。<br />先创建课程，建立属于你的学习空间。</p><button className="button primary" onClick={() => setEditor({ course: null })}><Plus size={17} />创建课程</button></div>
              <div className="folio" aria-hidden="true"><div className="folio-back" /><div className="folio-page"><BookOpen size={24} strokeWidth={1.4} /><p>学有所问<br />问有所据</p><div className="folio-lines"><i /><i /><i /></div><span>我的课程笔记</span></div><div className="folio-cite">[1]</div></div>
            </section>
            <section className="course-section" aria-labelledby="courses-heading">
              <div className="section-header"><div><h2 id="courses-heading">我的课程 <span>{courses.length}</span></h2><p>每门课程，都有自己的资料与对话。</p></div><label className="search"><Search size={16} /><input aria-label="搜索课程" placeholder="查找课程" value={search} onChange={event => setSearch(event.target.value)} /></label></div>
              {courses.length === 0 ? <div className="empty-library"><div className="empty-icon"><FolderOpen size={28} strokeWidth={1.4} /></div><h3>第一门课，从这里开始</h3><p>给它起个名字。操作系统、计算机网络，<br />或者任何你正在学习的主题。</p><button className="button secondary" onClick={() => setEditor({ course: null })}><Plus size={16} />创建第一门课程</button></div> : <div className="course-grid">
                {courses.filter(course => `${course.name} ${course.description}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())).map((course, index) => <article className={`course-card tone-${index % 3}`} key={course.id}>
                  <div className="course-card-heading"><span className="book-tile"><BookOpen size={22} strokeWidth={1.5} /></span><div className="card-tools"><button className="icon-button" aria-label={`编辑${course.name}`} onClick={() => setEditor({ course })}><Pencil size={15} /></button><button className="icon-button danger" aria-label={`删除${course.name}`} onClick={() => { setDeleting(course); setDeleteError(''); }}><Trash2 size={15} /></button></div></div>
                  <Link className="course-card-link" to={`/courses/${course.id}`}><h3>{course.name}</h3><p>{course.description || '还没有简介，留一点空间给新的知识。'}</p><div className="card-bottom"><span>{dateLabel(course.updated_at)}更新</span><ArrowUpRight size={19} /></div></Link>
                </article>)}
                {search.trim() && !courses.some(course => `${course.name} ${course.description}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())) && <p className="no-results">没有找到对应课程。试试其他名称或清空搜索。</p>}
              </div>}
            </section>
            <p className="development-note"><span className="phase-dot" />进入课程，整理资料并开启对话。可选择 Agent 自主查阅或 RAG 固定检索，点击回答中的引用核对来源。</p>
          </>} />
          <Route path="/courses/:courseId/*" element={<CourseWorkspace courses={courses} failed={!!error} onEdit={course => setEditor({ course })} onDelete={course => { setDeleting(course); setDeleteError(''); }} />} />
          <Route path="*" element={<div className="not-found"><h1>这里还没有内容</h1><Link className="button secondary" to="/">返回我的课程</Link></div>} />
        </Routes>}
      </main>
    </div>
    {notice && <div className="toast" role="status"><Check size={17} />{notice}</div>}
    {editor && <CourseEditor key={editor.course?.id ?? 'new'} course={editor.course} onClose={() => setEditor(null)} onSave={save} />}
    {deleting && <Dialog title="删除这门课程？" busy={deleteBusy} onClose={() => setDeleting(null)}><p className="dialog-description">将删除「{deleting.name}」及其资料、全部对话与引用快照，停止正在生成的回答，并清理原文件与索引。此操作无法撤销。</p>{deleteError && <p className="error" role="alert">{deleteError}</p>}<div className="dialog-actions"><button className="button secondary" onClick={() => setDeleting(null)} disabled={deleteBusy}>保留课程</button><button className="button destructive" onClick={() => void remove()} disabled={deleteBusy}>{deleteBusy ? '正在删除…' : '确认删除'}</button></div></Dialog>}
  </div>;
}

interface WorkspaceProps { courses: Course[]; failed: boolean; onEdit: (course: Course) => void; onDelete: (course: Course) => void }

function CourseWorkspace({ courses, failed, onEdit, onDelete }: WorkspaceProps) {
  const { courseId, '*': section } = useParams();
  const course = courses.find(item => item.id === courseId);
  if (!course) return <div className="not-found"><h1>{failed ? '课程暂时无法加载' : '课程不存在或已删除'}</h1><Link to="/" className="button secondary">返回我的课程</Link></div>;
  return <>
    <Link className="back-link" to="/"><ArrowLeft size={15} />全部课程</Link>
    <section className="workspace-heading course-heading-compact"><div><span className="workspace-book"><BookOpen size={29} strokeWidth={1.5} /></span><h1>{course.name}</h1><p>{course.description || '在这里整理资料，逐步搭建这门课的知识地图。'}</p><span className="course-date">创建于 {new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium' }).format(new Date(course.created_at))}</span></div><div className="workspace-actions"><button className="button secondary" onClick={() => onEdit(course)}><Pencil size={15} />编辑课程</button><button className="icon-button danger" aria-label="删除课程" onClick={() => onDelete(course)}><Trash2 size={17} /></button></div></section>
    <nav className="workspace-tabs" aria-label="课程工作区"><Link className={section?.startsWith('documents') ? '' : 'active'} to={`/courses/${course.id}`}><MessageSquare size={16} />学习对话</Link><Link className={section?.startsWith('documents') ? 'active' : ''} to={`/courses/${course.id}/documents`}><FileText size={16} />课程资料</Link></nav>
    <Routes key={course.id}><Route index element={<ChatWorkspace courseId={course.id} courseName={course.name} />} /><Route path="conversations/:conversationId" element={<ChatWorkspace courseId={course.id} courseName={course.name} />} /><Route path="documents" element={<div className="course-documents-page"><DocumentLibrary courseId={course.id} /></div>} /><Route path="*" element={<p className="subtle-note">没有找到此课程页面。</p>} /></Routes>
  </>;
}
