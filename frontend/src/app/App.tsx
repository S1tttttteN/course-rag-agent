import { useCallback, useEffect, useState } from 'react';
import { Link, NavLink, Route, Routes, useLocation, useMatch, useNavigate } from 'react-router-dom';
import { BookOpen, Check, ChevronRight, FileText, LoaderCircle, Menu, MessageSquare, Pencil, Plus, Search, Trash2 } from 'lucide-react';
import { conversationsApi } from '../api/conversations';
import { coursesApi } from '../api/courses';
import { Dialog } from '../components/Dialog';
import { compactNavigationQuery, useMediaQuery } from '../components/useMediaQuery';
import { ChatWorkspace } from '../features/conversations/ChatWorkspace';
import { readRecentConversation } from '../features/conversations/recentConversation';
import { CourseEditor } from '../features/courses/CourseEditor';
import { DocumentLibrary } from '../features/documents/DocumentLibrary';
import type { Conversation } from '../types/conversation';
import type { Course, CourseInput } from '../types/course';

function CoursesHome({ courses, onCreate, onEdit, onDelete }: {
  courses: Course[];
  onCreate: () => void;
  onEdit: (course: Course) => void;
  onDelete: (course: Course) => void;
}) {
  const [search, setSearch] = useState('');
  const [recent] = useState(readRecentConversation);
  const [resume, setResume] = useState<Conversation | null>(null);
  const recentCourse = courses.find(course => course.id === recent?.courseId);
  const recentCourseId = recentCourse?.id;
  useEffect(() => {
    if (!recent || !recentCourseId) return;
    let ignore = false;
    void conversationsApi.list(recentCourseId).then(items => {
      if (!ignore) setResume(items.find(item => item.id === recent.conversationId) ?? null);
    }).catch(() => {
      if (!ignore) setResume(null);
    });
    return () => { ignore = true; };
  }, [recent, recentCourseId]);

  const query = search.trim().toLocaleLowerCase();
  const filtered = courses.filter(course => (course.name + ' ' + course.description).toLocaleLowerCase().includes(query));
  return <section className="courses-home" aria-labelledby="courses-title">
    <header className="page-heading">
      <div><h1 id="courses-title">我的课程</h1><p>按课程整理资料，接着之前的问题学下去。</p></div>
      <button type="button" className="button primary" onClick={onCreate}><Plus size={16} />新建课程</button>
    </header>
    {recentCourse && resume && <Link className="resume-link" to={'/courses/' + recentCourse.id + '/conversations/' + resume.id}>
      <MessageSquare size={19} />
      <span><small>继续上次研习</small><strong>{resume.title}</strong><span>{recentCourse.name}</span></span>
      <ChevronRight size={18} />
    </Link>}
    {courses.length > 0 ? <>
      <div className="course-list-heading">
        <h2>全部课程 <span>{courses.length}</span></h2>
        <label className="search"><Search size={16} /><input aria-label="搜索课程" placeholder="搜索课程" value={search} onChange={event => setSearch(event.target.value)} /></label>
      </div>
      <ul className="course-list">
        {filtered.map(course => <li key={course.id} className="course-row">
          <Link to={'/courses/' + course.id}>
            <BookOpen size={18} />
            <span><strong>{course.name}</strong>{course.description && <small>{course.description}</small>}</span>
          </Link>
          <button type="button" className="icon-button" aria-label={'编辑' + course.name} onClick={() => onEdit(course)}><Pencil size={15} /></button>
          <button type="button" className="icon-button danger" aria-label={'删除' + course.name} onClick={() => onDelete(course)}><Trash2 size={15} /></button>
        </li>)}
      </ul>
      {!filtered.length && <p className="empty-state">没有找到匹配的课程。<button type="button" className="text-button" onClick={() => setSearch('')}>清空搜索</button></p>}
    </> : <div className="empty-state"><BookOpen size={24} /><h2>还没有课程</h2><p>先新建一门课程，再上传笔记或讲义。</p></div>}
  </section>;
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
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [conversationRail, setConversationRail] = useState<HTMLDivElement | null>(null);
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const courseId = useMatch('/courses/:courseId/*')?.params.courseId;
  const course = courses.find(item => item.id === courseId);
  const compact = useMediaQuery(compactNavigationQuery);
  const isLibrary = pathname === '/courses/' + courseId + '/documents';

  const reload = useCallback(async () => {
    setLoading(true);
    setError('');
    try { setCourses(await coursesApi.list()); }
    catch (error) { setError(error instanceof Error ? error.message : '课程加载失败。'); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => { setNavigationOpen(false); }, [pathname, compact]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 4000);
    return () => clearTimeout(timer);
  }, [notice]);

  function edit(course: Course | null) { setEditor({ course }); }
  function confirmDelete(course: Course) { setDeleting(course); setDeleteError(''); }

  async function save(input: CourseInput) {
    const editing = editor?.course;
    const saved = editing ? await coursesApi.update(editing.id, input) : await coursesApi.create(input);
    setCourses(previous => [saved, ...previous.filter(course => course.id !== saved.id)]);
    setEditor(null);
    setNotice(editing ? '课程已更新' : '课程已创建');
    if (!editing) navigate('/courses/' + saved.id);
  }
  async function remove() {
    if (!deleting || deleteBusy) return;
    setDeleteBusy(true);
    setDeleteError('');
    try {
      await coursesApi.delete(deleting.id);
      setCourses(previous => previous.filter(course => course.id !== deleting.id));
      setDeleting(null);
      if (courseId === deleting.id) navigate('/');
      setNotice('课程已删除');
    } catch (error) { setDeleteError(error instanceof Error ? error.message : '删除失败，请重试。'); }
    finally { setDeleteBusy(false); }
  }

  const navigation = <>
    <NavLink to="/" end className="nav-row" onClick={() => setNavigationOpen(false)}><BookOpen size={17} />我的课程</NavLink>
    <div className="rail-heading">
      <h2>课程目录</h2>
      <button type="button" className="icon-button" aria-label="新建课程" onClick={() => edit(null)}><Plus size={17} /></button>
    </div>
    <nav className="course-nav" aria-label="课程目录">
      {courses.map(item => <NavLink key={item.id} to={'/courses/' + item.id} className="nav-row" onClick={() => setNavigationOpen(false)}>
        <BookOpen size={16} /><span>{item.name}</span>
      </NavLink>)}
      {!loading && !error && !courses.length && <p className="subtle-note">还没有课程。</p>}
    </nav>
    {course && <div ref={setConversationRail} />}
  </>;

  const chat = course && <ChatWorkspace key={course.id} courseId={course.id} courseName={course.name}
    navigationTarget={conversationRail} onNavigate={() => setNavigationOpen(false)}
    library={isLibrary ? <DocumentLibrary key={course.id} courseId={course.id} /> : undefined} />;
  const missingCourse = <div className="empty-state"><h1>课程不存在或已删除</h1><Link to="/" className="button secondary">返回我的课程</Link></div>;

  return <div className="app-shell">
    <a href="#main" className="skip-link">跳到主要内容</a>
    {compact ? navigationOpen && <Dialog title="课程与对话" variant="drawer" busy={false} onClose={() => setNavigationOpen(false)}>
      <div className="drawer-navigation">{navigation}</div>
    </Dialog> : <aside className="sidebar" aria-label="课程与对话">
      <Link to="/" className="brand"><span aria-hidden="true">[n]</span>课程研习室</Link>
      {navigation}
    </aside>}

    <div className="main-shell">
      <header className="topbar">
        <div className="breadcrumbs">
          {compact && <button type="button" className="icon-button" aria-label="打开课程与对话" aria-expanded={navigationOpen} onClick={() => setNavigationOpen(true)}><Menu size={20} /></button>}
          <Link to="/">{course ? '我的课程' : '课程研习室'}</Link>
          {course && <><ChevronRight size={14} /><strong>{course.name}</strong></>}
        </div>
        {course && <nav className="workspace-tabs" aria-label="课程工作区">
          <Link to={'/courses/' + course.id} aria-current={isLibrary ? undefined : 'page'}><MessageSquare size={16} /><span>学习对话</span></Link>
          <Link to={'/courses/' + course.id + '/documents'} aria-current={isLibrary ? 'page' : undefined}><FileText size={16} /><span>课程资料</span></Link>
        </nav>}
      </header>
      <main id="main" tabIndex={-1}>
        {loading ? <p className="operation-note page-status" role="status"><LoaderCircle size={19} className="spin" />正在打开学习空间…</p>
          : error ? <div className="error-banner page-status" role="alert"><p>{error}</p><button type="button" className="button secondary" onClick={() => void reload()}>重新加载</button></div>
            : <Routes>
              <Route path="/" element={<CoursesHome courses={courses} onCreate={() => edit(null)} onEdit={edit} onDelete={confirmDelete} />} />
              <Route path="/courses/:courseId" element={chat ?? missingCourse} />
              <Route path="/courses/:courseId/conversations/:conversationId" element={chat ?? missingCourse} />
              <Route path="/courses/:courseId/documents" element={chat ?? missingCourse} />
              <Route path="*" element={<div className="empty-state"><h1>这里还没有内容</h1><Link to="/" className="button secondary">返回我的课程</Link></div>} />
            </Routes>}
      </main>
    </div>
    {notice && <div className="toast" role="status"><Check size={16} />{notice}</div>}
    {editor && <CourseEditor key={editor.course?.id ?? 'new'} course={editor.course} onClose={() => setEditor(null)} onSave={save} />}
    {deleting && <Dialog title="删除这门课程？" busy={deleteBusy} onClose={() => setDeleting(null)}>
      <p className="dialog-description">「{deleting.name}」的资料、对话和引用将一并删除，正在生成的回答也会停止。此操作无法撤销。</p>
      {deleteError && <p className="error" role="alert">{deleteError}</p>}
      <div className="dialog-actions">
        <button type="button" className="button secondary" disabled={deleteBusy} onClick={() => setDeleting(null)}>保留课程</button>
        <button type="button" className="button destructive" disabled={deleteBusy} onClick={() => void remove()}>{deleteBusy ? '正在删除…' : '确认删除课程'}</button>
      </div>
    </Dialog>}
  </div>;
}
