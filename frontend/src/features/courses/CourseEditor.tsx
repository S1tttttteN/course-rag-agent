import { useState } from 'react';
import type { FormEvent } from 'react';
import { Dialog } from '../../components/Dialog';
import type { Course, CourseInput } from '../../types/course';

interface Props {
  course: Course | null;
  onClose: () => void;
  onSave: (input: CourseInput) => Promise<void>;
}

export function CourseEditor({ course, onClose, onSave }: Props) {
  const [name, setName] = useState(course?.name ?? '');
  const [description, setDescription] = useState(course?.description ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy || !name.trim()) return;
    setBusy(true);
    setError('');
    try { await onSave({ name: name.trim(), description: description.trim() }); }
    catch (error) { setError(error instanceof Error ? error.message : '保存失败，请重试。'); setBusy(false); }
  }
  return <Dialog title={course ? '编辑课程' : '创建一门课程'} busy={busy} onClose={onClose}>
    <p className="dialog-description">给学习内容一个归属。每门课程独立管理自己的资料和对话。</p>
    <form onSubmit={submit}>
      <label htmlFor="course-name">课程名称 <span className="required">必填</span></label>
      <input id="course-name" autoFocus maxLength={80} required value={name} disabled={busy}
        placeholder="例如：操作系统" onChange={e => setName(e.target.value)} />
      <label htmlFor="course-description">课程简介 <span className="optional">选填</span></label>
      <textarea id="course-description" rows={3} maxLength={1000} value={description} disabled={busy}
        placeholder="这门课，你想学懂什么？" onChange={e => setDescription(e.target.value)} />
      {error && <p className="error" role="alert">{error}</p>}
      <div className="dialog-actions">
        <button type="button" className="button secondary" onClick={onClose} disabled={busy}>取消</button>
        <button type="submit" className="button primary" disabled={busy || !name.trim()}>{busy ? '正在保存…' : course ? '保存修改' : '创建课程'}</button>
      </div>
    </form>
  </Dialog>;
}
