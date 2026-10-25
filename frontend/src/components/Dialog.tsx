import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';

interface Props {
  title: string;
  busy: boolean;
  onClose: () => void;
  children: ReactNode;
}

export function Dialog({ title, busy, onClose, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    dialog.querySelector<HTMLInputElement>('input')?.focus();
    return () => dialog.close();
  }, []);
  return <dialog ref={ref} aria-labelledby="dialog-title" onCancel={event => {
    event.preventDefault();
    if (!busy) onClose();
  }}>
    <div className="dialog-heading">
      <h2 id="dialog-title">{title}</h2>
      <button className="icon-button" aria-label="关闭弹窗" onClick={onClose} disabled={busy}><X size={20} /></button>
    </div>
    {children}
  </dialog>;
}
