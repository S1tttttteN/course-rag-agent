import { useEffect, useId, useRef } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';

interface Props {
  title: string;
  busy: boolean;
  onClose: () => void;
  children: ReactNode;
  variant?: 'dialog' | 'sheet' | 'drawer';
}

export function Dialog({ title, busy, onClose, children, variant = 'dialog' }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const dialog = ref.current!;
    const opener = document.activeElement;
    dialog.showModal();
    if (variant === 'dialog') dialog.querySelector<HTMLInputElement>('input')?.focus();
    return () => {
      dialog.close();
      if (document.activeElement === document.body && opener instanceof HTMLElement && opener.isConnected) {
        opener.focus({ preventScroll: true });
      }
    };
  }, [variant]);
  return <dialog ref={ref} className={`overlay overlay-${variant}`} aria-labelledby={titleId} onCancel={event => {
    event.preventDefault();
    if (!busy) onClose();
  }}>
    <div className="dialog-heading">
      <h2 id={titleId}>{title}</h2>
      <button type="button" className="icon-button" aria-label={`关闭${title}`} onClick={onClose} disabled={busy}><X size={18} /></button>
    </div>
    {children}
  </dialog>;
}
