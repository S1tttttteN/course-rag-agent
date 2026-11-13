import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { Dialog } from './Dialog';
import { compactReadingQuery, useMediaQuery } from './useMediaQuery';

interface Props {
  title: string;
  children: ReactNode;
  onClose: () => void;
}

export function ReadingPanel({ title, children, onClose }: Props) {
  const compact = useMediaQuery(compactReadingQuery);
  if (compact) {
    return <Dialog title={title} variant="sheet" busy={false} onClose={onClose}>
      <div className="reading-panel-body">{children}</div>
    </Dialog>;
  }
  return <aside className="reading-panel" aria-label={title}>
    <header className="reading-panel-heading">
      <h2>{title}</h2>
      <button type="button" className="icon-button" aria-label={`关闭${title}`} onClick={onClose}>
        <X size={18} />
      </button>
    </header>
    <div className="reading-panel-body">{children}</div>
  </aside>;
}
