import { useSyncExternalStore } from 'react';

export const compactNavigationQuery = '(max-width: 800px)';
export const compactReadingQuery = '(max-width: 1179px)';

export function useMediaQuery(query: string) {
  return useSyncExternalStore(
    listener => {
      const media = window.matchMedia(query);
      media.addEventListener('change', listener);
      return () => media.removeEventListener('change', listener);
    },
    () => window.matchMedia(query).matches,
  );
}
