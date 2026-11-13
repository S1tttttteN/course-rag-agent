const storageKey = 'course-study-room.recent-conversation';

interface RecentConversation {
  courseId: string;
  conversationId: string;
}

export function readRecentConversation(): RecentConversation | null {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) ?? 'null');
    return typeof value?.courseId === 'string' && typeof value?.conversationId === 'string'
      ? { courseId: value.courseId, conversationId: value.conversationId }
      : null;
  } catch {
    return null;
  }
}

export function rememberConversation(courseId: string, conversationId: string) {
  try {
    localStorage.setItem(storageKey, JSON.stringify({ courseId, conversationId }));
  } catch {
    // Resuming a conversation is optional when browser storage is unavailable.
  }
}
