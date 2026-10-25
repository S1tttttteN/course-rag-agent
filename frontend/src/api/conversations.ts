import { request } from './client';
import type { ChatConfig, Citation, Conversation, ConversationDetail, Run } from '../types/conversation';

const base = (course: string, conversation?: string) => `/courses/${course}/conversations${conversation ? `/${conversation}` : ''}`;
export const conversationsApi = {
  config: () => request<ChatConfig>('/chat/config'),
  list: (course: string) => request<Conversation[]>(base(course)),
  create: (course: string) => request<Conversation>(base(course), {method: 'POST', body: JSON.stringify({})}),
  detail: (course: string, conversation: string) => request<ConversationDetail>(base(course, conversation)),
  rename: (course: string, conversation: string, title: string) => request<Conversation>(base(course, conversation), {method: 'PATCH', body: JSON.stringify({title})}),
  delete: (course: string, conversation: string) => request<void>(base(course, conversation), {method: 'DELETE'}),
  ask: (course: string, conversation: string, question: string, top_k: number, semantic: boolean, rerank: boolean, mode: 'rag' | 'agent') => request<Run>(`${base(course, conversation)}/runs`, {method: 'POST', body: JSON.stringify({question, top_k, semantic, rerank, mode})}),
  cancel: (course: string, conversation: string, run: string) => request<Run>(`${base(course, conversation)}/runs/${run}/cancel`, {method: 'POST'}),
  citation: (course: string, conversation: string, id: string) => request<Citation>(`${base(course, conversation)}/citations/${id}`),
  events: (course: string, conversation: string, run: string, after: number) => new EventSource(`/api${base(course, conversation)}/runs/${run}/events?after=${after}`),
};
