export type RunStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'cancelled' | 'failed' | 'interrupted';
export interface Conversation { id: string; course_id: string; title: string; created_at: string; updated_at: string }
export interface Citation {
  id: string; number: number; document_id: string; version_id: string; chunk_id: string;
  source: string; version: number; text: string; page: number | null; start_index: number;
  kind: 'course' | 'web_snippet' | 'web_page'; url: string; retrieved_at: string;
  source_state: 'current' | 'updated' | 'deleted' | 'external';
}
export interface Run {
  id: string; conversation_id: string; question: string; web_search: boolean; status: RunStatus;
  phase: string; error: string; created_at: string; last_event_id: number;
  trace: { unknown_citations?: number[]; tools?: ToolExecution[]; tool_limit?: number };
}
export interface ToolExecution {
  id: string; number: number; name: string; label: string; input: Record<string, string>;
  status: 'running' | 'completed' | 'failed' | 'interrupted' | 'limited'; summary: string; duration_ms: number | null;
  retrieval?: { keyword: boolean; semantic: boolean; reranked: boolean; note: string };
}
export interface ChatMessage {
  id: string; run_id: string; role: 'user' | 'assistant'; content: string;
  status: RunStatus | 'pending'; created_at: string; citations: Citation[]; run: Run | null;
}
export interface ConversationDetail extends Conversation { messages: ChatMessage[]; active_run: Run | null }
export interface ChatConfig { provider: string; model: string; configured: boolean; note: string }
export interface QuestionInput { question: string; web_search?: boolean }
