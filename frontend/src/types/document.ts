export interface DocumentVersion {
  id: string;
  number: number;
  filename: string;
  size: number;
  status: 'queued' | 'processing' | 'ready' | 'failed';
  step: string;
  error: string;
  chunk_count: number;
  embedding_profile: string;
  created_at: string;
}

export interface CourseDocument {
  id: string;
  name: string;
  course_id: string;
  active_version_id: string | null;
  active_version_number: number | null;
  chunk_count: number;
  usable: boolean;
  latest: DocumentVersion;
}

export interface LibraryConfig {
  provider: string;
  model: string;
  configured: boolean;
  note: string;
  max_upload_mb: number;
  cleanup_pending: number;
}

export interface Preview {
  source: string;
  version: number;
  total: number;
  offset: number;
  chunks: {id: string; text: string; page: number | null; ordinal: number}[];
}

export interface SearchResult {
  results: { id: string; document_id: string; version_id: string; version: number; source: string; text: string; page: number | null; rerank_score?: number }[];
  trace: {semantic: boolean; keyword: boolean; reranked: boolean; candidates: number; note: string};
}
