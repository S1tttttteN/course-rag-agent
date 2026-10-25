import { request } from './client';
import type { CourseDocument, LibraryConfig, Preview, SearchResult } from '../types/document';

export const documentsApi = {
  config: () => request<LibraryConfig>('/library/config'),
  list: (courseId: string) => request<CourseDocument[]>(`/courses/${courseId}/documents`),
  upload: (courseId: string, file: File, documentId?: string) => {
    const body = new FormData();
    body.append('file', file);
    return request<CourseDocument>(`/courses/${courseId}/documents${documentId ? `/${documentId}/file` : ''}`, {method: documentId ? 'PUT' : 'POST', body});
  },
  retry: (courseId: string, id: string) => request<CourseDocument>(`/courses/${courseId}/documents/${id}/retry`, {method: 'POST'}),
  delete: (courseId: string, id: string) => request<void>(`/courses/${courseId}/documents/${id}`, {method: 'DELETE'}),
  preview: (courseId: string, id: string, offset = 0) => request<Preview>(`/courses/${courseId}/documents/${id}/preview?offset=${offset}&limit=5`),
  search: (courseId: string, query: string, semantic: boolean, rerank: boolean) => request<SearchResult>(`/courses/${courseId}/search`, {method: 'POST', body: JSON.stringify({query, semantic, rerank})}),
};
