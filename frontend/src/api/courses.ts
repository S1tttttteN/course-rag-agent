import { request } from './client';
import type { Course, CourseInput } from '../types/course';

export const coursesApi = {
  list: () => request<Course[]>('/courses'),
  create: (input: CourseInput) => request<Course>('/courses', { method: 'POST', body: JSON.stringify(input) }),
  update: (id: string, input: CourseInput) => request<Course>(`/courses/${id}`, { method: 'PATCH', body: JSON.stringify(input) }),
  delete: (id: string) => request<void>(`/courses/${id}`, { method: 'DELETE' }),
};
