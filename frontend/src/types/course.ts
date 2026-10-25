export interface Course {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface CourseInput {
  name: string;
  description: string;
}
