from app.db.models import CleanupJob, Course
from app.core.locks import catalog_lock
from app.repositories.courses import CourseRepository


class CourseNotFound(Exception):
    pass


class CourseService:
    def __init__(self, repository: CourseRepository):
        self.repository = repository

    def list(self):
        return self.repository.list()

    def get(self, course_id: str):
        course = self.repository.get(course_id)
        if course is None:
            raise CourseNotFound(course_id)
        return course

    def create(self, name: str, description: str):
        return self.repository.save(Course(name=name, description=description))

    def update(self, course_id: str, changes: dict):
        course = self.get(course_id)
        for field, value in changes.items():
            setattr(course, field, value)
        return self.repository.save(course)

    def delete(self, course_id: str):
        with catalog_lock:
            course = self.get(course_id)
            self.repository.session.add(CleanupJob(course_id=course_id))
            self.repository.delete(course)
