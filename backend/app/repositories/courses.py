from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Course


class CourseRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self) -> list[Course]:
        return list(self.session.scalars(select(Course).order_by(Course.updated_at.desc(), Course.id)))

    def get(self, course_id: str) -> Course | None:
        return self.session.get(Course, course_id)

    def save(self, course: Course) -> Course:
        self.session.add(course)
        self.session.commit()
        self.session.refresh(course)
        return course

    def delete(self, course: Course) -> None:
        self.session.delete(course)
        self.session.commit()
