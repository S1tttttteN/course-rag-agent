"""封装课程 ORM 记录的查询与持久化。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Course


class CourseRepository:
    """使用请求级数据库会话读写课程记录。

    Attributes:
        session: 保存课程变更所使用的 SQLAlchemy 会话。
    """

    def __init__(self, session: Session):
        """绑定课程仓储使用的数据库会话。

        Args:
            session: 用于当前操作的 SQLAlchemy 会话。
        """
        self.session = session

    def list(self) -> list[Course]:
        """查询所有课程。

        Returns:
            按更新时间降序排列的课程列表，同一时间按 ID 排序。
        """
        return list(
            self.session.scalars(
                select(Course).order_by(Course.updated_at.desc(), Course.id)
            )
        )

    def get(self, course_id: str) -> Course | None:
        """按 ID 查找课程。

        Args:
            course_id: 课程 ID。

        Returns:
            匹配的课程记录，或不存在时的 None。
        """
        return self.session.get(Course, course_id)

    def save(self, course: Course) -> Course:
        """提交并刷新课程记录。

        Args:
            course: 待创建或更新的课程对象。

        Returns:
            从数据库刷新的课程对象。
        """
        self.session.add(course)
        self.session.commit()
        self.session.refresh(course)
        return course

    def delete(self, course: Course) -> None:
        """删除课程并提交当前事务。

        Args:
            course: 要删除的课程对象。
        """
        self.session.delete(course)
        self.session.commit()
