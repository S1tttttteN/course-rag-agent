"""管理课程目录及课程删除后的存储清理。"""

from app.db.models import CleanupJob, Course
from app.core.locks import catalog_lock
from app.repositories.courses import CourseRepository


class CourseNotFound(Exception):
    """请求的课程不存在时抛出的业务异常。"""

    pass


class CourseService:
    """管理课程目录并协调删除后的清理。

    Attributes:
        repository: 用于读取和保存课程的仓储。
    """

    def __init__(self, repository: CourseRepository):
        """绑定业务操作使用的课程仓储。

        Args:
            repository: 当前操作使用的课程仓储。
        """
        self.repository = repository

    def list(self):
        """读取课程目录。

        Returns:
            按更新时间降序排列的课程记录。
        """
        return self.repository.list()

    def get(self, course_id: str):
        """读取指定课程。

        Args:
            course_id: 课程 ID。

        Returns:
            匹配的课程记录。

        Raises:
            CourseNotFound: 指定课程不存在。
        """
        course = self.repository.get(course_id)
        if course is None:
            raise CourseNotFound(course_id)
        return course

    def create(self, name: str, description: str):
        """创建并保存一门课程。

        Args:
            name: 新课程名称。
            description: 课程说明。

        Returns:
            已保存的课程对象。
        """
        return self.repository.save(Course(name=name, description=description))

    def update(self, course_id: str, changes: dict):
        """修改指定课程的字段并提交。

        Args:
            course_id: 课程 ID。
            changes: 要更新的课程字段及其新值。

        Returns:
            更新后的课程记录。

        Raises:
            CourseNotFound: 指定课程不存在。
        """
        course = self.get(course_id)
        for field, value in changes.items():
            setattr(course, field, value)
        return self.repository.save(course)

    def delete(self, course_id: str):
        """在同一事务中删除课程并登记存储清理。

        Args:
            course_id: 课程 ID。

        Raises:
            CourseNotFound: 指定课程不存在。
        """
        with catalog_lock:
            course = self.get(course_id)
            self.repository.session.add(CleanupJob(course_id=course_id))
            self.repository.delete(course)
