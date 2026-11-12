"""提供课程的创建、查询、更新和删除接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.repositories.courses import CourseRepository
from app.services.courses import CourseService

CourseName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)]


class CourseCreate(BaseModel):
    """创建课程的请求体。

    Attributes:
        name: 课程名称，去除首尾空白后须为 1 至 80 个字符。
        description: 可选课程说明，最多 1000 个字符，默认为空字符串。
    """

    model_config = ConfigDict(extra="forbid")
    name: CourseName
    description: Description = ""


class CourseUpdate(BaseModel):
    """更新课程部分字段的请求体。

    Attributes:
        name: 要替换的课程名称，省略时保留原值。
        description: 要替换的课程说明，省略时保留原值。
    """

    model_config = ConfigDict(extra="forbid")
    name: CourseName | None = None
    description: Description | None = None

    @model_validator(mode="after")
    def check_changes(self):
        """验证至少提交了一个非空修改项。

        Returns:
            校验通过的请求体。

        Raises:
            ValueError: 未提供修改项，或显式传入了 null。
        """
        if not self.model_fields_set:
            raise ValueError("请至少提供一个修改项")
        if any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError("修改内容不能为 null")
        return self


class CourseRead(BaseModel):
    """课程接口返回的数据模型。

    Attributes:
        id: 课程 ID。
        name: 课程名称。
        description: 课程说明。
        created_at: 课程创建时间。
        updated_at: 课程最近更新时间。
    """

    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str


def service(session: Annotated[Session, Depends(get_session)]) -> CourseService:
    """为当前请求构造课程服务。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        使用当前数据库会话的课程服务。
    """
    return CourseService(CourseRepository(session))


Service = Annotated[CourseService, Depends(service)]
router = APIRouter(prefix="/courses", tags=["courses"])


@router.get("", response_model=list[CourseRead])
def list_courses(svc: Service):
    """列出所有课程。

    Args:
        svc: 注入的课程服务。

    Returns:
        按更新时间排序的课程记录。
    """
    return svc.list()


@router.post("", response_model=CourseRead, status_code=201)
def create_course(body: CourseCreate, svc: Service):
    """创建课程并返回记录。

    Args:
        body: 包含课程名称和说明的请求体。
        svc: 注入的课程服务。

    Returns:
        已保存的新课程记录。
    """
    return svc.create(body.name, body.description)


@router.get("/{course_id}", response_model=CourseRead)
def get_course(course_id: UUID, svc: Service):
    """读取指定课程。

    Args:
        course_id: 课程 ID。
        svc: 注入的课程服务。

    Returns:
        匹配的课程记录。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    return svc.get(str(course_id))


@router.patch("/{course_id}", response_model=CourseRead)
def update_course(course_id: UUID, body: CourseUpdate, svc: Service):
    """更新请求中明确提供的课程字段。

    Args:
        course_id: 课程 ID。
        body: 包含待修改字段的请求体。
        svc: 注入的课程服务。

    Returns:
        更新后的课程记录。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    return svc.update(str(course_id), body.model_dump(exclude_unset=True))


@router.delete("/{course_id}", status_code=204)
async def delete_course(course_id: UUID, svc: Service, request: Request):
    """请求取消课程的活动运行，然后删除课程。

    Args:
        course_id: 课程 ID。
        svc: 注入的课程服务。
        request: 当前 HTTP 请求，可用于访问应用状态。

    Returns:
        状态码为 204 的空响应。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    request.app.state.runs.cancel_scope(str(course_id))
    svc.delete(str(course_id))
    return Response(status_code=204)
