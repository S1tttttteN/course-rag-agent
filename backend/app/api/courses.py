from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.repositories.courses import CourseRepository
from app.services.courses import CourseService

CourseName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)]


class CourseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: CourseName
    description: Description = ""


class CourseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: CourseName | None = None
    description: Description | None = None

    @model_validator(mode="after")
    def check_changes(self):
        if not self.model_fields_set:
            raise ValueError("请至少提供一个修改项")
        if any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError("修改内容不能为 null")
        return self


class CourseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str


def service(session: Annotated[Session, Depends(get_session)]) -> CourseService:
    return CourseService(CourseRepository(session))


Service = Annotated[CourseService, Depends(service)]
router = APIRouter(prefix="/courses", tags=["courses"])


@router.get("", response_model=list[CourseRead])
def list_courses(svc: Service):
    return svc.list()


@router.post("", response_model=CourseRead, status_code=201)
def create_course(body: CourseCreate, svc: Service):
    return svc.create(body.name, body.description)


@router.get("/{course_id}", response_model=CourseRead)
def get_course(course_id: UUID, svc: Service):
    return svc.get(str(course_id))


@router.patch("/{course_id}", response_model=CourseRead)
def update_course(course_id: UUID, body: CourseUpdate, svc: Service):
    return svc.update(str(course_id), body.model_dump(exclude_unset=True))


@router.delete("/{course_id}", status_code=204)
async def delete_course(course_id: UUID, svc: Service, request: Request):
    request.app.state.runs.cancel_scope(str(course_id))
    svc.delete(str(course_id))
    return Response(status_code=204)
