from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import MAX_UPLOAD_BYTES
from app.core.models import EMBEDDING_MODEL, EMBEDDING_PROVIDER, EMBEDDING_PROVIDERS, embedding_provider_conf, _get_api_key
from app.db.models import CleanupJob
from app.db.session import get_session
from app.retrieval.search import search
from app.services import documents

router = APIRouter(tags=["documents"])
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/library/config")
def library_config(session: SessionDep):
    conf = EMBEDDING_PROVIDERS.get(EMBEDDING_PROVIDER, {})
    configured, note = True, ""
    try:
        conf = embedding_provider_conf()
        _get_api_key(conf, EMBEDDING_PROVIDER, "嵌入模型")
    except (RuntimeError, ValueError) as error:
        configured, note = False, str(error)
    return {"provider": EMBEDDING_PROVIDER, "model": EMBEDDING_MODEL or conf.get("default_model", ""),
            "configured": configured, "note": note, "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
            "cleanup_pending": session.scalar(select(func.count()).select_from(CleanupJob))}


@router.get("/courses/{course_id}/documents")
def list_documents(course_id: UUID, session: SessionDep):
    return documents.list_documents(session, str(course_id))


@router.post("/courses/{course_id}/documents", status_code=202)
def upload_document(course_id: UUID, file: UploadFile, session: SessionDep):
    try:
        return documents.upload(session, str(course_id), file.filename or "", file.file.read(MAX_UPLOAD_BYTES + 1))
    finally:
        file.file.close()


@router.put("/courses/{course_id}/documents/{document_id}/file", status_code=202)
def replace_document(course_id: UUID, document_id: UUID, file: UploadFile, session: SessionDep):
    try:
        return documents.upload(session, str(course_id), file.filename or "", file.file.read(MAX_UPLOAD_BYTES + 1), str(document_id))
    finally:
        file.file.close()


@router.post("/courses/{course_id}/documents/{document_id}/retry", status_code=202)
def retry_document(course_id: UUID, document_id: UUID, session: SessionDep):
    return documents.retry(session, str(course_id), str(document_id))


@router.delete("/courses/{course_id}/documents/{document_id}", status_code=204)
def delete_document(course_id: UUID, document_id: UUID, session: SessionDep):
    documents.remove(session, str(course_id), str(document_id))
    return Response(status_code=204)


@router.get("/courses/{course_id}/documents/{document_id}/preview")
def preview_document(course_id: UUID, document_id: UUID, session: SessionDep,
                     offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=20)] = 5):
    return documents.preview(session, str(course_id), str(document_id), offset, limit)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    top_k: int = Field(default=4, ge=1, le=20)
    semantic: bool = True
    rerank: bool = False


@router.post("/courses/{course_id}/search")
def search_documents(course_id: UUID, body: SearchInput):
    return search(str(course_id), body.query, body.top_k, body.semantic, body.rerank)
