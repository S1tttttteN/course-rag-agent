"""课程资料的 HTTP 入口，接收上传和替换请求，并提供预览与检索接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import MAX_UPLOAD_BYTES
from app.core.models import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDERS,
    embedding_provider_conf,
    _get_api_key,
)
from app.db.models import CleanupJob
from app.db.session import get_session
from app.services import documents

router = APIRouter(tags=["documents"])
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/library/config")
def library_config(session: SessionDep):
    """读取资料库配置和清理积压情况。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        包含嵌入模型配置、上传大小上限和待清理任务数的字典。
    """
    conf = EMBEDDING_PROVIDERS.get(EMBEDDING_PROVIDER, {})
    configured, note = True, ""
    try:
        conf = embedding_provider_conf()
        _get_api_key(conf, EMBEDDING_PROVIDER, "嵌入模型")
    except (RuntimeError, ValueError) as error:
        configured, note = False, str(error)
    return {
        "provider": EMBEDDING_PROVIDER,
        "model": EMBEDDING_MODEL or conf.get("default_model", ""),
        "configured": configured,
        "note": note,
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
        "cleanup_pending": session.scalar(select(func.count()).select_from(CleanupJob)),
    }


@router.get("/courses/{course_id}/documents")
def list_documents(course_id: UUID, session: SessionDep):
    """列出指定课程的资料。

    Args:
        course_id: 课程 ID。
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        包含各资料最新版本及可用版本的字典列表。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    return documents.list_documents(session, str(course_id))


@router.post("/courses/{course_id}/documents", status_code=202)
def upload_document(course_id: UUID, file: UploadFile, session: SessionDep):
    """接收文件、登记待索引资料，并关闭上传流。

    Args:
        course_id: 课程 ID。
        file: 客户端上传的文件。
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        新资料及版本信息。

    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 文件类型、大小或名称不符合要求，或同名资料已存在。
    """
    try:
        return documents.upload(
            session,
            str(course_id),
            file.filename or "",
            file.file.read(MAX_UPLOAD_BYTES + 1),
        )
    finally:
        file.file.close()


@router.put("/courses/{course_id}/documents/{document_id}/file", status_code=202)
def replace_document(
    course_id: UUID, document_id: UUID, file: UploadFile, session: SessionDep
):
    """上传新文件作为现有资料的待索引版本。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        file: 客户端上传的文件。
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        包含新版本和当前可用版本的资料信息。

    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 资料归属、文件类型、文件大小或当前版本状态不允许替换。
    """
    try:
        return documents.upload(
            session,
            str(course_id),
            file.filename or "",
            file.file.read(MAX_UPLOAD_BYTES + 1),
            str(document_id),
        )
    finally:
        file.file.close()


@router.post("/courses/{course_id}/documents/{document_id}/retry", status_code=202)
def retry_document(course_id: UUID, document_id: UUID, session: SessionDep):
    """将失败的资料版本重新加入索引队列。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        更新后的资料及版本信息。

    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 资料不存在，或最新版本无法重试。
    """
    return documents.retry(session, str(course_id), str(document_id))


@router.delete("/courses/{course_id}/documents/{document_id}", status_code=204)
def delete_document(course_id: UUID, document_id: UUID, session: SessionDep):
    """删除资料并登记后台清理任务。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        session: 用于当前操作的 SQLAlchemy 会话。

    Returns:
        状态码为 204 的空响应。

    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 资料不存在或不属于指定课程。
    """
    documents.remove(session, str(course_id), str(document_id))
    return Response(status_code=204)


@router.get(
    "/courses/{course_id}/documents/{document_id}/versions/{version_id}/pdf",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}},
)
def read_pdf(course_id: UUID, document_id: UUID, version_id: UUID, session: SessionDep):
    """提供指定版本的 PDF 原件。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        version_id: 要读取的版本 ID。
        session: 用于当前操作的 SQLAlchemy 会话。
    Returns:
        支持字节范围请求的 PDF 文件响应，供阅读器内联显示。
    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 资料或版本不存在、归属不符、版本非 PDF，或原文件缺失。
    """
    path, filename = documents.pdf_original(
        session, str(course_id), str(document_id), str(version_id)
    )
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/courses/{course_id}/documents/{document_id}/versions/{version_id}/docx",
    response_class=FileResponse,
)
def read_docx(course_id: UUID, document_id: UUID, version_id: UUID, session: SessionDep):
    """为站内阅读器提供指定版本的 Word 原件。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        version_id: 要读取的版本 ID。
        session: 用于当前操作的 SQLAlchemy 会话。
    Returns:
        供浏览器阅读器加载的 DOCX 文件响应。
    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 归属不符、格式不支持、原件缺失或压缩包无效。
    """
    path = documents.docx_original(session, str(course_id), str(document_id), str(version_id))
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Disposition": "attachment"},
    )


@router.get("/courses/{course_id}/documents/{document_id}/versions/{version_id}/text")
def read_text(
    course_id: UUID, document_id: UUID, version_id: UUID, session: SessionDep, response: Response
) -> dict[str, str]:
    """读取指定版本的 TXT 或 Markdown 全文。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        version_id: 要读取的版本 ID。
        session: 用于当前操作的 SQLAlchemy 会话。
        response: 用于设置缓存响应头的响应对象。
    Returns:
        包含解码后全文的字典。
    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 归属不符、原件缺失、格式不支持、解码失败或正文超长。
    """
    text = documents.text_original(session, str(course_id), str(document_id), str(version_id))
    response.headers["Cache-Control"] = "no-store"
    return {"text": text}


@router.get("/courses/{course_id}/documents/{document_id}/versions/{version_id}/rich-preview")
def read_rich_preview(
    course_id: UUID, document_id: UUID, version_id: UUID, session: SessionDep, response: Response
) -> dict[str, str]:
    """读取指定版本的 HTML 或 DOCX 静态预览。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        version_id: 要读取的版本 ID。
        session: 用于当前操作的 SQLAlchemy 会话。
        response: 用于设置缓存响应头的响应对象。
    Returns:
        包含清理后 HTML 正文和 CSS 文本的字典。
    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 资料归属不符、原件缺失、格式不支持或无法转换。
    """
    preview = documents.rich_original(session, str(course_id), str(document_id), str(version_id))
    response.headers["Cache-Control"] = "no-store"
    return preview


@router.get("/courses/{course_id}/documents/{document_id}/preview")
def preview_document(
    course_id: UUID,
    document_id: UUID,
    session: SessionDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
):
    """分页读取资料可用版本的原文片段。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        session: 用于当前操作的 SQLAlchemy 会话。
        offset: 从零开始的片段偏移量。
        limit: 每页片段数，取值为 1 至 20。

    Returns:
        包含预览片段及分页信息的字典。

    Raises:
        CourseNotFound: 指定课程不存在。
        documents.DocumentError: 资料不存在或没有可预览的版本。
    """
    return documents.preview(session, str(course_id), str(document_id), offset, limit)
