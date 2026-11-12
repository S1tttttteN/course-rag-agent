"""组装 FastAPI 应用，统一挂载路由、处理 HTTP 错误并管理后台资源。"""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.courses import router
from app.api.documents import router as documents_router
from app.api.conversations import router as conversations_router
from app.core.config import ALLOWED_ORIGINS, DATA_DIR
from app.db.session import engine
from app.services.courses import CourseNotFound
from app.services.documents import DocumentError
from app.services.conversations import ConversationError
from app.services import runs
from app.qa.manager import RunManager
from app.jobs.indexing import IndexWorker
from app.retrieval import index


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """在接收请求前迁移数据库并恢复运行状态，退出时先停止任务再关闭存储。

    Args:
        _app: 需要初始化后台资源的 FastAPI 应用。

    Yields:
        应用初始化完成后的控制权，退出上下文时释放资源。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
    runs.recover()
    _app.state.runs = RunManager()
    worker = IndexWorker() if getattr(_app.state, "indexing_enabled", True) else None
    if worker:
        worker.start()
    try:
        yield
    finally:
        await _app.state.runs.close()
        if worker:
            worker.stop()
            await asyncio.to_thread(worker.thread.join)
        index.close()
        engine.dispose()


app = FastAPI(title="课程研习室 API", version="0.1.0-alpha.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)
app.include_router(router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")


@app.exception_handler(ConversationError)
async def conversation_error(_request: Request, error: ConversationError):
    """将对话业务异常转换为 JSON 响应。

    Args:
        _request: 触发异常的 HTTP 请求。
        error: 包含 HTTP 状态码和错误说明的对话异常。

    Returns:
        使用业务异常状态码的 JSON 响应。
    """
    return JSONResponse(status_code=error.status, content={"detail": error.detail})


@app.exception_handler(DocumentError)
async def document_error(_request: Request, error: DocumentError):
    """将资料业务异常转换为 JSON 响应。

    Args:
        _request: 触发异常的 HTTP 请求。
        error: 包含 HTTP 状态码和错误说明的资料异常。

    Returns:
        使用业务异常状态码的 JSON 响应。
    """
    return JSONResponse(status_code=error.status, content={"detail": error.detail})


@app.exception_handler(CourseNotFound)
async def missing_course(_request: Request, _error: CourseNotFound):
    """为缺失的课程返回统一错误说明。

    Args:
        _request: 触发异常的 HTTP 请求。
        _error: 课程不存在异常。

    Returns:
        状态码为 404 的 JSON 响应。
    """
    return JSONResponse(
        status_code=404, content={"detail": "课程不存在或已被删除，请返回课程列表。"}
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _error: RequestValidationError):
    """为请求校验失败返回统一输入提示。

    Args:
        _request: 触发异常的 HTTP 请求。
        _error: FastAPI 产生的请求校验异常。

    Returns:
        状态码为 422 的 JSON 响应。
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请检查输入：名称和查询不能为空，字段长度及课程、资料地址需有效。"
        },
    )


@app.get("/api/health", tags=["system"])
def health():
    """报告 API 的健康状态及应用版本。

    Returns:
        包含状态和版本的字典。
    """
    return {"status": "ok", "version": app.version}
