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
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["Content-Type"])
app.include_router(router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")


@app.exception_handler(ConversationError)
async def conversation_error(_request: Request, error: ConversationError):
    return JSONResponse(status_code=error.status, content={"detail": error.detail})


@app.exception_handler(DocumentError)
async def document_error(_request: Request, error: DocumentError):
    return JSONResponse(status_code=error.status, content={"detail": error.detail})


@app.exception_handler(CourseNotFound)
async def missing_course(_request: Request, _error: CourseNotFound):
    return JSONResponse(status_code=404, content={"detail": "课程不存在或已被删除，请返回课程列表。"})


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _error: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "请检查输入：名称和查询不能为空，字段长度及课程、资料地址需有效。"})


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "version": app.version}
