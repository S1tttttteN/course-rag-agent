import asyncio
from contextlib import aclosing
import logging
from traceback import extract_tb, format_list

from sqlalchemy import select

from app.core.locks import catalog_lock
from app.core.models import get_llm, ANSWER_REVIEW, RUN_TIMEOUT
from app.db.models import Conversation, Document, Run
from app.db.session import SessionLocal
from app.qa.rag import answer_messages, rewrite_query, text_content
from app.qa.metrics import RunUsage
from app.qa.evidence import EvidenceCheckError, publish_checked
from app.repositories.conversations import completed_turns
from app.retrieval.search import search
from app.services import runs
from app.services.conversations import ACTIVE, event

log = logging.getLogger(__name__)


class AnswerLimit(Exception):
    pass


class RunManager:
    def __init__(self):
        self.tasks = {}
        self.offloads = set()
        from app.core.models import CHAT_PROVIDER
        self.semaphore = asyncio.Semaphore(1 if CHAT_PROVIDER == 'ollama' else 2)
        self.closing = False

    def start(self, run_id):
        task = asyncio.create_task(self.execute(run_id))
        self.tasks[run_id] = task
        def ended(done):
            self.tasks.pop(run_id, None)
            if done.cancelled():
                runs.finish(run_id, "interrupted" if self.closing else "cancelled", "服务停止，回答已中断。请重新发送问题。" if self.closing else "")
            elif done.exception():
                log.error("Run task failed outside handler: %s", type(done.exception()).__name__)
                runs.finish(run_id, "failed", "回答未完成，请重试。")
        task.add_done_callback(ended)

    async def offload(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self.offloads.add(task)
        def ended(done):
            self.offloads.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieve exceptions even if the caller was cancelled.
        task.add_done_callback(ended)
        return await asyncio.shield(task)

    async def retrieve(self, *args):
        return await self.offload(search, *args)

    async def execute(self, run_id):
        buffer = ""
        try:
            async with self.semaphore, asyncio.timeout(RUN_TIMEOUT):
                with SessionLocal() as session:
                    run = session.get(Run, run_id)
                    if run is None:
                        return
                    conversation = session.get(Conversation, run.conversation_id)
                    course_id, question = conversation.course_id, run.question
                    top_k, semantic, rerank = run.top_k, bool(run.semantic), bool(run.rerank)
                    mode = run.mode
                    history = completed_turns(session, conversation.id, run.id)
                    has_sources = session.scalar(select(Document.id).where(Document.course_id == course_id, Document.active_version_id.is_not(None)).limit(1)) is not None
                if mode == "agent":
                    from app.qa.agent import execute_agent
                    await execute_agent(self, run_id, course_id, question, history, top_k, semantic, rerank)
                    runs.finish(run_id, "completed")
                    return
                query, rewrite_note = question, ""
                usage = RunUsage(run_id)
                llm = None
                if history and has_sources:
                    if not runs.status(run_id, "正在理解追问"):
                        return
                    llm = get_llm()
                    query, rewrite_note = await rewrite_query(llm, question, history, [usage])
                if not runs.status(run_id, "正在检索课程资料", query=query):
                    return
                result = await self.retrieve(course_id, query, top_k, semantic, rerank)
                trace = {**result["trace"], "history_turns": len(history), "rewrite_note": rewrite_note, "mode": "rag", "answer_review": ANSWER_REVIEW}
                if not runs.status(run_id, "正在核对参考片段", trace=trace):
                    return
                sources = runs.snapshot(run_id, result["results"])
                if not sources and ANSWER_REVIEW:
                    runs.append(run_id, "当前课程资料中未找到足够依据。请补充相关资料，或换用资料中的关键词重新提问。")
                    runs.finish(run_id, "completed")
                    return
                if not runs.status(run_id, "正在生成回答"):
                    return
                llm = llm or get_llm()
                answer_length = 0
                async with aclosing(llm.astream(answer_messages(question, history, sources), config={"callbacks": [usage]})) as stream:
                    async for chunk in stream:
                        text = text_content(chunk)
                        buffer += text
                        answer_length += len(text)
                        if answer_length > 20000 or getattr(chunk, "response_metadata", {}).get("finish_reason") in {"length", "max_tokens"} or getattr(chunk, 'response_metadata', {}).get('done_reason') == 'length':
                            raise AnswerLimit()
                        if not ANSWER_REVIEW and buffer:
                            if not runs.append(run_id, buffer):
                                return
                            buffer = ''
                if not answer_length:
                    raise ValueError("empty model response")
                if ANSWER_REVIEW:
                    await publish_checked(llm, run_id, question, buffer, sources, usage)
                buffer = ""
                runs.finish(run_id, "completed")
        except asyncio.CancelledError:
            runs.finish(run_id, "interrupted" if self.closing else "cancelled", "服务停止，回答已中断。请重新发送问题。" if self.closing else "")
        except TimeoutError:
            runs.finish(run_id, "failed", "本次回答超时。已保存部分内容，请稍后重新提问。")
        except AnswerLimit:
            runs.finish(run_id, "failed", "回答达到长度上限，请缩小问题范围后重试。已发布内容会保留。")
        except EvidenceCheckError:
            runs.finish(run_id, "failed", "回答的证据核对未完成，未发布草稿。已保留参考片段，请稍后重试。")
        except Exception as error:
            # Keep stack locations for debugging without logging provider error bodies or credentials.
            log.error("Question run %s failed (%s)\n%s", run_id, type(error).__name__, "".join(format_list(extract_tb(error.__traceback__))))
            detail = str(error) if isinstance(error, RuntimeError) and "需要在 .env 中配置" in str(error) else f"回答未完成（{type(error).__name__}），请检查模型配置和网络后重新提问。"
            if isinstance(error, (TypeError, AttributeError)):
                detail = "回答执行遇到内部错误，请重试；若再次出现，请查看后端日志。"
            runs.finish(run_id, "failed", detail)

    def cancel(self, run_id):
        with catalog_lock, SessionLocal() as session:
            run = session.get(Run, run_id)
            if run is None or run.status not in ACTIVE:
                return
            run.status, run.phase = "cancelling", "正在停止"
            event(session, run_id, "status", {"status": "cancelling", "phase": "正在停止"})
            session.commit()
        task = self.tasks.get(run_id)
        if task:
            task.cancel()
        else:
            runs.finish(run_id, "cancelled")

    def cancel_scope(self, course_id, conversation_id=None):
        with SessionLocal() as session:
            query = select(Run.id).join(Conversation, Run.conversation_id == Conversation.id).where(Conversation.course_id == course_id, Run.status.in_(ACTIVE))
            if conversation_id:
                query = query.where(Conversation.id == conversation_id)
            ids = list(session.scalars(query))
        for run_id in ids:
            self.cancel(run_id)

    async def close(self):
        self.closing = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.offloads:
            await asyncio.gather(*list(self.offloads), return_exceptions=True)
