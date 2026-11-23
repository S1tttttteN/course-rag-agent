"""调度后台 Agent 问答任务并控制其运行状态。"""

import asyncio
import logging
from traceback import extract_tb, format_list

from sqlalchemy import select

from app.core.locks import catalog_lock
from app.core.models import RUN_TIMEOUT
from app.db.models import Conversation, Run
from app.db.session import SessionLocal
from app.qa.agent import AnswerLimit, execute_agent
from app.qa.history import load_history, recent_history
from app.retrieval.search import search_course_material
from app.services import runs
from app.services.conversations import ACTIVE, event

log = logging.getLogger(__name__)


class RunManager:
    """管理进程内的问答任务，控制并发、超时、取消与停机收尾。

    Attributes:
        tasks: 按运行 ID 索引的异步任务。
        offloads: 仍在执行的线程操作对应的任务集合。
        semaphore: 限制并行问答的信号量。
        closing: 是否正在停止服务。
    """

    def __init__(self, checkpointer):
        """使用应用级检查点连接初始化问答任务和并发限制。"""
        self.checkpointer = checkpointer
        self.submission_lock = asyncio.Lock()
        self.tasks = {}
        self.offloads = set()
        from app.core.models import CHAT_PROVIDER

        self.semaphore = asyncio.Semaphore(1 if CHAT_PROVIDER == "ollama" else 2)
        self.closing = False

    def start(self, run_id):
        """为已持久化的运行创建异步任务，并在结束时清理任务登记。

        Args:
            run_id: 问答运行 ID。
        """
        task = asyncio.create_task(self.execute(run_id))
        self.tasks[run_id] = task

        def ended(done):
            """移除已结束的任务并补记取消或失败状态。

            Args:
                done: 已结束的问答任务。
            """
            self.tasks.pop(run_id, None)
            if done.cancelled():
                runs.finish(
                    run_id,
                    "interrupted" if self.closing else "cancelled",
                    "服务停止，回答已中断。请重新发送问题。" if self.closing else "",
                )
            elif done.exception():
                log.error(
                    "Run task failed outside handler: %s",
                    type(done.exception()).__name__,
                )
                runs.finish(run_id, "failed", "回答未完成，请重试。")

        task.add_done_callback(ended)

    async def offload(self, function, *args):
        """在线程中执行阻塞操作，并跟踪仍在收尾的操作供停机时等待。

        Args:
            function: 需要在线程中运行的同步可调用对象。
            args: 传给同步函数的位置参数。

        Returns:
            同步函数返回的结果。
        """
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self.offloads.add(task)

        def ended(done):
            """移除已结束的线程任务并取出其异常。

            Args:
                done: 封装线程操作的已结束任务。
            """
            self.offloads.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieve exceptions even if the caller was cancelled.

        task.add_done_callback(ended)
        return await asyncio.shield(task)

    async def retrieve(self, course_id, query):
        """在线程中执行课程检索。

        Args:
            course_id: 当前课程 ID。
            query: 检索问题。

        Returns:
            包含检索结果与执行摘要的字典。
        """
        return await self.offload(search_course_material, course_id, query)

    async def execute(self, run_id):
        """按运行配置执行问答并持久化正文、事件及终态。

        Args:
            run_id: 问答运行 ID。
        """
        try:
            async with self.semaphore, asyncio.timeout(RUN_TIMEOUT):
                with SessionLocal() as session:
                    run = session.get(Run, run_id)
                    if run is None:
                        return
                    conversation = session.get(Conversation, run.conversation_id)
                    course_id, conversation_id = conversation.course_id, conversation.id
                    question = run.question
                    web_search = run.web_search
                config, saved, additions = await load_history(
                    self.checkpointer, conversation_id, run_id
                )
                checkpoint_id = await execute_agent(
                    self, run_id, course_id, conversation_id, question,
                    recent_history(saved + additions), config, saved, additions,
                    web_search=web_search,
                )
                runs.finish(run_id, "completed", agent_checkpoint_id=checkpoint_id)
        except asyncio.CancelledError:
            runs.finish(
                run_id,
                "interrupted" if self.closing else "cancelled",
                "服务停止，回答已中断。请重新发送问题。" if self.closing else "",
            )
        except TimeoutError:
            runs.finish(
                run_id, "failed", "本次回答超时。已保存部分内容，请稍后重新提问。"
            )
        except AnswerLimit:
            runs.finish(
                run_id,
                "failed",
                "回答达到长度上限，请缩小问题范围后重试。已发布内容会保留。",
            )
        except Exception as error:
            # 只保留异常类型和堆栈位置，避免把服务商响应或凭据写入日志。
            log.error(
                "Question run %s failed (%s)\n%s",
                run_id,
                type(error).__name__,
                "".join(format_list(extract_tb(error.__traceback__))),
            )
            detail = (
                str(error)
                if isinstance(error, RuntimeError)
                and "需要在 .env 中配置" in str(error)
                else f"回答未完成（{type(error).__name__}），请检查模型配置和网络后重新提问。"
            )
            if isinstance(error, (TypeError, AttributeError)):
                detail = "回答执行遇到内部错误，请重试；若再次出现，请查看后端日志。"
            runs.finish(run_id, "failed", detail)

    def cancel(self, run_id):
        """将活动运行标为正在取消，并停止对应任务。

        Args:
            run_id: 问答运行 ID。
        """
        with catalog_lock, SessionLocal() as session:
            run = session.get(Run, run_id)
            if run is None or run.status not in ACTIVE:
                return
            run.status, run.phase = "cancelling", "正在停止"
            event(
                session, run_id, "status", {"status": "cancelling", "phase": "正在停止"}
            )
            session.commit()
        task = self.tasks.get(run_id)
        if task:
            task.cancel()
        else:
            runs.finish(run_id, "cancelled")

    async def cancel_scope(self, course_id, conversation_id=None):
        """取消指定范围内的活动运行并等待图写入结束。

        Args:
            course_id: 课程 ID。
            conversation_id: 可选的对话 ID，省略时处理整门课程。
        """
        with SessionLocal() as session:
            query = (
                select(Run.id)
                .join(Conversation, Run.conversation_id == Conversation.id)
                .where(Conversation.course_id == course_id, Run.status.in_(ACTIVE))
            )
            if conversation_id:
                query = query.where(Conversation.id == conversation_id)
            ids = list(session.scalars(query))
        tasks = [self.tasks[run_id] for run_id in ids if run_id in self.tasks]
        for run_id in ids:
            self.cancel(run_id)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        """进入停机状态，等待问答任务和已提交的线程操作结束。"""
        self.closing = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.offloads:
            await asyncio.gather(*list(self.offloads), return_exceptions=True)
