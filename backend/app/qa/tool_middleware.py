"""统一控制 Agent 工具执行的预算、超时和运行记录。"""

import asyncio
import json
from time import monotonic
from uuid import uuid4

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from app.qa.tools import BudgetExceeded
from app.services import runs

MAX_CALLS = 8
TOOL_TIMEOUT = 35


@wrap_tool_call
async def manage_tool_call(request, handler):
    """串行执行一次工具调用，记录公开进度和结果。

    Args:
        request: 包含工具名称、参数及调用 ID 的 LangChain 请求。
        handler: 执行实际工具调用的异步处理函数。

    Returns:
        工具结果或供模型处理的错误消息。

    Raises:
        BudgetExceeded: 调用次数或引用容量已达上限。
        asyncio.CancelledError: 调用被取消或运行已不可更新。
    """
    context = request.runtime.context
    async with context.lock:
        if context.calls >= MAX_CALLS:
            raise BudgetExceeded()
        context.calls += 1
        call = request.tool_call
        name = call["name"]
        label = {
            "search_course": "搜索课程资料",
            "list_documents": "列出课程资料",
            "read_document": "分页阅读资料",
            "search_web": "搜索公开网络",
            "read_webpage": "阅读外部网页",
        }.get(name, "未知工具")
        # Only declared input fields are shown. Never publish tool output or provider exceptions.
        args = call.get("args", {})
        safe = {
            key: str(value)[:220]
            for key, value in args.items()
            if key in {"query", "offset", "limit", "document_id", "url"}
        }
        record = {
            "id": str(uuid4()),
            "number": context.calls,
            "name": name[:60],
            "label": label,
            "input": safe,
            "status": "running",
            "summary": "正在执行",
            "duration_ms": None,
        }
        if not runs.tool_update(context.run_id, record):
            raise asyncio.CancelledError()
        runs.status(context.run_id, f"{label} · {context.calls}/{MAX_CALLS}")
        started = monotonic()
        before = len(context.sources)
        try:
            async with asyncio.timeout(TOOL_TIMEOUT):
                result = await handler(request)
            failed = getattr(result, "status", "success") == "error"
            record.update(
                status="failed" if failed else "completed",
                summary="参数或工具调用无效，请调整后重试。"
                if failed
                else f"已完成，新增 {len(context.sources) - before} 个参考片段。",
            )
            if not failed and name == "list_documents":
                record["summary"] = (
                    f"已查阅目录，累计找到 {len(context.catalog_items)} 份资料。"
                )
            if not failed and name == "search_course" and isinstance(result, ToolMessage):
                record["retrieval"] = json.loads(result.content)["retrieval"]
            if failed:
                result = ToolMessage(
                    content="工具调用失败。检查工具名称和参数，仅使用当前课程返回的资料及版本ID。",
                    tool_call_id=call["id"],
                    status="error",
                )
        except asyncio.CancelledError:
            record.update(status="interrupted", summary="执行已中断。")
            raise
        except BudgetExceeded:
            record.update(status="limited", summary="已达到本轮资料阅读预算。")
            raise
        except Exception as error:
            record.update(
                status="failed",
                summary="工具超时。可尝试缩小范围。"
                if isinstance(error, TimeoutError)
                else "工具未成功。请检查资料是否仍可用及参数是否正确。",
            )
            result = ToolMessage(
                content=record["summary"], tool_call_id=call["id"], status="error"
            )
        finally:
            record["duration_ms"] = round((monotonic() - started) * 1000)
            runs.tool_update(context.run_id, record)
        return result
