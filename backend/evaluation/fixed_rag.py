"""执行仅用于离线对照的固定 RAG。"""

import asyncio
from contextlib import aclosing
from time import monotonic
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.models import get_llm
from app.qa.history import answer_message, is_summary, question_message, recent_history
from app.qa.messages import ANSWER_CHAR_LIMIT, answer_messages, output_was_truncated, text_content
from app.qa.metrics import RunUsage
from app.retrieval.search import search_course_material
from app.services.runs import unregistered_citations


async def rewrite_query(llm, question, history, callbacks=None):
    """将追问改写成可独立检索的问题。

    Args:
        llm: 用于调用对话模型的客户端。
        question: 用户提交的问题正文。
        history: 可选摘要及按时间升序排列的已完成问答消息。
        callbacks: 传给模型调用的可选回调列表。

    Returns:
        查询文本与降级说明。无历史时使用原问题，改写失败时拼接上一问题与当前问题。
    """
    if not history:
        return question, ""
    previous = "\n".join(
        f"{'历史摘要' if is_summary(message) else '助手' if isinstance(message, AIMessage) else '用户'}：{message.content}"
        for message in history
    )
    try:
        async with asyncio.timeout(25):
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content="将当前问题改写为可独立检索的中文问题。仅根据对话补全代词和省略的主题，不回答、不添加新知识、不改变用户意图。直接输出问题，不超过500字。"
                    ),
                    HumanMessage(
                        content=f"历史对话（仅用于理解指代）：\n{previous}\n\n当前问题：{question}"
                    ),
                ],
                config={"callbacks": callbacks or []},
            )
        rewritten = text_content(response).strip()
        if rewritten:
            return rewritten[:500], ""
    except Exception as error:
        note = f"追问改写暂不可用（{type(error).__name__}），已结合上一问题检索。"
    else:
        note = "追问改写未返回文本，已结合上一问题检索。"
    previous_question = next(
        (message.content for message in reversed(history) if isinstance(message, HumanMessage)),
        "",
    )
    return (
        f"{previous_question[:300]}\n{question}",
        note,
    )


async def run_fixed_turn(course, question, history, timeout):
    """执行固定 RAG 一轮，并把完成的问答加入该场景的历史。

    Args:
        course: 评测环境的课程 ID。
        question: 当前问题。
        history: 当前配置和场景下已完成的消息列表。
        timeout: 整轮执行秒数。

    Returns:
        与 Agent 评测共用的回答结构和进程内耗时。
    """
    started = monotonic()
    usage = RunUsage()
    trace = {}
    citations = []
    parts = []
    length = 0
    status = "completed"
    error = ""
    try:
        async with asyncio.timeout(timeout):
            model = get_llm()
            messages = recent_history(history)
            query, note = await rewrite_query(model, question, messages, [usage])
            result = await asyncio.to_thread(search_course_material, course, query)
            trace = {**result["trace"], "rewrite_note": note}
            citations = [
                {**row, "number": number, "chunk_id": row["id"], "kind": "course"}
                for number, row in enumerate(result["results"], 1)
            ]
            async with aclosing(model.astream(
                answer_messages(question, messages, citations),
                config={"callbacks": [usage]},
            )) as stream:
                async for chunk in stream:
                    text = text_content(chunk)
                    length += len(text)
                    if length > ANSWER_CHAR_LIMIT or output_was_truncated(chunk):
                        raise ValueError("模型输出达到长度上限")
                    parts.append(text)
            if not "".join(parts).strip():
                raise ValueError("模型未生成回答")
    except TimeoutError:
        status, error = "timeout", "本轮回答超时。"
    except Exception as exc:
        status, error = "failed", f"本轮回答未完成（{type(exc).__name__}）。"

    text = "".join(parts)
    if status == "completed":
        turn_id = uuid4().hex
        history[:] = recent_history([
            *history, question_message(turn_id, question), answer_message(turn_id, text),
        ])
    trace["unknown_citations"] = unregistered_citations(text, {row["number"] for row in citations})
    trace["usage"] = usage.summary()
    answer = {
        "status": status, "content": text, "citations": citations,
        "run": {"trace": trace, "error": error},
    }
    return answer, round(monotonic() - started, 3)
