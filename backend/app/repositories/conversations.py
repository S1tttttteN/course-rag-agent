"""封装对话目录、消息历史和已完成问答的查询。"""

from sqlalchemy import select
from app.db.models import Conversation, Message, Run


def course_conversations(session, course_id):
    """查询课程的对话目录。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。

    Returns:
        按更新时间降序排列的对话列表，同一时间按 ID 排序。
    """
    return list(
        session.scalars(
            select(Conversation)
            .where(Conversation.course_id == course_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
        )
    )


def conversation_messages(session, conversation_id):
    """查询对话中的用户与助手消息。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        conversation_id: 对话 ID。

    Returns:
        按运行时间和 ID 排列的消息列表，每轮先列用户消息。
    """
    return list(
        session.scalars(
            select(Message)
            .join(Run, Message.run_id == Run.id)
            .where(Message.conversation_id == conversation_id)
            .order_by(Run.created_at, Run.id, Message.role.desc())
        )
    )


def completed_turns(session, conversation_id, exclude_run_id):
    """按时间顺序返回最近三轮已完成问答作为模型上下文，排除当前运行。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        conversation_id: 对话 ID。
        exclude_run_id: 需要排除的当前运行 ID。

    Returns:
        按时间升序排列的问答字典列表，问题和回答分别截断到 2000 个字符。
    """
    runs = list(
        session.scalars(
            select(Run)
            .where(
                Run.conversation_id == conversation_id,
                Run.id != exclude_run_id,
                Run.status == "completed",
            )
            .order_by(Run.created_at.desc())
            .limit(3)
        )
    )
    result = []
    for run in reversed(runs):
        answer = session.scalar(
            select(Message).where(Message.run_id == run.id, Message.role == "assistant")
        )
        if answer:
            result.append(
                {"question": run.question[:2000], "answer": answer.content[:2000]}
            )
    return result
