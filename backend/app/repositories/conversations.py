"""封装对话目录、消息历史和已完成问答的查询。"""

from sqlalchemy import and_, or_, select
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


def completed_turns(session, conversation_id, exclude_run_id, *, after=None):
    """按时间倒序读取已完成问答，供调用方按预算截取。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        conversation_id: 对话 ID。
        exclude_run_id: 需要排除的当前运行 ID。
        after: 历史起点对应的已提交运行。

    Yields:
        运行 ID、用户问题和已发布回答。
    """
    query = (
        select(Run.id, Run.question, Message.content)
        .join(Message, Message.run_id == Run.id)
        .where(
            Run.conversation_id == conversation_id,
            Run.id != exclude_run_id,
            Run.status == "completed",
            Message.role == "assistant",
        )
        .order_by(Run.created_at.desc(), Run.id.desc())
        .execution_options(yield_per=10)
    )
    if after is not None:
        query = query.where(
            or_(
                Run.created_at > after.created_at,
                and_(Run.created_at == after.created_at, Run.id > after.id),
            )
        )
    with session.execute(query) as rows:
        yield from rows
