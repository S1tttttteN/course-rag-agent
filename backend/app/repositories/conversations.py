from sqlalchemy import select
from app.db.models import Conversation, Message, Run


def course_conversations(session, course_id):
    return list(session.scalars(select(Conversation).where(Conversation.course_id == course_id).order_by(Conversation.updated_at.desc(), Conversation.id)))


def conversation_messages(session, conversation_id):
    return list(session.scalars(select(Message).join(Run, Message.run_id == Run.id).where(Message.conversation_id == conversation_id).order_by(Run.created_at, Run.id, Message.role.desc())))


def completed_turns(session, conversation_id, exclude_run_id):
    runs = list(session.scalars(select(Run).where(
        Run.conversation_id == conversation_id, Run.id != exclude_run_id, Run.status == "completed",
    ).order_by(Run.created_at.desc()).limit(3)))
    result = []
    for run in reversed(runs):
        answer = session.scalar(select(Message).where(Message.run_id == run.id, Message.role == "assistant"))
        if answer:
            result.append({"question": run.question[:2000], "answer": answer.content[:2000]})
    return result
