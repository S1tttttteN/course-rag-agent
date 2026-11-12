"""定义课程资料和对话的数据模型，SQLite 保存原文片段、运行事件与引用快照。"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint, false, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 SQLAlchemy ORM 模型的声明基类。"""

    pass


def utc_now() -> str:
    """获取当前的 UTC 时间。

    Returns:
        包含时区偏移的 ISO 8601 时间字符串。
    """
    return datetime.now(timezone.utc).isoformat()


class Course(Base):
    """课程目录中的一门课程。

    Attributes:
        id: 课程 ID。
        name: 课程名称。
        description: 课程说明。
        created_at: 创建时间。
        updated_at: 最近更新时间。
    """

    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, default=utc_now)
    updated_at: Mapped[str] = mapped_column(
        String(40), nullable=False, default=utc_now, onupdate=utc_now
    )


class Document(Base):
    """一门课程中的资料及其当前发布版本。

    Attributes:
        id: 资料 ID。
        course_id: 所属课程 ID。
        name: 课程内唯一的资料名称。
        active_version_id: 已发布版本 ID，尚未发布时为 None。
        created_at: 创建时间。
        updated_at: 最近更新时间。
    """

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("course_id", "name"),)

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    course_id: Mapped[str] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    active_version_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class DocumentVersion(Base):
    """一次资料上传所产生的索引版本。

    Attributes:
        id: 资料版本 ID。
        document_id: 所属资料 ID。
        number: 资料内的版本号。
        filename: 保存的原件文件名。
        content_hash: 原件的 SHA-256 摘要。
        size: 原件字节数。
        status: 索引处理状态。
        step: 索引进度说明。
        error: 处理失败时的说明。
        chunk_count: 发布的片段数量。
        embedding_profile: 建立向量索引时使用的嵌入配置标识。
        created_at: 创建时间。
    """

    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "number"),)

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64))
    size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")
    step: Mapped[str] = mapped_column(String(100), default="等待处理")
    error: Mapped[str] = mapped_column(Text, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_profile: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Chunk(Base):
    """资料版本中的可检索原文片段。

    Attributes:
        id: 片段 ID。
        version_id: 所属版本 ID。
        ordinal: 片段在版本内的顺序。
        text: 片段原文。
        page: 来源页码，未知时为 None。
        start_index: 片段在来源文本中的起始位置。
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    start_index: Mapped[int] = mapped_column(Integer, default=0)


class CleanupJob(Base):
    """记录待清理的课程或资料存储。

    任务与目录记录独立保存，以便删除课程或资料后仍可清理原件和向量。

    Attributes:
        id: 清理任务 ID。
        course_id: 待清理范围的课程 ID。
        document_id: 待清理资料 ID，为 None 时清理整门课程。
        attempts: 已记录的失败次数。
        error: 最近一次清理失败的说明。
        created_at: 创建时间。
    """

    __tablename__ = "cleanup_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    course_id: Mapped[str] = mapped_column(String(36))
    document_id: Mapped[str | None] = mapped_column(String(36))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Conversation(Base):
    """一门课程中的问答会话。

    Attributes:
        id: 对话 ID。
        course_id: 所属课程 ID。
        title: 对话标题。
        created_at: 创建时间。
        updated_at: 最近更新时间。
    """

    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    course_id: Mapped[str] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(80), default="新对话")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Run(Base):
    """一次问答及其执行状态。

    Attributes:
        id: 问答运行 ID。
        conversation_id: 所属对话 ID。
        question: 用户原始问题。
        mode: 问答模式。
        query: 实际用于检索的查询。
        status: 执行状态。
        phase: 当前阶段说明。
        error: 失败或中断说明。
        trace: 序列化为 JSON 的执行摘要。
        top_k: 检索候选条数。
        semantic: 语义检索开关，以整数保存。
        rerank: 重排序开关，以整数保存。
        web_search: 本轮是否允许 Agent 联网。
        answer_review: 本轮是否在发布正文前核对证据。
        created_at: 创建时间。
    """

    __tablename__ = "runs"
    __table_args__ = (
        Index(
            "uq_runs_active_conversation",
            "conversation_id",
            unique=True,
            sqlite_where=text("status IN ('queued', 'running', 'cancelling')"),
        ),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(10), default="rag")
    query: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="queued")
    phase: Mapped[str] = mapped_column(String(100), default="等待回答")
    error: Mapped[str] = mapped_column(Text, default="")
    trace: Mapped[str] = mapped_column(Text, default="{}")
    top_k: Mapped[int] = mapped_column(Integer, default=4)
    semantic: Mapped[int] = mapped_column(Integer, default=1)
    rerank: Mapped[int] = mapped_column(Integer, default=1)
    web_search: Mapped[bool] = mapped_column(Boolean, server_default=false())
    answer_review: Mapped[bool] = mapped_column(Boolean, server_default=false())
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Message(Base):
    """问答运行中的一条用户或助手消息。

    Attributes:
        id: 消息 ID。
        conversation_id: 所属对话 ID。
        run_id: 所属运行 ID。
        role: 消息角色。
        content: 消息正文。
        status: 消息处理状态。
        created_at: 创建时间。
    """

    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("run_id", "role"),)
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Citation(Base):
    """回答生成时保存的来源快照。

    Attributes:
        id: 引用 ID。
        run_id: 所属运行 ID。
        number: 本次运行内的引用编号。
        kind: 课程片段、网络摘要或网页正文等来源类型。
        url: 网络来源地址。
        retrieved_at: 网络来源的获取时间。
        document_id: 来源资料 ID。
        version_id: 来源资料版本 ID。
        chunk_id: 来源片段 ID。
        source: 来源名称。
        version: 资料版本号。
        text: 保存的原文摘录。
        page: 来源页码，未知时为 None。
        start_index: 摘录在来源文本中的起始位置。
    """

    __tablename__ = "citations"
    __table_args__ = (UniqueConstraint("run_id", "number"),)
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20), default="course")
    url: Mapped[str] = mapped_column(Text, default="")
    retrieved_at: Mapped[str] = mapped_column(String(40), default="")
    # 引用保存回答当时的原文和来源信息。资料替换或删除后，历史回答仍可展示同一份依据。
    document_id: Mapped[str] = mapped_column(String(36))
    version_id: Mapped[str] = mapped_column(String(36))
    chunk_id: Mapped[str] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    start_index: Mapped[int] = mapped_column(Integer, default=0)


class RunEvent(Base):
    """供 SSE 按游标回放的持久化运行事件。

    Attributes:
        id: 自增事件 ID。
        run_id: 所属运行 ID。
        kind: SSE 事件类型。
        data: 序列化为 JSON 的事件负载。
    """

    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30))
    data: Mapped[str] = mapped_column(Text)
