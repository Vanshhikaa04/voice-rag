"""SQL schema. Every row that holds user data carries workspace_id, and queries filter on it."""
import uuid
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import settings


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def uid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


def _pk():
    return mapped_column(String(32), primary_key=True, default=uid)


def _ws():
    return mapped_column(String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = _pk()
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    email: Mapped[str] = mapped_column(String(254), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="member")       # admin | member
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Connector(Base):
    __tablename__ = "connectors"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    owner_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24))                         # website | github | slack | gdrive | gmail
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(JSON, default=dict)              # non-secret settings
    secret: Mapped[str] = mapped_column(Text, default="")                 # encrypted credential JSON
    visibility: Mapped[str] = mapped_column(String(12), default="workspace")
    status: Mapped[str] = mapped_column(String(16), default="idle")       # idle | syncing | error | needs_auth
    last_error: Mapped[str] = mapped_column(Text, default="")
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("workspace_id", "source_key"),)
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    owner_id: Mapped[str] = mapped_column(String(32))
    connector_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("connectors.id", ondelete="SET NULL"), nullable=True)
    source_key: Mapped[str] = mapped_column(String(512))                  # stable identity for update detection
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(24), default="file")
    mime: Mapped[str] = mapped_column(String(120), default="")
    visibility: Mapped[str] = mapped_column(String(12), default="workspace")   # workspace | private
    content_hash: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = _pk()
    conversation_id: Mapped[str] = mapped_column(String(32), ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(12))
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback: Mapped[int] = mapped_column(Integer, default=0)             # -1 | 0 | 1
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16), default="explicit")   # explicit | extracted
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("workspace_id", "key"),)
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    key: Mapped[str] = mapped_column(String(240))                         # normalised type:name
    name: Mapped[str] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(String(24))                         # project | person | document | technology | decision | organization


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = (UniqueConstraint("workspace_id", "src_id", "dst_id", "label", "document_id"),)
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    src_id: Mapped[str] = mapped_column(String(32), ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    dst_id: Mapped[str] = mapped_column(String(32), ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(80))
    document_id: Mapped[str] = mapped_column(String(32), ForeignKey("documents.id", ondelete="CASCADE"), index=True)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    workflow: Mapped[str] = mapped_column(String(40), default="ask")
    goal: Mapped[str] = mapped_column(Text)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")     # queued | running | succeeded | failed
    verdict: Mapped[str] = mapped_column(String(16), default="")          # verified | partial | unsupported
    result: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    artifact_path: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentStep(Base):
    __tablename__ = "agent_steps"
    id: Mapped[str] = _pk()
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    agent: Mapped[str] = mapped_column(String(24))
    task: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="running")
    ms: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(40))                       # gmail.create_draft | slack.post | github.comment
    connector_id: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="pending")    # pending | approved | rejected | executed | failed
    outcome: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Insight(Base):
    __tablename__ = "insights"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    kind: Mapped[str] = mapped_column(String(16))                         # update | task | deadline | decision | anomaly
    title: Mapped[str] = mapped_column(String(300))
    detail: Mapped[str] = mapped_column(Text, default="")
    document_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    due: Mapped[str] = mapped_column(String(32), default="")
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


class QueryEvent(Base):
    __tablename__ = "query_events"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    question: Mapped[str] = mapped_column(Text, default="")
    channel: Mapped[str] = mapped_column(String(12), default="text")
    status: Mapped[str] = mapped_column(String(32))
    verified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    top_score: Mapped[float] = mapped_column(Float, default=0)
    total_ms: Mapped[float] = mapped_column(Float, default=0)
    first_token_ms: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


class UsageEvent(Base):
    __tablename__ = "usage_events"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    purpose: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(24))
    model: Mapped[str] = mapped_column(String(80))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = unpriced model
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    cases: Mapped[int] = mapped_column(Integer)
    hit_at_1: Mapped[float] = mapped_column(Float)
    hit_at_k: Mapped[float] = mapped_column(Float)
    mrr: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Workflow(Base):
    """A user-built sequence of agent steps; {input} in a task is replaced by the run's input."""
    __tablename__ = "workflows"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    user_id: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(120))
    steps: Mapped[list] = mapped_column(JSON, default=list)
    output: Mapped[str] = mapped_column(String(8), default="text")        # text | docx | pptx
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[str] = _pk()
    workspace_id: Mapped[str] = _ws()
    user_id: Mapped[str] = mapped_column(String(32), default="")
    action: Mapped[str] = mapped_column(String(60))
    target: Mapped[str] = mapped_column(String(200), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


Index("ix_insight_ws_read", Insight.workspace_id, Insight.read)


def audit(db: Session, p, action: str, target: str = "", **detail) -> None:
    db.add(AuditLog(workspace_id=p.workspace_id, user_id=p.id, action=action, target=target[:200], detail=detail))
    db.commit()


@lru_cache
def engine():
    url = settings().database_url
    args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
    e = create_engine(url, connect_args=args, pool_pre_ping=True)
    if url.startswith("sqlite"):
        event.listen(e, "connect", lambda c, _: (c.execute("PRAGMA journal_mode=WAL"), c.execute("PRAGMA foreign_keys=ON")))
    Base.metadata.create_all(e)
    return e


@lru_cache
def session_factory():
    return sessionmaker(engine(), expire_on_commit=False)


def SessionLocal() -> Session:
    return session_factory()()


def get_db():
    with SessionLocal() as db:
        yield db
