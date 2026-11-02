"""Long-term memory: private to one user, always listable and deletable."""
import logging
import re

from sqlalchemy.orm import Session

from . import llm, vectors
from .db import Memory
from .security import Principal

log = logging.getLogger("nexus.memory")

_EXPLICIT = re.compile(r"^\s*(please\s+)?(remember( that)?|note that|yaad rakh(o|na)?( ki)?)\s+(?P<fact>.+)$", re.I | re.S)
_SYSTEM = (
    "Decide which durable facts about the user or their projects are worth remembering from this exchange: "
    "preferences, roles, ongoing projects, standing decisions. Ignore one-off questions, small talk and anything "
    'stated only by the assistant. Never store credentials or secrets. Return JSON {"facts":[str]}, usually empty.'
)
_SECRETS = re.compile(r"(api[_ -]?key|password|secret|token)\b|sk-[a-z0-9]{8,}", re.I)


def explicit_fact(text: str) -> str | None:
    m = _EXPLICIT.match(text)
    return m.group("fact").strip() if m else None


def add(db: Session, p: Principal, content: str, source: str = "explicit") -> Memory:
    item = Memory(workspace_id=p.workspace_id, user_id=p.id, content=content.strip()[:1000], source=source)
    db.add(item)
    db.commit()
    vectors.upsert_memory(item)
    return item


def delete(db: Session, p: Principal, memory_id: str) -> bool:
    item = db.query(Memory).filter_by(id=memory_id, user_id=p.id).one_or_none()
    if not item:
        return False
    vectors.delete_memory(item.id)
    db.delete(item)
    db.commit()
    return True


def learn(db: Session, p: Principal, question: str, answer: str) -> None:
    """Best effort, after the reply has been sent. A failure here must never affect the user's answer."""
    if not llm.available() or len(question) < 20:
        return
    try:
        facts = llm.chat_json(_SYSTEM, f"User: {question}\nAssistant: {answer[:600]}", purpose="memory",
                              workspace_id=p.workspace_id, max_tokens=200).get("facts", [])
        known = {m.content.lower() for m in db.query(Memory).filter_by(user_id=p.id).all()}
        for fact in facts[:3]:
            if isinstance(fact, str) and fact.lower() not in known and not _SECRETS.search(fact):
                add(db, p, fact, "extracted")
    except Exception as e:
        log.warning("memory extraction skipped: %s", e)
