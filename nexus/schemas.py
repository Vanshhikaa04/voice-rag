"""Request validation. Anything the API accepts is bounded here."""
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from .agents import WORKERS


class Register(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=10, max_length=200)
    workspace: str = Field(default="My workspace", min_length=1, max_length=120)


class Login(BaseModel):
    email: EmailStr
    password: str = Field(max_length=200)


class Invite(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=10, max_length=200)
    role: Literal["admin", "member"] = "member"


class UserPatch(BaseModel):
    role: Literal["admin", "member"] | None = None
    active: bool | None = None


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    channel: Literal["text", "voice"] = "text"


class Search(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=30)


class Feedback(BaseModel):
    value: Literal[-1, 0, 1]


class Speak(BaseModel):
    text: str = Field(min_length=1, max_length=2400)


class Visibility(BaseModel):
    visibility: Literal["workspace", "private"]


class ConnectorIn(BaseModel):
    kind: Literal["website", "github", "slack", "gdrive", "gmail", "postgres"]
    name: str = Field(min_length=1, max_length=200)
    config: dict[str, str] = {}
    secrets: dict[str, str] = {}
    visibility: Literal["workspace", "private"] = "workspace"


class MemoryIn(BaseModel):
    content: str = Field(min_length=1, max_length=1000)


class RunIn(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    workflow: str = Field(default="ask", max_length=40)
    document_ids: list[str] = Field(default=[], max_length=6)
    transcript: str = Field(default="", max_length=60000)
    repo: str = Field(default="", max_length=140)
    issue: int | None = None


class Step(BaseModel):
    agent: str
    task: str = Field(min_length=1, max_length=2000)

    @field_validator("agent")
    @classmethod
    def known(cls, v: str) -> str:
        if v not in WORKERS:
            raise ValueError(f"Unknown agent: {v}")
        return v


class WorkflowIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    steps: list[Step] = Field(min_length=1, max_length=8)
    output: Literal["text", "docx", "pptx"] = "text"


class Decision(BaseModel):
    approve: bool
