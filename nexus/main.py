"""FastAPI application. Run: python -m uvicorn nexus.main:app --port 8100"""
import asyncio
import json
import logging
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from . import agents, analytics, cache, connectors, graph, ingest, llm, memory, parsers, proactive, rag, vectors, voice
from . import schemas as S
from .config import keys, settings
from .db import (AgentRun, AgentStep, Approval, AuditLog, Connector, Conversation, Document, Insight, Memory, Message,
                 SessionLocal, User, Workflow, Workspace, audit, engine, get_db, now)
from .security import (COOKIE, Principal, Throttle, current_user, hash_password, login_throttle, make_token, require_admin,
                       verify_password)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("nexus")
chat_throttle = Throttle(40, 60)
upload_throttle = Throttle(60, 60)


async def _scheduler():
    """Keeps connected sources fresh. One connector at a time, off the event loop."""
    def tick():
        with SessionLocal() as db:
            for conn in connectors.due(db):
                connectors.sync(db, conn)
    while True:
        await asyncio.sleep(60)
        try:
            await asyncio.to_thread(tick)
        except Exception:
            log.exception("scheduler tick failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine()
    with SessionLocal() as db:                 # a restart must not leave a connector stuck
        db.query(Connector).filter_by(status="syncing").update({"status": "idle"})
        db.query(AgentRun).filter(AgentRun.status.in_(["queued", "running"])).update(
            {"status": "failed", "error": "Interrupted by a server restart", "finished_at": now()})
        db.commit()
    task = asyncio.create_task(_scheduler()) if settings().scheduler_enabled else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="NEXUS AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings().cors_origins.split(","), allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    r = await call_next(request)
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "same-origin"
    return r


@app.exception_handler(llm.LLMUnavailable)
async def _llm_down(_: Request, e: llm.LLMUnavailable):
    return JSONResponse({"detail": str(e)}, status_code=503)


@app.exception_handler(voice.VoiceUnavailable)
async def _voice_down(_: Request, e: voice.VoiceUnavailable):
    return JSONResponse({"detail": str(e)}, status_code=503)


def _cookie(resp: Response, token: str) -> None:
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", secure=settings().cookie_secure,
                    max_age=settings().token_ttl_minutes * 60, path="/")


def _user(u: User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "active": u.active,
            "workspace_id": u.workspace_id, "created_at": u.created_at.isoformat()}


# ---- auth ----------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/auth/register")
def register(body: S.Register, response: Response, db: Session = Depends(get_db)):
    """Self-service signup always creates a new, isolated workspace. Joining one needs an admin invite."""
    if not settings().allow_signup:
        raise HTTPException(403, "Signup is disabled")
    email = body.email.lower()
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "That email is already registered")
    ws = Workspace(name=body.workspace)
    db.add(ws)
    db.flush()
    u = User(workspace_id=ws.id, email=email, name=body.name, password_hash=hash_password(body.password), role="admin")
    db.add(u)
    db.commit()
    _cookie(response, make_token(u))
    return _user(u)


@app.post("/api/auth/login")
def login(body: S.Login, request: Request, response: Response, db: Session = Depends(get_db)):
    email = body.email.lower()
    login_throttle.check(f"{request.client.host}:{email}")
    u = db.query(User).filter_by(email=email).first()
    if not u or not u.active or not verify_password(u.password_hash, body.password):
        raise HTTPException(401, "Incorrect email or password")
    _cookie(response, make_token(u))
    return _user(u)


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return {**_user(db.get(User, p.id)), "workspace": db.get(Workspace, p.workspace_id).name}


# ---- users and audit ------------------------------------------------------------------------------
@app.get("/api/users")
def users(p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    return [_user(u) for u in db.query(User).filter_by(workspace_id=p.workspace_id).order_by(User.created_at)]


@app.post("/api/users")
def invite(body: S.Invite, p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "That email is already registered")
    u = User(workspace_id=p.workspace_id, email=email, name=body.name, password_hash=hash_password(body.password), role=body.role)
    db.add(u)
    db.commit()
    audit(db, p, "user.added", email, role=body.role)
    return _user(u)


@app.patch("/api/users/{user_id}")
def update_user(user_id: str, body: S.UserPatch, p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    u = db.query(User).filter_by(id=user_id, workspace_id=p.workspace_id).one_or_none()
    if not u:
        raise HTTPException(404, "User not found")
    if u.id == p.id:
        raise HTTPException(400, "You cannot change your own role or status")
    if body.role is not None:
        u.role = body.role
    if body.active is not None:
        u.active = body.active
    db.commit()
    audit(db, p, "user.updated", u.email, role=u.role, active=u.active)
    return _user(u)


@app.get("/api/audit")
def audit_log(p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.query(AuditLog).filter_by(workspace_id=p.workspace_id).order_by(AuditLog.created_at.desc()).limit(200).all()
    names = {u.id: u.email for u in db.query(User).filter_by(workspace_id=p.workspace_id)}
    return [{"id": a.id, "at": a.created_at.isoformat(), "user": names.get(a.user_id, "system"), "action": a.action, "target": a.target,
             "detail": a.detail} for a in rows]


# ---- documents and search -------------------------------------------------------------------------
def _visible(q, p: Principal):
    return q.filter(Document.workspace_id == p.workspace_id, or_(Document.visibility == "workspace", Document.owner_id == p.id))


def _doc(d: Document) -> dict:
    return {"id": d.id, "title": d.title, "url": d.url, "kind": d.kind, "mime": d.mime, "visibility": d.visibility, "chunks": d.chunk_count,
            "connector_id": d.connector_id, "owner_id": d.owner_id, "updated_at": d.updated_at.isoformat()}


@app.post("/api/documents")
async def upload(file: UploadFile = File(...), visibility: str = Form("workspace"), p: Principal = Depends(current_user),
                 db: Session = Depends(get_db)):
    upload_throttle.check(p.id)
    if visibility not in ("workspace", "private"):
        raise HTTPException(422, "Invalid visibility")
    name = Path(file.filename or "upload").name
    data = await file.read(settings().max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings().max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Files are limited to {settings().max_upload_mb} MB")

    def work():
        text = parsers.parse(name, data, p.workspace_id)
        return ingest.upsert(db, p.workspace_id, p.id, ingest.Item(f"upload:{p.id}:{name}", name, text, kind="file", mime=file.content_type or ""),
                             visibility=visibility)
    try:
        doc, state = await asyncio.to_thread(work)
    except parsers.ParseError as e:
        raise HTTPException(422, str(e))
    if name.lower().endswith((".csv", ".xlsx")):               # the Data Analyst computes on the original table
        folder = settings().data_dir / "uploads"
        folder.mkdir(exist_ok=True)
        (folder / doc.id).write_bytes(data)
    audit(db, p, "document.uploaded", name, state=state)
    if state != "unchanged":
        threading.Thread(target=_review_upload, args=(doc.id,), daemon=True).start()
    return {**_doc(doc), "state": state}


def _review_upload(doc_id: str) -> None:
    with SessionLocal() as db:
        if doc := db.get(Document, doc_id):
            proactive.review(db, [doc])


@app.get("/api/documents")
def documents(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return [_doc(d) for d in _visible(db.query(Document), p).order_by(Document.updated_at.desc()).limit(500)]


def _own_doc(db: Session, p: Principal, doc_id: str) -> Document:
    d = _visible(db.query(Document), p).filter(Document.id == doc_id).one_or_none()
    if not d:
        raise HTTPException(404, "Document not found")
    if d.owner_id != p.id and not p.is_admin:
        raise HTTPException(403, "Only the owner or an admin can change this document")
    return d


@app.patch("/api/documents/{doc_id}")
def set_visibility(doc_id: str, body: S.Visibility, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    d = _own_doc(db, p, doc_id)
    d.visibility = body.visibility
    vectors.set_visibility(d.id, body.visibility)
    db.commit()
    audit(db, p, "document.visibility", d.title, visibility=body.visibility)
    return _doc(d)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    d = _own_doc(db, p, doc_id)
    title = d.title
    (settings().data_dir / "uploads" / d.id).unlink(missing_ok=True)
    ingest.remove(db, d)
    audit(db, p, "document.deleted", title)
    return {"ok": True}


@app.post("/api/search")
def search(body: S.Search, p: Principal = Depends(current_user)):
    hits = rag.retrieve(p, body.query, body.limit)
    return [{"document_id": h.document_id, "title": h.title, "url": h.url, "text": h.text, "relevance": round(h.cosine, 3)} for h in hits]


# ---- chat -----------------------------------------------------------------------------------------
@app.post("/api/chat")
def chat(body: S.Chat, p: Principal = Depends(current_user)):
    chat_throttle.check(p.id)

    def events():
        with SessionLocal() as db:                  # outlives the request-scoped dependency
            for ev in rag.answer(db, p, body.message, body.conversation_id, body.channel):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/conversations")
def conversations(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(Conversation).filter_by(user_id=p.id).order_by(Conversation.updated_at.desc()).limit(100).all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in rows]


def _conv(db: Session, p: Principal, cid: str) -> Conversation:
    c = db.query(Conversation).filter_by(id=cid, user_id=p.id).one_or_none()
    if not c:
        raise HTTPException(404, "Conversation not found")
    return c


@app.get("/api/conversations/{cid}")
def conversation(cid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    c = _conv(db, p, cid)
    rows = db.query(Message).filter_by(conversation_id=c.id).order_by(Message.created_at).all()
    return {"id": c.id, "title": c.title,
            "messages": [{"id": m.id, "role": m.role, "content": m.content, "meta": m.meta, "feedback": m.feedback} for m in rows]}


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    db.delete(_conv(db, p, cid))
    db.commit()
    return {"ok": True}


@app.post("/api/messages/{mid}/feedback")
def feedback(mid: str, body: S.Feedback, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    m = db.query(Message).join(Conversation, Message.conversation_id == Conversation.id).filter(
        Message.id == mid, Conversation.user_id == p.id).one_or_none()
    if not m:
        raise HTTPException(404, "Message not found")
    m.feedback = body.value
    db.commit()
    return {"ok": True}


# ---- voice ----------------------------------------------------------------------------------------
@app.post("/api/voice/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(""), p: Principal = Depends(current_user)):
    upload_throttle.check(p.id)
    data = await file.read(25 * 1024 * 1024)
    if not data:
        raise HTTPException(422, "Empty audio")
    try:
        return await asyncio.to_thread(voice.transcribe, data, file.filename or "speech.webm", language or None, p.workspace_id)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Speech service failed: {e}")


@app.post("/api/voice/speak")
def speak(body: S.Speak, p: Principal = Depends(current_user)):
    chat_throttle.check(p.id)
    try:
        audio, mime = voice.speak(body.text, rag.detect_language(body.text))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Speech service failed: {e}")
    return Response(audio, media_type=mime)


# ---- connectors -----------------------------------------------------------------------------------
def _conn(c: Connector) -> dict:
    return {"id": c.id, "kind": c.kind, "name": c.name, "config": c.config, "visibility": c.visibility, "status": c.status,
            "last_error": c.last_error, "last_synced_at": c.last_synced_at.isoformat() if c.last_synced_at else None,
            "has_secret": bool(c.secret), "owner_id": c.owner_id}


@app.get("/api/connectors/catalog")
def catalog(p: Principal = Depends(current_user)):
    return [{"kind": k, "label": c.label, "fields": c.fields, "oauth": getattr(c, "oauth", False), "missing": connectors.requirements(k),
             "admin_only": k == "postgres"} for k, c in connectors.REGISTRY.items()]


@app.get("/api/connectors")
def list_connectors(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(Connector).filter(Connector.workspace_id == p.workspace_id, or_(Connector.visibility == "workspace", Connector.owner_id == p.id))
    out = []
    for c in rows.order_by(Connector.created_at.desc()):
        out.append({**_conn(c), "documents": db.query(Document).filter_by(connector_id=c.id).count()})
    return out


def _start_sync(conn_id: str) -> None:
    def work():
        with SessionLocal() as db:
            connectors.sync(db, db.get(Connector, conn_id))
    threading.Thread(target=work, daemon=True).start()


@app.post("/api/connectors")
def create_connector(body: S.ConnectorIn, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    impl = connectors.REGISTRY[body.kind]
    if body.kind == "postgres" and not p.is_admin:
        raise HTTPException(403, "Only admins can connect databases")
    if getattr(impl, "oauth", False):
        raise HTTPException(422, "This source connects through its authorization screen")
    if body.kind == "website":
        from .security import assert_public_url

        try:
            assert_public_url(body.config.get("url", ""))
        except ValueError as e:
            raise HTTPException(422, str(e))
    required = [f["name"] for f in impl.fields if not f["secret"]]
    if missing := [f for f in required if not body.config.get(f)]:
        raise HTTPException(422, f"Missing: {', '.join(missing)}")
    c = Connector(workspace_id=p.workspace_id, owner_id=p.id, kind=body.kind, name=body.name, config=body.config, visibility=body.visibility)
    connectors.save_secret(c, {k: v for k, v in body.secrets.items() if v})
    db.add(c)
    db.commit()
    audit(db, p, "connector.created", body.name, kind=body.kind)
    _start_sync(c.id)
    return _conn(c)


def _managed(db: Session, p: Principal, cid: str) -> Connector:
    c = db.query(Connector).filter_by(id=cid, workspace_id=p.workspace_id).one_or_none()
    if not c or (c.owner_id != p.id and not p.is_admin):
        raise HTTPException(404, "Connector not found")
    return c


@app.post("/api/connectors/{cid}/sync")
def sync_connector(cid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    c = _managed(db, p, cid)
    if c.status == "syncing":
        raise HTTPException(409, "Already syncing")
    _start_sync(c.id)
    return {"ok": True}


@app.delete("/api/connectors/{cid}")
def delete_connector(cid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    c = _managed(db, p, cid)
    for d in db.query(Document).filter_by(connector_id=c.id).all():
        ingest.remove(db, d)
    name = c.name
    db.delete(c)
    db.commit()
    audit(db, p, "connector.deleted", name)
    return {"ok": True}


@app.get("/api/connectors/oauth/{kind}/start")
def oauth_start(kind: str, p: Principal = Depends(current_user)):
    if kind not in connectors.GOOGLE_SCOPES:
        raise HTTPException(404, "Unknown connector")
    if missing := connectors.requirements(kind):
        raise HTTPException(503, f"Server is missing {', '.join(missing)}")
    state = secrets.token_urlsafe(24)
    cache.set(f"oauth:{state}", json.dumps({"user": p.id, "ws": p.workspace_id, "kind": kind}), ttl=600)
    s = settings()
    return {"url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": s.google_client_id, "redirect_uri": f"{s.public_url}/api/connectors/oauth/callback", "response_type": "code",
        "scope": connectors.GOOGLE_SCOPES[kind], "access_type": "offline", "prompt": "consent", "state": state})}


@app.get("/api/connectors/oauth/callback")
def oauth_callback(code: str, state: str, db: Session = Depends(get_db)):
    saved = cache.get(f"oauth:{state}")
    if not saved:
        raise HTTPException(400, "Authorization expired; start again")
    info, s = json.loads(saved), settings()
    r = httpx.post("https://oauth2.googleapis.com/token", timeout=30, data={
        "code": code, "client_id": s.google_client_id, "client_secret": s.google_client_secret,
        "redirect_uri": f"{s.public_url}/api/connectors/oauth/callback", "grant_type": "authorization_code"})
    r.raise_for_status()
    refresh = r.json().get("refresh_token")
    if not refresh:
        raise HTTPException(400, "Google did not issue a refresh token")
    c = Connector(workspace_id=info["ws"], owner_id=info["user"], kind=info["kind"], name=connectors.REGISTRY[info["kind"]].label, visibility="private")
    connectors.save_secret(c, {"refresh_token": refresh})
    db.add(c)
    db.commit()
    _start_sync(c.id)
    return RedirectResponse(f"{s.public_url}/sources")


# ---- memory and graph -----------------------------------------------------------------------------
@app.get("/api/memories")
def memories(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(Memory).filter_by(user_id=p.id).order_by(Memory.created_at.desc()).all()
    return [{"id": m.id, "content": m.content, "source": m.source, "created_at": m.created_at.isoformat()} for m in rows]


@app.post("/api/memories")
def add_memory(body: S.MemoryIn, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    m = memory.add(db, p, body.content)
    return {"id": m.id, "content": m.content, "source": m.source, "created_at": m.created_at.isoformat()}


@app.delete("/api/memories/{mid}")
def delete_memory(mid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    if not memory.delete(db, p, mid):
        raise HTTPException(404, "Memory not found")
    return {"ok": True}


@app.get("/api/graph")
def knowledge_graph(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return graph.overview(db, p)


# ---- agents, workflows, approvals -----------------------------------------------------------------
def _run(r: AgentRun, steps: list | None = None) -> dict:
    out = {"id": r.id, "workflow": r.workflow, "goal": r.goal, "status": r.status, "verdict": r.verdict, "result": r.result, "error": r.error,
           "has_artifact": bool(r.artifact_path), "started_at": r.started_at.isoformat(),
           "finished_at": r.finished_at.isoformat() if r.finished_at else None}
    if steps is not None:
        out["steps"] = [{"id": s.id, "agent": s.agent, "task": s.task, "output": s.output, "status": s.status, "ms": round(s.ms)} for s in steps]
    return out


@app.get("/api/agents")
def agent_catalog(p: Principal = Depends(current_user)):
    return {"agents": agents.AGENTS, "available": llm.available(),
            "templates": [{"id": k, "label": v["label"], "output": v["output"], "steps": v["steps"]} for k, v in agents.TEMPLATES.items()]}


@app.post("/api/runs")
def start_run(body: S.RunIn, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    chat_throttle.check(p.id)
    params = {k: v for k, v in body.model_dump(exclude={"goal", "workflow"}).items() if v}
    try:
        run = agents.start(db, p, body.goal, body.workflow, params)
    except agents.AgentError as e:
        raise HTTPException(422, str(e))
    audit(db, p, "run.started", run.id, workflow=body.workflow)
    return _run(run)


@app.get("/api/runs")
def runs(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(AgentRun).filter_by(user_id=p.id).order_by(AgentRun.started_at.desc()).limit(50).all()
    return [_run(r, db.query(AgentStep).filter_by(run_id=r.id).order_by(AgentStep.created_at).all()) for r in rows]


def _own_run(db: Session, p: Principal, rid: str) -> AgentRun:
    r = db.query(AgentRun).filter_by(id=rid, user_id=p.id).one_or_none()
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@app.get("/api/runs/{rid}")
def run_detail(rid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    r = _own_run(db, p, rid)
    return _run(r, db.query(AgentStep).filter_by(run_id=r.id).order_by(AgentStep.created_at).all())


@app.get("/api/runs/{rid}/artifact")
def artifact(rid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    r = _own_run(db, p, rid)
    if not r.artifact_path:
        raise HTTPException(404, "This run produced no file")
    return FileResponse(r.artifact_path, filename=f"nexus-{r.workflow}-{r.id[:6]}{Path(r.artifact_path).suffix}")


@app.get("/api/workflows")
def workflows(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(Workflow).filter_by(workspace_id=p.workspace_id).order_by(Workflow.created_at.desc()).all()
    return [{"id": w.id, "name": w.name, "steps": w.steps, "output": w.output, "mine": w.user_id == p.id} for w in rows]


@app.post("/api/workflows")
def create_workflow(body: S.WorkflowIn, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    w = Workflow(workspace_id=p.workspace_id, user_id=p.id, name=body.name, steps=[s.model_dump() for s in body.steps], output=body.output)
    db.add(w)
    db.commit()
    audit(db, p, "workflow.created", body.name)
    return {"id": w.id, "name": w.name, "steps": w.steps, "output": w.output, "mine": True}


@app.delete("/api/workflows/{wid}")
def delete_workflow(wid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    w = db.query(Workflow).filter_by(id=wid, workspace_id=p.workspace_id).one_or_none()
    if not w or (w.user_id != p.id and not p.is_admin):
        raise HTTPException(404, "Workflow not found")
    db.delete(w)
    db.commit()
    return {"ok": True}


def _approval(a: Approval) -> dict:
    return {"id": a.id, "action": a.action, "payload": a.payload, "status": a.status, "outcome": a.outcome, "run_id": a.run_id,
            "created_at": a.created_at.isoformat()}


@app.get("/api/approvals")
def approvals(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return [_approval(a) for a in db.query(Approval).filter_by(user_id=p.id).order_by(Approval.created_at.desc()).limit(50)]


@app.post("/api/approvals/{aid}")
def decide(aid: str, body: S.Decision, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    a = db.query(Approval).filter_by(id=aid, user_id=p.id).one_or_none()
    if not a:
        raise HTTPException(404, "Approval not found")
    if a.status != "pending":
        raise HTTPException(409, "Already decided")
    a.decided_at = now()
    if not body.approve:
        a.status = "rejected"
    else:
        a.status = "approved"
        db.commit()
        try:
            a.outcome, a.status = agents.execute_approval(db, a), "executed"
        except Exception as e:
            log.warning("approved action failed: %s", e)
            a.outcome, a.status = f"{type(e).__name__}: {e}"[:500], "failed"
    db.commit()
    audit(db, p, f"approval.{a.status}", a.action, payload=a.payload)
    return _approval(a)


# ---- proactive, analytics, system -----------------------------------------------------------------
@app.get("/api/insights")
def insights(p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    rows = (db.query(Insight).outerjoin(Document, Insight.document_id == Document.id).filter(Insight.workspace_id == p.workspace_id)
            .filter(or_(Insight.document_id.is_(None), Document.visibility == "workspace", Document.owner_id == p.id))
            .order_by(Insight.created_at.desc()).limit(50).all())
    return [{"id": i.id, "kind": i.kind, "title": i.title, "detail": i.detail, "due": i.due, "read": i.read, "document_id": i.document_id,
             "created_at": i.created_at.isoformat()} for i in rows]


@app.post("/api/insights/{iid}/read")
def mark_read(iid: str, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    i = db.query(Insight).filter_by(id=iid, workspace_id=p.workspace_id).one_or_none()
    if not i:
        raise HTTPException(404, "Insight not found")
    i.read = True
    db.commit()
    return {"ok": True}


@app.get("/api/analytics")
def analytics_summary(days: int = 30, p: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return analytics.summary(db, p, max(1, min(days, 365)))


@app.post("/api/analytics/eval")
def run_evaluation(p: Principal = Depends(require_admin), db: Session = Depends(get_db)):
    r = analytics.run_eval(db, p)
    audit(db, p, "eval.run", r.id, hit_at_k=r.hit_at_k)
    return {"hit_at_1": r.hit_at_1, "hit_at_k": r.hit_at_k, "mrr": r.mrr, "cases": r.cases, "misses": r.detail["misses"]}


@app.get("/api/system")
def system(p: Principal = Depends(current_user)):
    """What is really active, so the UI never implies a capability that is not there."""
    e, s, k = llm.embedder(), settings(), keys()
    prov = llm.providers()
    return {
        "llm": [{"provider": x.name, "model": x.model} for x in prov],
        "embedder": {"provider": e.name, "model": e.model, "semantic": e.semantic},
        "reranker": s.reranker_model or "fusion score",
        "database": s.database_url.split(":")[0].split("+")[0],
        "vector_store": "qdrant server" if s.qdrant_url else "qdrant embedded",
        "cache": "redis" if s.redis_url else "in-process",
        "stt": voice.stt_provider(), "tts": voice.tts_provider(),
        "scheduler": {"enabled": s.scheduler_enabled, "interval_minutes": s.sync_interval_minutes},
        "keys": {"openai": bool(k.openai_api_key), "anthropic": bool(k.anthropic_api_key), "sarvam": bool(k.sarvam_api_key),
                 "elevenlabs": bool(k.elevenlabs_api_key), "google_oauth": bool(s.google_client_id)},
    }
