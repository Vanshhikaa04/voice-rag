"""Data-source connectors. Each one lists content through the provider's real API and yields ingest Items.

A connector reports what it still needs (`requirements`); it never produces sample data. Syncing is
incremental: ingest skips unchanged content by hash, and full-listing connectors also drop deleted items.
"""
import base64
import logging
import re
from collections.abc import Iterator
from datetime import timedelta
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from sqlalchemy.orm import Session

from . import ingest, parsers, proactive
from .config import settings
from .db import Connector, Document, now
from .ingest import Item
from .security import assert_public_url, decrypt, encrypt

log = logging.getLogger("nexus.connectors")
TIMEOUT = 30
MAX_ITEMS = 300
UA = {"User-Agent": "NexusAI-Connector/1.0"}


class NeedsAuth(RuntimeError):
    """The saved credential was refused; the user has to reconnect."""


def _get(client: httpx.Client, url: str, **kw) -> httpx.Response:
    r = client.get(url, **kw)
    if r.headers.get("x-ratelimit-remaining") == "0":
        raise RuntimeError(f"{urlparse(url).hostname} rate limit reached; add a token or wait for the next sync")
    if r.status_code in (401, 403):
        raise NeedsAuth(f"{urlparse(url).hostname} refused the credential ({r.status_code})")
    r.raise_for_status()
    return r


class Website:
    kind, label, full_listing = "website", "Website", True
    fields = [{"name": "url", "label": "Start URL", "secret": False}]

    def fetch(self, config: dict, creds: dict, client: httpx.Client) -> Iterator[Item]:
        start = assert_public_url(config["url"])
        host = urlparse(start).netloc
        queue, seen = [start], set()
        while queue and len(seen) < int(config.get("max_pages", 50)):
            url = urldefrag(queue.pop(0))[0]
            if url in seen:
                continue
            seen.add(url)
            try:
                assert_public_url(url)                       # every hop, including redirect targets below
                r = client.get(url, headers=UA, follow_redirects=False)
                if r.is_redirect:
                    queue.append(urljoin(url, r.headers["location"]))
                    continue
                r.raise_for_status()
            except (ValueError, httpx.HTTPError) as e:
                log.info("skipping %s: %s", url, e)
                continue
            if "html" not in r.headers.get("content-type", ""):
                continue
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(r.content, "html.parser")
            title = (soup.title.string if soup.title and soup.title.string else url).strip()
            links = [urldefrag(urljoin(url, a["href"]))[0] for a in soup.find_all("a", href=True)]
            queue += [l for l in links if urlparse(l).netloc == host and l not in seen]
            text = parsers.parse("page.html", r.content)
            yield Item(f"web:{url}", title, text, url, "website", "text/html")


class GitHub:
    kind, label, full_listing = "github", "GitHub", False
    fields = [{"name": "repo", "label": "Repository (owner/name)", "secret": False},
              {"name": "token", "label": "Personal access token (optional for public repos)", "secret": True}]
    EXT = {".md", ".txt", ".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java", ".yml", ".yaml", ".toml", ".sql"}

    def _headers(self, creds: dict) -> dict:
        h = {"Accept": "application/vnd.github+json", **UA}
        if creds.get("token"):
            h["Authorization"] = f"Bearer {creds['token']}"
        return h

    def fetch(self, config: dict, creds: dict, client: httpx.Client) -> Iterator[Item]:
        repo, h = config["repo"].strip(), self._headers(creds)
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("Repository must look like owner/name")
        api = f"https://api.github.com/repos/{repo}"
        branch = _get(client, api, headers=h).json()["default_branch"]
        tree = _get(client, f"{api}/git/trees/{branch}?recursive=1", headers=h).json().get("tree", [])
        files = [t for t in tree if t["type"] == "blob" and t.get("size", 0) < 200_000
                 and any(t["path"].endswith(e) for e in self.EXT)]
        files.sort(key=lambda t: (not t["path"].lower().endswith(".md"), t["path"].count("/")))
        # the raw host has no 60-per-hour API limit, so one sync cannot exhaust it
        raw = {k: v for k, v in h.items() if k != "Accept"}
        for t in files[:int(config.get("max_files", 120))]:
            text = _get(client, f"https://raw.githubusercontent.com/{repo}/{branch}/{t['path']}", headers=raw).text
            if text.strip():
                yield Item(f"github:{repo}:{t['path']}", f"{repo}/{t['path']}", text,
                           f"https://github.com/{repo}/blob/{branch}/{t['path']}", "code", "text/plain")
        for issue in _get(client, f"{api}/issues", headers=h, params={"state": "all", "per_page": 50, "sort": "updated"}).json():
            kind = "pull" if "pull_request" in issue else "issue"
            body = f"State: {issue['state']}\nAuthor: {issue['user']['login']}\n\n{issue.get('body') or ''}"
            yield Item(f"github:{repo}:{kind}:{issue['number']}", f"{repo} #{issue['number']}: {issue['title']}", body,
                       issue["html_url"], "issue", "text/plain")

    def comment(self, config: dict, creds: dict, payload: dict) -> str:
        r = httpx.post(f"https://api.github.com/repos/{config['repo']}/issues/{int(payload['number'])}/comments",
                       headers=self._headers(creds), json={"body": payload["body"]}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()["html_url"]


class Slack:
    kind, label, full_listing = "slack", "Slack", False
    fields = [{"name": "token", "label": "Bot token (xoxb-...)", "secret": True}]
    API = "https://slack.com/api"

    def _call(self, client: httpx.Client, method: str, creds: dict, **params) -> dict:
        d = client.get(f"{self.API}/{method}", headers={"Authorization": f"Bearer {creds.get('token', '')}"}, params=params).json()
        if not d.get("ok"):
            if d.get("error") in ("invalid_auth", "not_authed", "token_revoked", "account_inactive"):
                raise NeedsAuth(f"Slack: {d['error']}")
            raise RuntimeError(f"Slack: {d.get('error')}")
        return d

    def fetch(self, config: dict, creds: dict, client: httpx.Client) -> Iterator[Item]:
        for ch in self._call(client, "conversations.list", creds, types="public_channel", limit=100)["channels"]:
            if not ch.get("is_member"):
                continue
            rows = self._call(client, "conversations.history", creds, channel=ch["id"], limit=200)["messages"]
            lines = [f"{m.get('user', 'bot')}: {m['text']}" for m in reversed(rows) if m.get("text")]
            if lines:
                yield Item(f"slack:{ch['id']}", f"Slack #{ch['name']}", "\n".join(lines), "", "chat", "text/plain")

    def post(self, creds: dict, payload: dict) -> str:
        d = httpx.post(f"{self.API}/chat.postMessage", headers={"Authorization": f"Bearer {creds['token']}"},
                       json={"channel": payload["channel"], "text": payload["text"]}, timeout=TIMEOUT).json()
        if not d.get("ok"):
            raise RuntimeError(f"Slack: {d.get('error')}")
        return f"Posted to {payload['channel']}"


GOOGLE_SCOPES = {"gdrive": "https://www.googleapis.com/auth/drive.readonly",
                 "gmail": "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.compose"}


class _Google:
    fields: list = []
    oauth = True

    def access_token(self, creds: dict) -> str:
        s = settings()
        r = httpx.post("https://oauth2.googleapis.com/token", timeout=TIMEOUT, data={
            "client_id": s.google_client_id, "client_secret": s.google_client_secret,
            "refresh_token": creds.get("refresh_token", ""), "grant_type": "refresh_token"})
        if r.status_code >= 400:
            raise NeedsAuth("Google refused the saved authorization")
        return r.json()["access_token"]


class GoogleDrive(_Google):
    kind, label, full_listing = "gdrive", "Google Drive", True
    API = "https://www.googleapis.com/drive/v3/files"
    EXPORT = {"application/vnd.google-apps.document": ("text/plain", ".txt"),
              "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
              "application/vnd.google-apps.presentation": ("text/plain", ".txt")}

    def fetch(self, config: dict, creds: dict, client: httpx.Client) -> Iterator[Item]:
        h = {"Authorization": f"Bearer {self.access_token(creds)}"}
        files = _get(client, self.API, headers=h, params={"pageSize": 200, "q": "trashed=false",
                     "fields": "files(id,name,mimeType,webViewLink,size)"}).json()["files"]
        for f in files:
            if f["mimeType"] in self.EXPORT:
                mime, ext = self.EXPORT[f["mimeType"]]
                data = _get(client, f"{self.API}/{f['id']}/export", headers=h, params={"mimeType": mime}).content
            elif f["name"].lower().endswith(tuple(parsers.SUPPORTED - parsers.MEDIA - parsers.IMAGE)):
                ext = "." + f["name"].rsplit(".", 1)[-1]
                data = _get(client, f"{self.API}/{f['id']}", headers=h, params={"alt": "media"}).content
            else:
                continue
            try:
                yield Item(f"gdrive:{f['id']}", f["name"], parsers.parse(f"x{ext}", data), f.get("webViewLink", ""), "file", f["mimeType"])
            except parsers.ParseError as e:
                log.info("skipping %s: %s", f["name"], e)


class Gmail(_Google):
    kind, label, full_listing = "gmail", "Gmail", False
    API = "https://gmail.googleapis.com/gmail/v1/users/me"

    @staticmethod
    def _body(part: dict) -> str:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", "replace")
        return "\n".join(Gmail._body(p) for p in part.get("parts", []))

    def fetch(self, config: dict, creds: dict, client: httpx.Client) -> Iterator[Item]:
        h = {"Authorization": f"Bearer {self.access_token(creds)}"}
        rows = _get(client, f"{self.API}/messages", headers=h, params={"maxResults": 100, "q": config.get("query", "newer_than:30d")}).json()
        for row in rows.get("messages", []):
            msg = _get(client, f"{self.API}/messages/{row['id']}", headers=h, params={"format": "full"}).json()
            head = {x["name"].lower(): x["value"] for x in msg["payload"].get("headers", [])}
            text = f"From: {head.get('from', '')}\nDate: {head.get('date', '')}\n\n{self._body(msg['payload'])}"
            yield Item(f"gmail:{row['id']}", head.get("subject", "(no subject)"), text, "", "email", "text/plain")

    def create_draft(self, creds: dict, payload: dict) -> str:
        """Creates a draft only: nothing is sent, the user sends it from Gmail."""
        raw = f"To: {payload['to']}\r\nSubject: {payload['subject']}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{payload['body']}"
        r = httpx.post(f"{self.API}/drafts", headers={"Authorization": f"Bearer {self.access_token(creds)}"}, timeout=TIMEOUT,
                       json={"message": {"raw": base64.urlsafe_b64encode(raw.encode()).decode()}})
        r.raise_for_status()
        return f"Draft {r.json()['id']} created in Gmail"


class Postgres:
    """Indexes rows of named tables through a read-only session. Admin-only, see the API layer."""
    kind, label, full_listing = "postgres", "PostgreSQL", True
    fields = [{"name": "tables", "label": "Tables (comma separated)", "secret": False},
              {"name": "dsn", "label": "Connection string (use a read-only role)", "secret": True}]

    def fetch(self, config: dict, creds: dict, client: httpx.Client) -> Iterator[Item]:
        import psycopg
        from psycopg import sql

        with psycopg.connect(creds["dsn"], options="-c default_transaction_read_only=on", connect_timeout=10) as conn:
            for table in filter(None, (t.strip() for t in config.get("tables", "").split(","))):
                ident = sql.SQL(".").join(map(sql.Identifier, table.split(".")))
                cur = conn.execute(sql.SQL("SELECT * FROM {} LIMIT 1000").format(ident))
                cols = [c.name for c in cur.description]
                text = "\n".join("; ".join(f"{k}: {v}" for k, v in zip(cols, row) if v is not None) for row in cur)
                yield Item(f"postgres:{table}", f"Table {table}", f"Columns: {', '.join(cols)}\n{text}", "", "database", "text/plain")


REGISTRY = {c.kind: c for c in (Website(), GitHub(), Slack(), GoogleDrive(), Gmail(), Postgres())}


def requirements(kind: str) -> list[str]:
    """Server-side configuration that is still missing for this connector type."""
    s = settings()
    if kind in GOOGLE_SCOPES and not (s.google_client_id and s.google_client_secret):
        return ["NEXUS_GOOGLE_CLIENT_ID", "NEXUS_GOOGLE_CLIENT_SECRET"]
    return []


def sync(db: Session, conn: Connector) -> dict:
    impl = REGISTRY[conn.kind]
    conn.status, conn.last_error = "syncing", ""
    db.commit()
    stats = {"created": 0, "updated": 0, "unchanged": 0, "removed": 0}
    changed: list[Document] = []
    seen: set[str] = set()
    try:
        creds = decrypt(conn.secret)
        with httpx.Client(timeout=TIMEOUT) as client:
            for i, item in enumerate(impl.fetch(conn.config, creds, client)):
                if i >= MAX_ITEMS:
                    break
                doc, state = ingest.upsert(db, conn.workspace_id, conn.owner_id, item, visibility=conn.visibility, connector_id=conn.id)
                seen.add(doc.source_key)
                stats[state] += 1
                if state != "unchanged":
                    changed.append(doc)
        if impl.full_listing:
            for doc in db.query(Document).filter_by(connector_id=conn.id).all():
                if doc.source_key not in seen:
                    ingest.remove(db, doc)
                    stats["removed"] += 1
        conn.status = "idle"
    except NeedsAuth as e:
        conn.status, conn.last_error = "needs_auth", str(e)
    except Exception as e:
        log.exception("sync failed for connector %s", conn.id)
        conn.status, conn.last_error = "error", f"{type(e).__name__}: {e}"[:500]
    conn.last_synced_at = now()
    db.commit()
    if conn.status == "idle":
        proactive.analyze(db, conn, changed, stats)
    return stats


def due(db: Session) -> list[Connector]:
    cutoff = now() - timedelta(minutes=settings().sync_interval_minutes)
    return [c for c in db.query(Connector).filter(Connector.status.in_(["idle", "error"])).all()
            if not c.last_synced_at or c.last_synced_at < cutoff]


def save_secret(conn: Connector, creds: dict) -> None:
    conn.secret = encrypt(creds)
