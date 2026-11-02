"""All settings come from the environment (or .env). Secrets never live in code."""
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    env: str = "development"                     # NEXUS_ENV=production enforces a real secret
    secret_key: str = ""
    data_dir: Path = ROOT / "data" / "nexus"
    database_url: str = ""                       # default: sqlite file in data_dir
    qdrant_url: str = ""                         # default: embedded Qdrant in data_dir
    qdrant_api_key: str = ""
    redis_url: str = ""                          # default: in-process cache
    cors_origins: str = "http://localhost:3000"
    token_ttl_minutes: int = 60 * 12
    allow_signup: bool = True
    cookie_secure: bool = False

    # model gateway
    llm_provider: str = ""                       # openai | anthropic; default: first with a key
    embedding_provider: str = ""                 # openai | local | hashing; default: auto
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = ""                     # optional cross-encoder name
    llm_timeout: float = 45.0
    price_overrides: str = ""                    # "model=input:output,..." USD per million tokens

    # retrieval
    top_k: int = 6
    candidates: int = 24
    min_relevance: float = 0.30                  # cosine floor for model embeddings; see README calibration
    max_upload_mb: int = 50

    # background
    sync_interval_minutes: int = 30
    scheduler_enabled: bool = True

    # STT / TTS
    voice_provider: str = ""                     # sarvam | elevenlabs | openai | local; default: auto
    whisper_model: str = "base"

    # connector app credentials (OAuth)
    google_client_id: str = ""
    google_client_secret: str = ""
    public_url: str = "http://localhost:3000"


class ExternalKeys(BaseSettings):
    """Conventional provider variable names, kept unprefixed."""
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    voicerag_model: str = "claude-haiku-5-5"
    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""


class _Prefixed(Settings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_prefix="NEXUS_")


@lru_cache
def settings() -> Settings:
    s = _Prefixed()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    if not s.secret_key:
        if s.env == "production":
            raise RuntimeError("NEXUS_SECRET_KEY must be set in production")
        path = s.data_dir / ".dev_secret"              # stable across restarts, never committed
        if not path.exists():
            path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
        s.secret_key = path.read_text(encoding="utf-8").strip()
    if not s.database_url:
        s.database_url = f"sqlite:///{(s.data_dir / 'nexus.db').as_posix()}"
    return s


@lru_cache
def keys() -> ExternalKeys:
    return ExternalKeys()
