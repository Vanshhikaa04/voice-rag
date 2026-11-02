"""Small key/value cache: Redis when NEXUS_REDIS_URL is set, otherwise an in-process LRU."""
import logging
import time
from collections import OrderedDict
from functools import lru_cache

from .config import settings

log = logging.getLogger("nexus.cache")
_local: OrderedDict[str, tuple[float, str]] = OrderedDict()
MAX_LOCAL = 5000


@lru_cache
def _redis():
    url = settings().redis_url
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, decode_responses=True, socket_timeout=1)


def get(key: str) -> str | None:
    r = _redis()
    if r:
        try:
            return r.get(key)
        except Exception as e:
            log.warning("redis get failed: %s", e)
            return None
    item = _local.get(key)
    if not item or item[0] < time.time():
        _local.pop(key, None)
        return None
    _local.move_to_end(key)
    return item[1]


def set(key: str, value: str, ttl: int = 3600) -> None:
    r = _redis()
    if r:
        try:
            r.set(key, value, ex=ttl)
        except Exception as e:
            log.warning("redis set failed: %s", e)
        return
    _local[key] = (time.time() + ttl, value)
    _local.move_to_end(key)
    while len(_local) > MAX_LOCAL:
        _local.popitem(last=False)
