from loguru import logger

from config import CONFIG
from dochub.utils.cache import redis_client

REDIS_KEY_PREFIX = CONFIG["redis"]["prefix"]
DOC_PARSE_PROGRESS = f"{REDIS_KEY_PREFIX}:doc:parse:progress"


def report_progress(doc_id: str, percent: float):
    redis_client.set(f"{DOC_PARSE_PROGRESS}:{doc_id}", percent, ex=3600 * 24)
    logger.info(f"Report document parsing progress: {percent:2f} ({doc_id})")


def get_progress(doc_id: str) -> float:
    return float(redis_client.get(f"{DOC_PARSE_PROGRESS}:{doc_id}"))
