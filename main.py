import json
import logging
import os
import time
from functools import lru_cache

import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import HfFileSystem

BUCKET = os.getenv(
    "HF_BUCKET",
    "buckets/Chatpataprani/Telegram-Cleaned-DB-bucket",
)
HF_TOKEN = os.getenv("HF_TOKEN") or None
DEVELOPER = "chatpataprani"
logger = logging.getLogger("telegramdb")
CACHE_SIZE = max(1, int(os.getenv("CACHE_SIZE", "1024")))
BATCH_SIZE = max(256, int(os.getenv("BATCH_SIZE", "4096")))
INDEX_DIR = os.getenv("INDEX_DIR", "telegram-index")
INDEX_SHARDS = max(1, int(os.getenv("INDEX_SHARDS", "1024")))
INDEX_MANIFEST = f"{BUCKET}/{INDEX_DIR}/manifest.json"

app = FastAPI(
    title="Telegram ID Lookup API",
    version="4.0",
    description=(
        "Indexed Telegram account ID lookup using partitioned Parquet "
        "shards on Hugging Face storage."
    ),
)

fs = HfFileSystem(token=HF_TOKEN)

# The index manifest is published last by the build workflow. Its presence
# means every shard listed in it has already been uploaded successfully.
try:
    with fs.open(INDEX_MANIFEST, "rb") as manifest_file:
        INDEX_INFO = json.loads(manifest_file.read().decode("utf-8"))
    INDEX_READY = (
        INDEX_INFO.get("version") == 1
        and int(INDEX_INFO.get("shard_count", 0)) == INDEX_SHARDS
        and isinstance(INDEX_INFO.get("files"), dict)
    )
except FileNotFoundError:
    INDEX_INFO = {}
    INDEX_READY = False
except Exception:
    logger.exception("Could not read the index manifest; using legacy scan.")
    INDEX_INFO = {}
    INDEX_READY = False

DATA_FILES = []
FRAGMENTS = []
dataset = None

if not INDEX_READY:
    DATA_FILES = sorted(fs.glob(f"{BUCKET}/data_*.parquet"))
    if not DATA_FILES:
        raise RuntimeError(
            f"No index or data_*.parquet files found in {BUCKET}. "
            "Check HF_BUCKET and Hugging Face permissions."
        )

    dataset = ds.dataset(DATA_FILES, filesystem=fs, format="parquet")
    required_columns = {"account_id", "phone"}
    missing_columns = required_columns.difference(dataset.schema.names)
    if missing_columns:
        raise RuntimeError(
            "Parquet dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    FRAGMENTS = list(dataset.get_fragments())
    for fragment in FRAGMENTS:
        fragment.ensure_complete_metadata()


def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "null", "none", "nan", "n/a"}:
        return None
    return value


def _scan_index(telegram_id):
    # account_id values are numeric strings in this dataset. The builder uses
    # the same integer modulo operation to place every ID in one shard.
    shard = str(int(telegram_id) % INDEX_SHARDS)
    shard_files = INDEX_INFO["files"].get(shard)
    if not shard_files:
        return None

    paths = [f"{BUCKET}/{path}" for path in shard_files]
    indexed_dataset = ds.dataset(paths, filesystem=fs, format="parquet")
    filter_expr = ds.field("account_id") == telegram_id
    scanner = indexed_dataset.scanner(
        filter=filter_expr,
        columns=["account_id", "phone"],
        batch_size=BATCH_SIZE,
        use_threads=True,
    )

    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            account_id = _clean(row.get("account_id"))
            if account_id == telegram_id:
                return {
                    "account_id": account_id,
                    "phone": _clean(row.get("phone")),
                }
    return None


def _scan_legacy(telegram_id):
    filter_expr = ds.field("account_id") == telegram_id
    candidate_files = []
    for fragment in FRAGMENTS:
        if fragment.split_by_row_group(filter=filter_expr):
            candidate_files.append(fragment)

    for fragment in candidate_files:
        scanner = fragment.scanner(
            filter=filter_expr,
            columns=["account_id", "phone"],
            batch_size=BATCH_SIZE,
            use_threads=True,
        )
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                account_id = _clean(row.get("account_id"))
                if account_id == telegram_id:
                    return {
                        "account_id": account_id,
                        "phone": _clean(row.get("phone")),
                    }
    return None


@lru_cache(maxsize=CACHE_SIZE)
def _lookup_cached(telegram_id):
    if INDEX_READY:
        return _scan_index(telegram_id)
    return _scan_legacy(telegram_id)


def lookup_telegram_id(telegram_id):
    telegram_id = telegram_id.strip()
    if not telegram_id:
        raise HTTPException(status_code=400, detail="Telegram ID is required.")
    if not telegram_id.isdigit():
        raise HTTPException(status_code=400, detail="Telegram ID must contain digits only.")

    started = time.perf_counter()
    try:
        found = _lookup_cached(telegram_id)
    except Exception as exc:
        logger.exception("Lookup failed (exception_type=%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Database lookup temporarily failed. Please retry.",
        ) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if not found:
        return {
            "status": "not_found",
            "account_id": telegram_id,
            "developer": DEVELOPER,
            "lookup_ms": elapsed_ms,
        }

    return {
        "status": "found",
        "account_id": found["account_id"],
        "phone": found["phone"],
        "developer": DEVELOPER,
        "lookup_ms": elapsed_ms,
    }


@app.get("/")
def root():
    return {
        "status": "online",
        "developer": DEVELOPER,
        "endpoint": "/telegram?id=<telegram_id>",
        "backend": "FastAPI + indexed Parquet shards + Hugging Face",
        "index_ready": INDEX_READY,
        "data_files": len(DATA_FILES),
        "index_shards": INDEX_SHARDS if INDEX_READY else 0,
        "cache_size": CACHE_SIZE,
        "phone_output": "full",
        "api_version": app.version,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_ready": INDEX_READY,
        "data_files": len(DATA_FILES),
    }


@app.get("/stats")
def stats():
    cache = _lookup_cached.cache_info()
    return {
        "index_ready": INDEX_READY,
        "index_shards": INDEX_SHARDS if INDEX_READY else 0,
        "data_files": len(DATA_FILES),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_entries": cache.currsize,
        "cache_max_size": cache.maxsize,
    }


@app.get("/telegram")
def telegram_lookup(id: str = Query(..., min_length=1, max_length=32)):
    return lookup_telegram_id(id)
