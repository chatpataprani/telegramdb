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

app = FastAPI(
    title="Telegram ID Lookup API",
    version="3.1",
    description=(
        "Fast Telegram account ID lookup using Parquet row-group statistics "
        "and Hugging Face storage."
    ),
)

# Initialise the remote dataset once per process.
fs = HfFileSystem(token=HF_TOKEN)
DATA_FILES = sorted(fs.glob(f"{BUCKET}/data_*.parquet"))

if not DATA_FILES:
    raise RuntimeError(
        f"No data_*.parquet files found in {BUCKET}. "
        "Set HF_BUCKET to the correct Hugging Face bucket path."
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

# Load each file's Parquet metadata once at startup. The metadata contains
# account_id min/max statistics for row-group pruning.
for fragment in FRAGMENTS:
    fragment.ensure_complete_metadata()


def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "null", "none", "nan", "n/a"}:
        return None
    return value


def _scan_candidate_fragments(telegram_id):
    filter_expr = ds.field("account_id") == telegram_id

    # First use row-group statistics only to identify which files can match.
    # Do not create a separate remote scanner for every matching row group:
    # that causes many repeated HTTP range requests to Hugging Face.
    candidate_files = []
    for fragment in FRAGMENTS:
        if next(fragment.split_by_row_group(filter=filter_expr), None) is not None:
            candidate_files.append(fragment)

    # Scan each candidate Parquet file once. PyArrow applies the exact filter
    # and row-group pruning internally while it reads the selected columns.
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
    return _scan_candidate_fragments(telegram_id)


def lookup_telegram_id(telegram_id):
    telegram_id = telegram_id.strip()
    if not telegram_id:
        raise HTTPException(status_code=400, detail="Telegram ID is required.")

    started = time.perf_counter()

    try:
        found = _lookup_cached(telegram_id)
    except Exception as exc:
        # Keep the client response generic, but record the real cause in Vercel logs.
        logger.exception("Parquet lookup failed (exception_type=%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Database scan temporarily failed. Please retry.",
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
        "backend": "FastAPI + PyArrow row-group pruning + Hugging Face",
        "data_files": len(DATA_FILES),
        "cache_size": CACHE_SIZE,
        "phone_output": "full",
        "api_version": app.version,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "developer": DEVELOPER,
        "data_files": len(DATA_FILES),
    }


@app.get("/stats")
def stats():
    cache = _lookup_cached.cache_info()
    return {
        "data_files": len(DATA_FILES),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_entries": cache.currsize,
        "cache_max_size": cache.maxsize,
    }


@app.get("/telegram")
def telegram_lookup(id: str = Query(..., min_length=1, max_length=32)):
    return lookup_telegram_id(id)
