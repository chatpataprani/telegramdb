import os
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import HfFileSystem

BUCKET = os.getenv(
    "HF_BUCKET",
    "buckets/Chatpataprani/Telegram-Cleaned-DB-bucket",
)
HF_TOKEN = os.getenv("HF_TOKEN") or None
DEVELOPER = "chatpataprani"

MAX_WORKERS = max(1, int(os.getenv("LOOKUP_WORKERS", "8")))
CACHE_SIZE = max(1, int(os.getenv("CACHE_SIZE", "512")))

app = FastAPI(
    title="Telegram ID Lookup API",
    version="2.0",
    description="Looks up Telegram account IDs in the configured Parquet dataset.",
)

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


def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "null", "none", "nan", "n/a"}:
        return None
    return value


def _mask_phone(value):
    """Never return a complete phone number from this public lookup API."""
    value = _clean(value)
    if not value:
        return None

    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) <= 2:
        return "*" * len(digits)
    return "*" * (len(digits) - 2) + digits[-2:]


def _scan_fragment(fragment, filter_expr, stop_event):
    if stop_event.is_set():
        return None

    try:
        scanner = fragment.scanner(
            filter=filter_expr,
            columns=["account_id", "phone"],
            batch_size=4096,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            if stop_event.is_set():
                return None
            for row in batch.to_pylist():
                account_id = _clean(row.get("account_id"))
                phone = _mask_phone(row.get("phone"))
                if account_id:
                    stop_event.set()
                    return {"account_id": account_id, "phone": phone}
    except Exception:
        return None

    return None


@lru_cache(maxsize=CACHE_SIZE)
def _lookup_cached(telegram_id):
    stop_event = Event()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [
        executor.submit(
            _scan_fragment,
            fragment,
            ds.field("account_id") == telegram_id,
            stop_event,
        )
        for fragment in FRAGMENTS
    ]

    try:
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                for pending in futures:
                    pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return result

        executor.shutdown(wait=True)
        return None
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def lookup_telegram_id(telegram_id):
    telegram_id = telegram_id.strip()
    if not telegram_id:
        raise HTTPException(status_code=400, detail="Telegram ID is required.")

    started = time.perf_counter()
    found = _lookup_cached(telegram_id)

    if not found:
        return {
            "status": "not_found",
            "account_id": telegram_id,
            "developer": DEVELOPER,
        }

    return {
        "status": "found",
        "account_id": found["account_id"],
        "phone": found["phone"],
        "developer": DEVELOPER,
        "lookup_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/")
def root():
    return {
        "status": "online",
        "developer": DEVELOPER,
        "endpoint": "/telegram?id=<telegram_id>",
        "backend": "FastAPI + PyArrow + Hugging Face",
        "data_files": len(DATA_FILES),
        "lookup_workers": MAX_WORKERS,
        "cache_size": CACHE_SIZE,
        "phone_output": "masked",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "developer": DEVELOPER,
        "data_files": len(DATA_FILES),
    }


@app.get("/telegram")
def telegram_lookup(id: str = Query(..., min_length=1, max_length=32)):
    return lookup_telegram_id(id)
