import os
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfFileSystem

BUCKET = os.getenv(
    "HF_BUCKET",
    "buckets/Chatpataprani/Telegram-DB-bucket",
)

HF_TOKEN = os.getenv("HF_TOKEN") or None
DEVELOPER = "chatpataprani"

# Remote Hugging Face Parquet is I/O-bound.
# Increase/decrease this if your Vercel instance has resource limits.
MAX_WORKERS = int(os.getenv("LOOKUP_WORKERS", "12"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "512"))

app = FastAPI(
    title="Telegram Demo Lookup API",
    version="1.1",
    description="Telegram ID lookup API — Developer: chatpataprani",
)

fs = HfFileSystem(token=HF_TOKEN)

DATA_FILES = sorted(fs.glob(f"{BUCKET}/data_*.parquet"))

if not DATA_FILES:
    raise RuntimeError(
        "No data_*.parquet files were found in the Hugging Face bucket."
    )

# Build the dataset once per warm Vercel instance.
dataset = ds.dataset(
    DATA_FILES,
    filesystem=fs,
    format="parquet",
)

FRAGMENTS = list(dataset.get_fragments())


def _clean(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"", "null", "none", "nan"}:
        return None

    return value


def _scan_fragment(fragment, filter_expr, stop_event):
    """Search one remote Parquet fragment."""
    if stop_event.is_set():
        return None

    try:
        scanner = fragment.scanner(
            filter=filter_expr,
            columns=["account_id", "phone", "email"],
            batch_size=4096,
            use_threads=False,
        )

        for batch in scanner.to_batches():
            if stop_event.is_set():
                return None

            for row in batch.to_pylist():
                number = _clean(row.get("phone"))
                email = _clean(row.get("email"))

                if number or email:
                    stop_event.set()
                    return {
                        "number": number,
                        "email": email,
                    }

    except Exception:
        return None

    return None


@lru_cache(maxsize=CACHE_SIZE)
def _lookup_cached(telegram_id):
    """Lookup with a warm-process cache for repeated IDs."""
    from threading import Event

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
                # Do not wait for every remote scan before returning the
                # HTTP response. Already-running workers stop at their
                # next check; queued work is cancelled.
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
        raise HTTPException(
            status_code=400,
            detail="Telegram ID is required.",
        )

    started = time.perf_counter()
    found = _lookup_cached(telegram_id)

    if not found:
        return {
            "message": "No data found",
            "developer": DEVELOPER,
        }

    response = {"developer": DEVELOPER}

    if found.get("number"):
        response["number"] = found["number"]

    if found.get("email"):
        response["email"] = found["email"]

    response["lookup_ms"] = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    return response


@app.get("/")
def root():
    return {
        "status": "online",
        "developer": DEVELOPER,
        "endpoint": "/telegram?id=<telegram_id>",
        "backend": "PyArrow + HuggingFace HfFileSystem",
        "data_files": len(DATA_FILES),
        "lookup_workers": MAX_WORKERS,
        "cache_size": CACHE_SIZE,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "developer": DEVELOPER,
        "data_files": len(DATA_FILES),
    }


@app.get("/telegram")
def telegram_lookup(id: str):
    return lookup_telegram_id(id)
