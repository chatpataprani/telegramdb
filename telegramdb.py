import os
import time

import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfFileSystem

# Hugging Face bucket containing the synthetic/demo Telegram dataset
BUCKET = os.getenv(
    "HF_BUCKET",
    "buckets/Chatpataprani/Telegram-DB-bucket",
)

HF_TOKEN = os.getenv("HF_TOKEN") or None
DEVELOPER = "chatpataprani"

app = FastAPI(
    title="Telegram Demo Lookup API",
    version="1.0",
    description="Telegram ID lookup API using synthetic/demo data — Developer: chatpataprani",
)

# Connect to Hugging Face storage.
fs = HfFileSystem(token=HF_TOKEN)

# The bucket uses data_*.parquet files.
DATA_FILES = sorted(fs.glob(f"{BUCKET}/data_*.parquet"))

if not DATA_FILES:
    raise RuntimeError(
        "No data_*.parquet files were found in the Hugging Face bucket."
    )

# Build one Arrow dataset over all Parquet parts.
dataset = ds.dataset(
    DATA_FILES,
    filesystem=fs,
    format="parquet",
)


def lookup_telegram_id(telegram_id: str):
    """Find a Telegram account_id and return only number/email fields."""
    telegram_id = telegram_id.strip()

    if not telegram_id:
        raise HTTPException(
            status_code=400,
            detail="Telegram ID is required.",
        )

    # account_id is stored as a string in the supplied dataset schema.
    filter_expr = ds.field("account_id") == telegram_id

    started = time.perf_counter()

    scanner = dataset.scanner(
        filter=filter_expr,
        columns=["account_id", "phone", "email"],
        batch_size=4096,
        use_threads=True,
    )

    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            number = row.get("phone")
            email = row.get("email")

            # Treat empty/null values as unavailable.
            if number is not None:
                number = str(number).strip() or None

            if email is not None:
                email = str(email).strip() or None

            if number or email:
                result = {
                    "developer": DEVELOPER,
                }

                if number:
                    result["number"] = number

                if email:
                    result["email"] = email

                return result

    return {
        "message": "No data found",
        "developer": DEVELOPER,
    }


@app.get("/")
def root():
    return {
        "status": "online",
        "developer": DEVELOPER,
        "endpoint": "/telegram?id=<telegram_id>",
        "backend": "PyArrow + HuggingFace HfFileSystem",
        "data_files": len(DATA_FILES),
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
