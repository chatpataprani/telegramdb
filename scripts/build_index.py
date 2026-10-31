#!/usr/bin/env python3
"""Build sorted, hash-partitioned Parquet lookup shards in the HF bucket."""

import json
import os
import shutil
from pathlib import Path

import duckdb
import pyarrow.dataset as ds
from huggingface_hub import HfFileSystem, batch_bucket_files

BUCKET_PATH = os.getenv(
    "HF_BUCKET",
    "buckets/Chatpataprani/Telegram-Cleaned-DB-bucket",
).rstrip("/")
BUCKET_ID = BUCKET_PATH.removeprefix("buckets/")
TOKEN = os.environ["HF_TOKEN"]
SHARD_COUNT = int(os.getenv("INDEX_SHARDS", "1024"))
OUTPUT_DIR = Path("/tmp/telegramdb-index")
TEMP_DIR = Path("/tmp/telegramdb-duckdb")
REMOTE_PREFIX = "telegram-index"

if SHARD_COUNT < 1:
    raise ValueError("INDEX_SHARDS must be at least 1")

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
shutil.rmtree(TEMP_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True)
TEMP_DIR.mkdir(parents=True)
(TEMP_DIR / "spill").mkdir(parents=True)

print(f"Listing source Parquet files in {BUCKET_PATH}", flush=True)
fs = HfFileSystem(token=TOKEN)
source_files = sorted(fs.glob(f"{BUCKET_PATH}/data_*.parquet"))
if not source_files:
    raise RuntimeError("No source data_*.parquet files found.")

print(f"Found {len(source_files)} source files; preparing Arrow scanner.", flush=True)
source = ds.dataset(source_files, filesystem=fs, format="parquet")
required = {"account_id", "phone"}
missing = required.difference(source.schema.names)
if missing:
    raise RuntimeError(f"Source is missing columns: {sorted(missing)}")

scanner = source.scanner(
    columns=["account_id", "phone"],
    batch_size=131072,
    use_threads=True,
)
reader = scanner.to_reader()

con = duckdb.connect(str(TEMP_DIR / "build.duckdb"))
con.execute("SET memory_limit='5GB'")
con.execute(f"SET temp_directory='{(TEMP_DIR / 'spill').as_posix()}'")
con.execute("SET threads=2")
con.register("source_rows", reader)

# Sorting by shard and account_id makes each shard's Parquet row-group
# min/max statistics useful for exact account_id filters.
copy_sql = f"""
COPY (
    SELECT
        CAST(account_id AS VARCHAR) AS account_id,
        CAST(phone AS VARCHAR) AS phone,
        CAST(TRY_CAST(account_id AS UBIGINT) % {SHARD_COUNT} AS INTEGER) AS shard
    FROM source_rows
    WHERE account_id IS NOT NULL
      AND TRY_CAST(account_id AS UBIGINT) IS NOT NULL
    ORDER BY shard, account_id
)
TO '{(OUTPUT_DIR / "shards").as_posix()}'
(FORMAT PARQUET, PARTITION_BY (shard), ROW_GROUP_SIZE 4096, COMPRESSION ZSTD)
"""
print("Scanning, partitioning and sorting all account rows. This can take a while.", flush=True)
con.execute(copy_sql)
con.close()

# The manifest is uploaded only after every Parquet shard has succeeded.
files_by_shard = {}
uploads = []
for path in sorted((OUTPUT_DIR / "shards").rglob("*.parquet")):
    shard_dir = next(part for part in path.parts if part.startswith("shard="))
    shard = shard_dir.split("=", 1)[1]
    relative = path.relative_to(OUTPUT_DIR / "shards").as_posix()
    remote = f"{REMOTE_PREFIX}/{relative}"
    files_by_shard.setdefault(shard, []).append(remote)
    uploads.append((str(path), remote))

if not uploads:
    raise RuntimeError("Index build produced no Parquet shard files.")

print(f"Built {len(uploads)} Parquet files across {len(files_by_shard)} shards.", flush=True)
for offset in range(0, len(uploads), 50):
    batch = uploads[offset : offset + 50]
    batch_bucket_files(BUCKET_ID, add=batch)
    print(f"Uploaded {min(offset + len(batch), len(uploads))}/{len(uploads)} files.", flush=True)

manifest = {
    "version": 1,
    "shard_count": SHARD_COUNT,
    "files": files_by_shard,
}
manifest_path = OUTPUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
batch_bucket_files(
    BUCKET_ID,
    add=[(str(manifest_path), f"{REMOTE_PREFIX}/manifest.json")],
)
print("Index manifest uploaded. Indexed lookup is now ready.", flush=True)
