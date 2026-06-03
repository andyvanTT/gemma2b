#!/usr/bin/env python3
"""Mirror the public pi05_base orbax checkpoint from GCS into ./weights.

Uses only the stdlib so it can run with any Python before the venv exists.
The openpi-assets bucket is public, so no auth is needed.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import urllib.request

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/params/"
# strip this leading path so files land at ./weights/pi05_base/params/...
STRIP = "checkpoints/"
DEST_ROOT = pathlib.Path(__file__).parent / "weights"

LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
OBJ_URL = "https://storage.googleapis.com/{bucket}/{name}"


def list_objects():
    items = []
    page_token = None
    while True:
        params = f"?prefix={PREFIX}&fields=items(name,size),nextPageToken&maxResults=1000"
        if page_token:
            params += f"&pageToken={page_token}"
        url = LIST_URL.format(bucket=BUCKET) + params
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        items.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    # skip "directory" placeholder objects ending in /
    return [it for it in items if not it["name"].endswith("/")]


def download(obj):
    name = obj["name"]
    rel = name[len(STRIP):]  # -> pi05_base/params/...
    dest = DEST_ROOT / rel
    size = int(obj.get("size", 0))
    if dest.exists() and dest.stat().st_size == size:
        return (name, size, "cached")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = OBJ_URL.format(bucket=BUCKET, name=urllib.request.quote(name))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return (name, size, "downloaded")


def main():
    objs = list_objects()
    total = sum(int(o.get("size", 0)) for o in objs)
    print(f"Found {len(objs)} objects, {total/1e9:.2f} GB total", flush=True)
    done_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (name, size, status) in enumerate(ex.map(download, objs), 1):
            done_bytes += size
            print(f"[{i}/{len(objs)}] {status:10s} {size/1e6:9.2f} MB  "
                  f"({done_bytes/1e9:.2f}/{total/1e9:.2f} GB)  {name}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
