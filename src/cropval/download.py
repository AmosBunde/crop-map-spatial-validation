"""Fetch the CV4A Kenya Crop Type dataset from Source Cooperative.

The bucket is public and S3-compatible, so plain HTTPS is enough: list the
prefix, then download each object. Downloads resume: a file whose size already
matches the listing is skipped, and each file is retried on network errors.
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

HOST = "https://data.source.coop/radiantearth"
PREFIX = "african-crops-kenya-02/"
RETRIES = 8


def list_objects() -> list[tuple[str, int]]:
    """Return (key, size) for every object under the dataset prefix."""
    out: list[tuple[str, int]] = []
    token = None
    while True:
        url = f"{HOST}/?list-type=2&prefix={PREFIX}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        with urllib.request.urlopen(url, timeout=60) as r:
            xml = r.read().decode()
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", xml)]
        m = re.search(r"<NextContinuationToken>([^<]+)<", xml)
        if not m:
            return out
        token = m.group(1)


def select(objects, tiles=None, dates=None):
    """Keep label/field rasters plus imagery for the chosen tiles and dates."""
    keep = []
    for key, size in objects:
        parts = key[len(PREFIX):].split("/")
        if parts[0] != "data":
            keep.append((key, size))  # documentation, FieldIds.csv
            continue
        tile = parts[1]
        if tiles and tile not in tiles:
            continue
        if len(parts) == 4 and dates and parts[2] not in dates:
            continue
        keep.append((key, size))
    return keep


def download(dest: Path, tiles=None, dates=None) -> int:
    objects = select(list_objects(), tiles, dates)
    total = sum(s for _, s in objects)
    fetched = 0
    with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
        for key, size in objects:
            path = dest / key[len(PREFIX):]
            if path.exists() and path.stat().st_size == size:
                bar.update(size)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            for attempt in range(1, RETRIES + 1):
                got = 0
                try:
                    with urllib.request.urlopen(f"{HOST}/{urllib.parse.quote(key)}", timeout=120) as r, open(tmp, "wb") as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                            got += len(chunk)
                            bar.update(len(chunk))
                    if got != size:
                        raise OSError(f"short read: {got} of {size} bytes")
                    break
                except (OSError, TimeoutError) as err:  # URLError is an OSError
                    bar.update(-got)
                    if attempt == RETRIES:
                        raise RuntimeError(f"giving up on {key} after {RETRIES} attempts") from err
                    time.sleep(min(60, 5 * attempt))
            tmp.rename(path)
            fetched += 1
    return fetched
