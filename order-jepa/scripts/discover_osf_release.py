#!/usr/bin/env python3
"""List the authors' OSF release tree without downloading large artifacts."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path


def with_view_token(url: str, token: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query["view_only"] = [token]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def get_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        with_view_token(url, token), headers={"User-Agent": "order-jepa-audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="bmw48")
    parser.add_argument("--view-token", default="a56a296ce3b24cceaf408383a175ce28")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("OSF release discovery must run on a compute node")

    queue = deque([f"https://api.osf.io/v2/nodes/{args.node}/files/osfstorage/"])
    seen: set[str] = set()
    files = []
    while queue:
        page_url = queue.popleft()
        if page_url in seen:
            continue
        seen.add(page_url)
        page = get_json(page_url, args.view_token)
        for item in page.get("data", []):
            attributes = item.get("attributes", {})
            kind = attributes.get("kind")
            entry = {
                "id": item.get("id"),
                "kind": kind,
                "name": attributes.get("name"),
                "materialized_path": attributes.get("materialized_path"),
                "size": attributes.get("size"),
                "date_modified": attributes.get("date_modified"),
                "extra": attributes.get("extra"),
                "download": item.get("links", {}).get("download"),
            }
            if kind == "file":
                files.append(entry)
            elif kind == "folder":
                related = (
                    item.get("relationships", {})
                    .get("files", {})
                    .get("links", {})
                    .get("related", {})
                    .get("href")
                )
                if related:
                    queue.append(related)
        next_url = page.get("links", {}).get("next")
        if next_url:
            queue.append(next_url)

    files.sort(key=lambda row: row["materialized_path"] or "")
    payload = {
        "schema": "order-jepa-osf-release-tree-v1",
        "node": args.node,
        "view_token": args.view_token,
        "files": files,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    checkpoint_like = [
        row
        for row in files
        if "checkpoint" in (row["materialized_path"] or "").lower()
        or "pusht" in (row["materialized_path"] or "").lower()
    ]
    print(json.dumps({"files": len(files), "checkpoint_or_pusht": checkpoint_like}, indent=2))


if __name__ == "__main__":
    main()

