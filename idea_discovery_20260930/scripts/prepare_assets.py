"""Bounded public benchmark preparation. Run ONLY in a CPU sbatch job."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-benchmark-probe"})
            with urllib.request.urlopen(req, timeout=90) as response, path.with_suffix(path.suffix + ".part").open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            path.with_suffix(path.suffix + ".part").replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    report = {"training": False, "assets": {}, "errors": {}}

    try:
        folder = args.root / "intphys2"
        url = "https://huggingface.co/datasets/facebook/IntPhys2/resolve/main/Main/metadata.csv"
        download(url, folder / "metadata_all.csv")
        rows = list(csv.DictReader((folder / "metadata_all.csv").open()))
        scenes = defaultdict(list)
        for row in rows:
            scenes[row["SceneIndex"]].append(row)
        strata = defaultdict(list)
        for sid, bank in scenes.items():
            row = bank[0]
            strata[(row["condition"], row["Difficulty"], row["Camera"])].append(sid)
        selected = []
        # Label-independent deterministic subset: up to 3 complete scenes per stratum.
        for key, ids in sorted(strata.items()):
            ids.sort(key=lambda value: hashlib.sha256(("discovery-20260930-" + value).encode()).hexdigest())
            selected.extend(ids[:3])
        chosen = [row for sid in selected for row in scenes[sid]]
        with (folder / "metadata.csv").open("w") as out:
            writer = csv.DictWriter(out, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(chosen)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda row: download("https://huggingface.co/datasets/facebook/IntPhys2/resolve/main/Main/" + row["file_name"], folder / row["file_name"]), chosen))
        report["assets"]["intphys2"] = {"scenes": len(selected), "videos": len(chosen), "selection": "3 complete scenes/condition,difficulty,camera stratum; SHA256 order; no label selection"}
    except Exception as error:
        report["errors"]["intphys2"] = repr(error)

    try:
        archive = args.root / "tapvid_davis.zip"
        download("https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip", archive)
        folder = args.root / "tapvid_davis"
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)
        report["assets"]["tapvid_davis"] = {"archive_bytes": archive.stat().st_size, "files": [str(p.relative_to(folder)) for p in folder.rglob("*.pkl")]}
    except Exception as error:
        report["errors"]["tapvid_davis"] = repr(error)

    try:
        archive = args.root / "push_chair_datasets.zip"
        if not archive.exists():
            import gdown
            result = gdown.download("https://drive.google.com/uc?id=1UW2lxqQiu90NTDsPmqxx8wVZ7aSlp504", str(archive), quiet=True)
            if result is None:
                raise RuntimeError("Public Sentinel archive download did not return a path")
        folder = args.root / "sentinel_push_chair"
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(folder)
        report["assets"]["sentinel"] = {"archive_bytes": archive.stat().st_size, "files": [str(p.relative_to(folder)) for p in folder.rglob("*.pkl")]}
    except Exception as error:
        report["errors"]["sentinel"] = repr(error)

    # Small released tracker weight, no policy/model execution in preparation.
    try:
        download("https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth", args.root / "scaled_online.pth")
        report["assets"]["cotracker3"] = {"checkpoint_bytes": (args.root / "scaled_online.pth").stat().st_size}
    except Exception as error:
        report["errors"]["cotracker3"] = repr(error)
    (args.root / "assets.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
