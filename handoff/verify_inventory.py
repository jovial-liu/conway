"""Check the release file inventory; no network, model, or statistics execution."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.repo_root.resolve()
    records = json.loads((Path(__file__).parent / "FILE_INDEX.json").read_text())["files"]
    failures = []
    for row in records:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append({"path": row["path"], "reason": "missing or outside checkout"})
            continue
        size = path.stat().st_size
        blob = hashlib.sha1(("blob %d\0" % size).encode())
        sha256 = hashlib.sha256()
        with path.open("rb") as handle:
            for data in iter(lambda: handle.read(1024 * 1024), b""):
                blob.update(data)
                sha256.update(data)
        if size != row["size_bytes"] or blob.hexdigest() != row["git_blob_sha1"]:
            failures.append({"path": row["path"], "reason": "size or Git blob SHA1 mismatch"})
        if row["sha256"] and sha256.hexdigest() != row["sha256"]:
            failures.append({"path": row["path"], "reason": "recorded SHA256 mismatch"})
    print(json.dumps({"checked_files": len(records), "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)

if __name__ == "__main__":
    main()
