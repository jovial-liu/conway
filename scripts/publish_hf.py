"""Publish tracked Conway source files without rewriting history or uploading experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

ROOT_FILES = {'README.md', 'README.zh-CN.md', 'LICENSE', 'CHANGELOG.md', 'pyproject.toml', 'MANIFEST.in'}
SOURCE_ROOTS = {'src', 'tests', 'docs', 'examples', 'scripts'}
SOURCE_SUFFIXES = {'.py', '.md', '.toml', '.yaml', '.yml'}


def source_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root)
    paths = []
    total = 0
    for name in raw.decode('utf-8').split('\0'):
        if not name:
            continue
        relative = Path(name)
        if relative.as_posix() not in ROOT_FILES and not (relative.parts[0] in SOURCE_ROOTS and relative.suffix in SOURCE_SUFFIXES):
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Invalid source file: {name}')
        size = path.stat().st_size
        if size > 2000000:
            raise ValueError(f'Source file exceeds 2 MB: {name}')
        total += size
        paths.append(relative)
    if total > 20000000 or not paths:
        raise ValueError('Empty or oversized source snapshot')
    return sorted(paths)


def stage_snapshot(root: Path, target: Path) -> dict:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    manifest = {'schema_version': 1, 'github_commit': commit, 'publication': 'source snapshot; not a Git-history mirror', 'files': {}}
    for relative in source_files(root):
        path = root / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest['files'][relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (target / '_source_commit.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='jnjnkj/conway')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix='conway-publish-') as directory:
        staged = Path(directory)
        manifest = stage_snapshot(root, staged)
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            return 0
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise RuntimeError('Missing HF_TOKEN repository secret')
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token)
        commit = api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=str(staged),
                                   commit_message=f"Publish Conway source {manifest['github_commit'][:12]}")
        # Read the immutable publication marker back, not merely the workflow exit status.
        marker = hf_hub_download(args.repo, '_source_commit.json', repo_type='model', revision=commit.oid, token=token)
        verified = json.loads(Path(marker).read_text(encoding='utf-8'))
        if verified != manifest:
            raise RuntimeError('Published source manifest did not match')
        print(f"Verified {len(manifest['files'])} source files at HF commit {commit.oid}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
