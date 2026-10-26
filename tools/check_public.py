"""Check repository content against the documented publication allowlist."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT_FILES = {'README.md', 'LICENSE', 'NOTICE', 'CONTRIBUTING.md', 'CHANGELOG.md',
              'CITATION.cff', 'mkdocs.yml', 'pyproject.toml', 'MANIFEST.in', '.gitignore'}
ROOT_DIRS = {'.github', 'docs', 'src', 'tests', 'tools'}
FORBIDDEN = {'.tex', '.aux', '.log', '.npz', '.rds', '.rdata', '.zip', '.key', '.pem'}
PATTERNS = [rb'/' + rb'Users/[^/\s]+/', rb'PRIVATE ' + rb'KEY-----', rb'gh[pousr]_[A-Za-z0-9]{30,}',
            rb'github_pat_[A-Za-z0-9_]{40,}']


def validate(path, content):
    parts = PurePosixPath(path).parts
    if not parts or '..' in parts or path.startswith('/'):
        return 'invalid path'
    if not (path in ROOT_FILES or parts[0] in ROOT_DIRS):
        return 'outside publication allowlist'
    if PurePosixPath(path).suffix.lower() in FORBIDDEN:
        return 'private or generated source format'
    if any(part.lower() in {'private', 'author', '.venv', 'roadmap'} for part in parts):
        return 'private directory'
    if b'\0' not in content[:8192] and any(re.search(pattern, content) for pattern in PATTERNS):
        return 'local path or credential-like content'
    return None


def check(root, history=False):
    paths = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root
    ).decode().split('\0')
    failures = []
    for path in filter(None, paths):
        target = root / path
        if target.is_symlink():
            failures.append(f'{path}: symlinks are not allowed')
        elif target.is_file():
            problem = validate(path, target.read_bytes())
            if problem:
                failures.append(f'{path}: {problem}')
    if history:
        objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=root).decode()
        for line in objects.splitlines():
            fields = line.split(' ', 1)
            if len(fields) != 2:
                continue
            oid, path = fields
            kind = subprocess.check_output(['git', 'cat-file', '-t', oid], cwd=root).strip()
            if kind != b'blob':
                continue
            content = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=root)
            problem = validate(path, content)
            if problem:
                failures.append(f'history {oid[:12]} {path}: {problem}')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = check(root, options.history)
    if failures:
        raise SystemExit('\n'.join(failures))
    print('Publication allowlist and selected content checks passed.')


if __name__ == '__main__':
    main()
