"""Guard behavior in small independent Git repositories."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('public_guard', Path(__file__).parents[1] / 'tools/check_public.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_paths_and_content():
    assert guard.validate('docs/tutorial.md', b'example') is None
    assert guard.validate('notes.md', b'example')
    assert guard.validate('docs/private/plan.md', b'example')
    assert guard.validate('docs/main.tex', b'example')
    assert guard.validate('../README.md', b'example')
    assert guard.validate('docs/example.md', b'/' + b'Users/someone/data')


def test_history_keeps_deleted_private_files_visible_to_guard(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    for key, value in [('user.name', 'Test'), ('user.email', 'test@example.org')]:
        subprocess.run(['git', 'config', key, value], cwd=tmp_path, check=True)
    target = tmp_path / 'personal.md'
    target.write_text('private plan', encoding='utf-8')
    subprocess.run(['git', 'add', 'personal.md'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '-qm', 'Initial'], cwd=tmp_path, check=True)
    target.unlink()
    subprocess.run(['git', 'add', '-u'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '-qm', 'Remove'], cwd=tmp_path, check=True)
    assert not guard.check(tmp_path)
    assert any('history' in failure for failure in guard.check(tmp_path, history=True))


def test_missing_git_is_not_a_success(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', '')
    with pytest.raises(FileNotFoundError):
        guard.check(tmp_path)
