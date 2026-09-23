"""Tests for v0.2.1: `doctor --check-install` and PATH warning."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest


# ---------- _check_ssmforge_on_path ----------

def test_check_ssmforge_on_path_returns_tuple():
    """`_check_ssmforge_on_path` returns a (bool, optional_str) tuple."""
    from ssmforge.cli import _check_ssmforge_on_path
    on_path, fix = _check_ssmforge_on_path()
    assert isinstance(on_path, bool)
    if fix is not None:
        assert isinstance(fix, str)


def test_check_ssmforge_on_path_finds_off_path_install(monkeypatch):
    """When shutil.which returns None but a known Scripts dir exists, we get a fix hint."""
    from ssmforge.cli import _check_ssmforge_on_path
    from ssmforge import cli as cli_mod
    import shutil

    # Pretend shutil.which can't find ssmforge
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "ssmforge" else f"/usr/bin/{cmd}")

    # Pretend we're on Windows with APPDATA pointing at a dir that has ssmforge.exe
    fake_appdata = "/tmp/fake_appdata"
    fake_scripts = os.path.join(fake_appdata, "Python", f"Python{sys.version_info.major}{sys.version_info.minor}", "Scripts")
    os.makedirs(fake_scripts, exist_ok=True)
    fake_exe = os.path.join(fake_scripts, "ssmforge.exe")
    with open(fake_exe, "w") as f:
        f.write("# fake ssmforge.exe")

    # Capture the original env get so we can wrap it without recursion
    orig_get = os.environ.get

    def patched_get(key, default=None):
        if key == "APPDATA":
            return fake_appdata
        return orig_get(key, default)

    try:
        monkeypatch.setattr(cli_mod.sys, "platform", "win32")
        monkeypatch.setattr(cli_mod.os.environ, "get", patched_get)
        on_path, fix = _check_ssmforge_on_path()
        assert on_path is False
        assert fix is not None
        assert "Scripts" in fix or "PATH" in fix
    finally:
        os.unlink(fake_exe)
        os.rmdir(fake_scripts)


def test_check_ssmforge_on_path_on_path_returns_no_fix(monkeypatch):
    """When shutil.which finds ssmforge, we get (True, None)."""
    from ssmforge.cli import _check_ssmforge_on_path
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/ssmforge" if cmd == "ssmforge" else f"/usr/bin/{cmd}")
    on_path, fix = _check_ssmforge_on_path()
    assert on_path is True
    assert fix is None


# ---------- _warn_path_once ----------

def test_warn_path_once_writes_to_stderr(capsys, monkeypatch):
    """When `ssmforge` is not on PATH, the warning prints to stderr."""
    from ssmforge.cli import _warn_path_once
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.delenv("SSMFORGE_NO_PATH_WARN", raising=False)

    _warn_path_once("test")
    captured = capsys.readouterr()
    assert "ssmforge" in captured.err
    assert "PATH" in captured.err or "python -m" in captured.err


def test_warn_path_once_respects_env_var(capsys, monkeypatch):
    """When SSMFORGE_NO_PATH_WARN is set, no warning is printed."""
    from ssmforge.cli import _warn_path_once
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.setenv("SSMFORGE_NO_PATH_WARN", "1")

    _warn_path_once("test")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_path_once_skipped_when_on_path(capsys, monkeypatch):
    """When ssmforge IS on PATH, no warning is printed."""
    from ssmforge.cli import _warn_path_once
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/ssmforge" if cmd == "ssmforge" else f"/usr/bin/{cmd}")
    monkeypatch.delenv("SSMFORGE_NO_PATH_WARN", raising=False)

    _warn_path_once("test")
    captured = capsys.readouterr()
    assert captured.err == ""


# ---------- main() suppresses warning for doctor subcommand ----------

def test_main_skips_warn_for_doctor(monkeypatch, capsys):
    """Running `ssmforge doctor ...` should NOT print the PATH warning."""
    from ssmforge import cli
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.delenv("SSMFORGE_NO_PATH_WARN", raising=False)

    # Patch _check_ssmforge_on_path so we don't depend on system PATH
    monkeypatch.setattr(cli, "_check_ssmforge_on_path", lambda: (False, "hint"))

    with pytest.raises(SystemExit):
        cli.main(["doctor", "--check-install"])
    captured = capsys.readouterr()
    # Warning prefix should NOT appear
    assert "Note: 'ssmforge' is not on your PATH" not in captured.err


def test_main_emits_warn_for_arch(monkeypatch, capsys):
    """Running `ssmforge arch ...` SHOULD print the PATH warning (when off-path)."""
    from ssmforge import cli
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.delenv("SSMFORGE_NO_PATH_WARN", raising=False)

    # Mock arch dispatch to bail out before touching HF
    monkeypatch.setattr(cli, "_cmd_arch", lambda args: None)

    monkeypatch.setattr(cli, "_check_ssmforge_on_path", lambda: (False, "hint"))

    cli.main(["arch", "fake/model", "--dry-run", "--quiet"])
    captured = capsys.readouterr()
    assert "Note: 'ssmforge' is not on your PATH" in captured.err


# ---------- doctor --check-install end-to-end ----------

def test_doctor_check_install_text_output_on_path(monkeypatch, capsys):
    """`doctor --check-install` shows ✓ when on PATH, exits 0."""
    from ssmforge import cli
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/local/bin/ssmforge" if cmd == "ssmforge" else f"/usr/bin/{cmd}")
    monkeypatch.setattr(cli, "_check_ssmforge_on_path", lambda: (True, None))

    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor", "--check-install"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "✓" in captured.out
    assert "ssmforge command on PATH: True" in captured.out


def test_doctor_check_install_text_output_off_path(monkeypatch, capsys):
    """`doctor --check-install` shows ✗ + fix hint when off PATH, exits 1."""
    from ssmforge import cli
    monkeypatch.setattr(cli, "_check_ssmforge_on_path", lambda: (False, "fix-hint-here"))

    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor", "--check-install"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "✗" in captured.out
    assert "ssmforge command on PATH: False" in captured.out
    assert "fix-hint-here" in captured.out


def test_doctor_check_install_json_output(monkeypatch, capsys):
    """`doctor --check-install --format json` returns parseable JSON."""
    import json as _json
    from ssmforge import cli
    monkeypatch.setattr(cli, "_check_ssmforge_on_path", lambda: (True, None))

    with pytest.raises(SystemExit):
        cli.main(["doctor", "--check-install", "--format", "json"])
    captured = capsys.readouterr()
    parsed = _json.loads(captured.out)
    assert parsed["ssmforge_command_on_path"] is True
    assert parsed["fix_hint"] is None


# ---------- doctor without --check-install still works (back-compat) ----------

def test_doctor_default_still_works(monkeypatch, capsys):
    """`doctor` (no flag) still prints version + env info, exits 0."""
    from ssmforge import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "ssmforge doctor" in captured.out
    assert "ssmforge_version" in captured.out
