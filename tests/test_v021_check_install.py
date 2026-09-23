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


# ---------- Regression: attention_bias config overrides weight presence ----------

def test_attention_bias_config_none_does_not_override_to_true(monkeypatch):
    """When config.attention_bias is None (e.g. Qwen2), weight-side bias
    tensors should NOT auto-set attention_bias to True.

    Regression: v0.2.0's full-load path detected Qwen2 as 'attention_bias=True'
    because the weights include zero-initialized bias tensors. The config says
    'no biases' (None + Qwen convention), so that's what we should report.
    """
    from ssmforge.analyze.state_dict_scan import scan_state_dict, QuirkReport
    from types import SimpleNamespace

    # Simulate a tiny Qwen2-like model with bias tensors AND config.attention_bias=None
    state_dict = {
        # layer 0
        "model.layers.0.self_attn.q_proj.weight":  __import__("numpy").zeros((896, 896)),
        "model.layers.0.self_attn.q_proj.bias":    __import__("numpy").zeros((896,)),
        "model.layers.0.self_attn.k_proj.weight":  __import__("numpy").zeros((128, 896)),
        "model.layers.0.self_attn.k_proj.bias":    __import__("numpy").zeros((128,)),
        "model.layers.0.self_attn.v_proj.weight":  __import__("numpy").zeros((128, 896)),
        "model.layers.0.self_attn.v_proj.bias":    __import__("numpy").zeros((128,)),
        "model.layers.0.self_attn.o_proj.weight":  __import__("numpy").zeros((896, 896)),
        "model.layers.0.self_attn.o_proj.bias":    __import__("numpy").zeros((896,)),
        # sample layer 1 to make sure num_layers detection works
        "model.layers.1.self_attn.q_proj.weight":  __import__("numpy").zeros((896, 896)),
    }
    cfg = SimpleNamespace(
        model_type="qwen2", architectures=["Qwen2ForCausalLM"],
        vocab_size=151936, hidden_size=896, intermediate_size=4864,
        num_hidden_layers=2, num_attention_heads=14, num_key_value_heads=2,
        attention_bias=None,  # the key field
        tie_word_embeddings=True,
        rope_theta=None, rope_parameters={"rope_theta": 1000000.0},
        rope_scaling=None, rms_norm_eps=1e-6,
    )

    report = scan_state_dict(state_dict, config=cfg)
    assert report.attention_bias is False, (
        f"Qwen2-like with attention_bias=None should report False even though "
        f"bias tensors exist. Got {report.attention_bias}, "
        f"bias_keys_found={report.bias_keys_found}"
    )
    # But we DO record that the bias tensors are present (informational)
    assert len(report.bias_keys_found) > 0


def test_attention_bias_config_true_overrides_to_true(monkeypatch):
    """When config.attention_bias=True, quirk should be True even without weights."""
    from ssmforge.analyze.state_dict_scan import scan_config_only, QuirkReport
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        model_type="phi3", architectures=["Phi3ForCausalLM"],
        vocab_size=32000, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        attention_bias=True,  # explicit
        tie_word_embeddings=False,
        rope_theta=None, rope_parameters={"rope_theta": 10000.0},
        rope_scaling=None,
    )
    report = scan_config_only(cfg)
    assert report.attention_bias is True


def test_attention_bias_config_false_stays_false(monkeypatch):
    """When config.attention_bias=False (Llama), quirk is False. Even if bias
    tensors were somehow present, config wins."""
    from ssmforge.analyze.state_dict_scan import scan_state_dict
    from types import SimpleNamespace
    state_dict = {
        "model.layers.0.self_attn.q_proj.bias": __import__("numpy").zeros((64,)),
    }
    cfg = SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"],
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        attention_bias=False,  # Llama convention
        tie_word_embeddings=False,
        rope_theta=None, rope_parameters={"rope_theta": 10000.0},
        rope_scaling=None, rms_norm_eps=1e-5,
    )
    report = scan_state_dict(state_dict, config=cfg)
    assert report.attention_bias is False


# ---------- v0.2.2: --output directory validation ----------

def test_check_output_path_none_returns_safe():
    """No --output flag → (None, None); no validation needed."""
    from ssmforge.cli import _check_output_path
    parent, err = _check_output_path(None)
    assert parent is None
    assert err is None


def test_check_output_path_dash_returns_safe():
    """`--output -` means stdout → (None, None)."""
    from ssmforge.cli import _check_output_path
    parent, err = _check_output_path("-")
    assert parent is None
    assert err is None


def test_check_output_path_writable_dir_is_ok(tmp_path):
    """Writing into an existing writable dir → (parent, None)."""
    from ssmforge.cli import _check_output_path
    target = tmp_path / "report.md"
    parent, err = _check_output_path(str(target))
    assert parent is not None
    assert err is None


def test_check_output_path_missing_parent_returns_error(tmp_path):
    """Writing into a non-existent parent dir → error with hint."""
    from ssmforge.cli import _check_output_path
    target = tmp_path / "nonexistent_subdir" / "report.md"
    parent, err = _check_output_path(str(target))
    assert parent is not None
    assert err is not None
    assert "does not exist" in err
    assert "--output ./report.md" in err  # hint present


def test_check_output_path_unwritable_returns_error(tmp_path):
    """If parent is read-only, return error with alternative suggestions."""
    from ssmforge.cli import _check_output_path
    import stat
    # Create a read-only directory
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0o555 = read+exec
    try:
        target = ro_dir / "report.md"
        parent, err = _check_output_path(str(target))
        # On Windows chmod is partially ignored, so skip if write somehow allowed
        if not os.access(ro_dir, os.W_OK):
            assert err is not None
            assert "Permission denied" in err
            assert "writable location" in err
        else:
            # Windows-style: chmod didn't actually block writes; skip
            pass
    finally:
        ro_dir.chmod(stat.S_IRWXU)  # restore for cleanup


def test_write_or_print_rejects_unwritable_directory(tmp_path, capsys):
    """Actual end-to-end test: --output to bad path → exit 1 with hint."""
    from ssmforge.cli import _write_or_print, main
    import stat
    # Create a read-only dir
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    target = ro_dir / "report.md"
    try:
        if not os.access(ro_dir, os.W_OK):
            # Unix: chmod actually blocks
            with pytest.raises(SystemExit) as exc:
                _write_or_print("test content", str(target))
            assert exc.value.code == 1
            captured = capsys.readouterr()
            assert "writable location" in captured.err
            assert "--output" in captured.err
        else:
            # Windows: chmod ignored, skip the actual test
            pass
    finally:
        ro_dir.chmod(stat.S_IRWXU)


# ---------- v0.2.2: backslash path detection (Git Bash on Windows trap) ----------

def test_check_output_path_backslash_only_warns(monkeypatch, capsys):
    """Input like 'C:\\\\nope.md' on POSIX gets caught with a backslash hint.

    Regression: v0.2.2 first attempt didn't catch this case because
    os.path.abspath silently converts it to a single-component path,
    whose parent ('.') always exists and is always writable.
    """
    from ssmforge.cli import _check_output_path
    # The literal backslash path that Git Bash on Windows passes through
    monkeypatch.setattr("sys.platform", "linux")  # simulate POSIX
    parent, err = _check_output_path(r"C:\nope.md")
    assert err is not None
    assert "backslashes" in err
    # Suggested fix is in the hint
    assert "/" in err  # forwarded-slash version mentioned


def test_check_output_path_forward_slash_not_flagged(monkeypatch, tmp_path):
    """'C:/nope.md' is treated as a real (Windows-on-POSIX) path;
    it's flagged as a non-existent directory, NOT a backslash issue."""
    from ssmforge.cli import _check_output_path
    monkeypatch.setattr("sys.platform", "linux")
    parent, err = _check_output_path("C:/nope.md")
    assert err is not None
    assert "backslashes" not in err  # not the backslash issue
    assert "does not exist" in err  # dir doesn't exist


def test_write_or_print_catches_written_to_wrong_path(monkeypatch, tmp_path, capsys):
    """If write_text 'succeeds' to a literal filename but the user's intent
    was different (e.g., they meant a Windows path), warn after writing.

    We simulate by using a path the user gave as 'C:\\\\nope.md' which
    on POSIX becomes a single literal filename in cwd.
    """
    from ssmforge.cli import _write_or_print
    monkeypatch.setattr("sys.platform", "linux")
    # The first call catches the backslash-as-filename issue via pre-check.
    # So _write_or_print fails the pre-check.
    bs = chr(92)
    with pytest.raises(SystemExit) as exc:
        _write_or_print("test", f"C:{bs}nope.md")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "backslashes" in captured.err


# ---------- v0.2.3: bash-mangled Windows path detection ----------

def test_check_output_path_bash_mangled_windows_path():
    """`C:UsersnetgeDesktop.md` (no separators) is detected as a bash-mangled
    Windows path. The fix in v0.2.2 caught backslash-only paths but missed
    this case where bash already stripped them.
    """
    from ssmforge.cli import _check_output_path
    parent, err = _check_output_path("C:UsersnetgeDesktopNope.md")
    assert err is not None
    assert "separators stripped" in err or "Git Bash strips" in err
    # Hint should mention quoting
    assert "quotes" in err or "wrap" in err.lower()


def test_check_output_path_bash_mangled_keeps_short_drive_only():
    """`C:report.md` is also caught — drive letter with no path separators
    is ambiguous on Windows (drive-relative vs absolute). Better to ask
    the user to disambiguate than silently write to the wrong place.
    """
    from ssmforge.cli import _check_output_path
    parent, err = _check_output_path("C:report.md")
    assert err is not None
    # Either "bash mangled" or "backslashes" — both are valid catches
    assert ("separators stripped" in err or "backslashes" in err)


def test_write_or_print_catches_bash_mangled(monkeypatch, capsys):
    """End-to-end: when user passes a bash-mangled Windows path,
    the CLI rejects it before silently writing to cwd."""
    from ssmforge.cli import _write_or_print
    # The actual case from the bug report
    with pytest.raises(SystemExit) as exc:
        _write_or_print("test content", "C:UsersnetgeOneDriveDesktopNope.md")
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "separators stripped" in captured.err


# ---------- v0.2.3: pre-parse guard for bash-mangled --output ----------

def test_pre_parse_guard_bash_mangled_path_separate_args(capsys):
    """`--output C:Users...` (separate args) is caught BEFORE argparse runs."""
    from ssmforge.cli import _pre_parse_path_guard
    with pytest.raises(SystemExit) as exc:
        _pre_parse_path_guard(["arch", "X", "--dry-run", "--output", "C:UsersnetgeDesktop.md"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "bash-mangled" in captured.err
    assert "C:UsersnetgeDesktop.md" in captured.err


def test_pre_parse_guard_bash_mangled_path_equals_form(capsys):
    """`--output=C:Users...` (equals form) is also caught."""
    from ssmforge.cli import _pre_parse_path_guard
    with pytest.raises(SystemExit):
        _pre_parse_path_guard(["arch", "X", "--output=C:UsersnetgeDesktop.md"])
    captured = capsys.readouterr()
    assert "bash-mangled" in captured.err


def test_pre_parse_guard_backslash_only(capsys):
    """Literal backslash path (no forward slashes) caught."""
    from ssmforge.cli import _pre_parse_path_guard
    bs = chr(92)
    with pytest.raises(SystemExit):
        _pre_parse_path_guard(["arch", "X", "--output", f"C:{bs}Users{bs}netge{bs}Desktop.md"])
    captured = capsys.readouterr()
    assert "backslashes" in captured.err or "quotes" in captured.err


def test_pre_parse_guard_accepts_posix_paths():
    """Plain POSIX paths pass through without error."""
    from ssmforge.cli import _pre_parse_path_guard
    # Should not raise SystemExit
    _pre_parse_path_guard(["arch", "X", "--output", "./report.md"])
    _pre_parse_path_guard(["arch", "X", "--output", "/tmp/report.md"])
    _pre_parse_path_guard(["arch", "X", "--output", "report.md"])


def test_pre_parse_guard_accepts_quoted_windows_path_with_forward_slashes():
    """`C:/Users/...` (forward slashes) is fine — works on Windows + POSIX."""
    from ssmforge.cli import _pre_parse_path_guard
    # Should not raise SystemExit
    _pre_parse_path_guard(["arch", "X", "--output", "C:/Users/netge/Desktop/report.md"])


def test_pre_parse_guard_accepts_bare_drive_filename():
    """`C:report.md` — drive letter + filename only — is caught as ambiguous.

    Even though it could legitimately mean "C: drive, relative file",
    this is almost always a sign of bash mangling. Better to error.
    """
    from ssmforge.cli import _pre_parse_path_guard
    with pytest.raises(SystemExit):
        _pre_parse_path_guard(["arch", "X", "--output", "C:report.md"])
