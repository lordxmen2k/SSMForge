"""Comprehensive manual test of every ssmforge CLI flag combination.

Uses tiny test models that should already be in the HF cache.
Each test verifies: exit code, output shape, and a key substring.

Run with: python comprehensive_test.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CLI = "/workspace/SSMForge/.venv/bin/python -m ssmforge.cli"
# Use the cached tiny models — should already be downloaded.
# Smallest first, then larger. Dry-run by default to skip weight loads.
MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",  # ~4MB, fastest
    "Qwen/Qwen2-0.5B-Instruct",  # ~1GB
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ~2.2GB
]


def run(cmd: str, expect_exit: int | None = None, timeout: int = 60) -> tuple[int, str, str]:
    """Run a CLI command, return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        cwd="/workspace/SSMForge",
    )


def run(cmd: str, expect_exit: int | None = None, timeout: int = 120) -> tuple[int, str, str]:
    """Run a CLI command, return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        cwd="/workspace/SSMForge",
    )
    if expect_exit is not None and result.returncode != expect_exit:
        print(f"  ✗ FAIL: expected exit {expect_exit}, got {result.returncode}")
        print(f"    cmd: {cmd}")
        print(f"    stdout: {result.stdout[:500]}")
        print(f"    stderr: {result.stderr[:500]}")
        return result.returncode, result.stdout, result.stderr
    return result.returncode, result.stdout, result.stderr


def ok(label: str):
    print(f"  ✓ {label}")


# ============ doctor ============
print("\n=== ssmforge doctor ===")
exit, stdout, stderr = run(f"{CLI} doctor", expect_exit=0)
if "ssmforge_version" in stdout and "transformers_version" in stdout:
    ok("doctor prints version + transformers + hf_home")

exit, stdout, stderr = run(f"{CLI} doctor --format json", expect_exit=0)
try:
    parsed = json.loads(stdout)
    if "ssmforge_version" in parsed and "transformers_version" in parsed:
        ok("doctor --format json works")
except Exception:
    print(f"  ✗ FAIL: doctor --format json not valid JSON")

# ============ --version ============
print("\n=== --version ===")
exit, stdout, stderr = run(f"{CLI} --version", expect_exit=0)
if stdout.strip().startswith("ssmforge ") and "0.1." in stdout:
    ok("--version prints version")

exit, stdout, stderr = run(f"{CLI} -V", expect_exit=0)
if stdout.strip().startswith("ssmforge "):
    ok("-V works as alias")

# ============ arch single model: every format ============
print("\n=== arch single model ===")
for model in MODELS:
    # JSON default
    exit, stdout, stderr = run(
        f'{CLI} arch {model} --dry-run --quiet', expect_exit=0, timeout=60
    )
    try:
        parsed = json.loads(stdout)
        if parsed.get("model_id") == model:
            ok(f"{model}: --dry-run JSON parses with correct model_id")
        else:
            print(f"  ✗ FAIL: {model} model_id mismatch: {parsed.get('model_id')}")
    except Exception:
        print(f"  ✗ FAIL: {model} JSON parse failed")

    # Markdown
    exit, stdout, stderr = run(
        f'{CLI} arch {model} --dry-run --format markdown --quiet', expect_exit=0, timeout=60
    )
    if "# `ssmforge arch`" in stdout and "## Profile" in stdout and "## Detected quirks" in stdout:
        ok(f"{model}: --format markdown has all sections")

    # --output to file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        exit, stdout, stderr = run(
            f'{CLI} arch {model} --dry-run --quiet --output {tmp_path}', expect_exit=0, timeout=60
        )
        if os.path.exists(tmp_path) and "fake" not in tmp_path:
            with open(tmp_path) as f:
                data = json.load(f)
            if data.get("model_id") == model:
                ok(f"{model}: --output writes valid JSON file")
    finally:
        os.unlink(tmp_path)

    # --output - (stdout)
    exit, stdout, stderr = run(
        f'{CLI} arch {model} --dry-run --quiet --output -', expect_exit=0, timeout=60
    )
    try:
        parsed = json.loads(stdout)
        if parsed.get("model_id") == model:
            ok(f"{model}: --output - prints to stdout")
    except Exception:
        print(f"  ✗ FAIL: {model} --output - didn't print JSON")

# ============ arch --profile ============
print("\n=== arch --profile ===")
exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[0]} --dry-run --profile', expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if "profile" in parsed and "family" in parsed.get("profile", {}):
        ok("--profile emits profile-only JSON")
except Exception:
    print(f"  ✗ FAIL: --profile didn't emit expected JSON")

# ============ arch --fields ============
print("\n=== arch --fields ===")
exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[1]} --dry-run --quiet --fields "profile,quirks.attention_bias"', expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if "profile" in parsed and parsed.get("profile", {}).get("family") is not None:
        ok("--fields subsets to profile + single quirk")
    # Should NOT have config (we didn't ask for it)
    if "config" not in parsed:
        ok("--fields excludes non-requested top-level keys")
    # Should have the requested quirk
    if parsed.get("quirks", {}).get("attention_bias") is not None:
        ok("--fields respects nested quirk path")
except Exception as e:
    print(f"  ✗ FAIL: --fields didn't work: {e}")

# Test --fields with invalid field name
exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[0]} --dry-run --quiet --fields "nonexistent"', expect_exit=0, timeout=60
)
if "not found" in stderr:
    ok("--fields warns on invalid field name")

# ============ arch --diff (legacy) ============
print("\n=== arch --diff (legacy 2-way) ===")
exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[0]} --diff {MODELS[1]} --dry-run --quiet', expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if "model_a" in parsed and "model_b" in parsed and "differences" in parsed:
        ok("--diff JSON has model_a/model_b/differences")
except Exception:
    print(f"  ✗ FAIL: --diff didn't produce expected JSON")

exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[0]} --diff {MODELS[1]} --dry-run --format markdown --quiet',
    expect_exit=0, timeout=60
)
if "# Architectural diff" in stdout:
    ok("--diff markdown format works")

# ============ arch --compare (new) ============
print("\n=== arch --compare (2-way and 3-way) ===")
exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} {MODELS[1]} --dry-run --quiet',
    expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if parsed.get("comparison_type") == "multi_model" and parsed.get("model_count") == 2:
        ok("--compare 2-way JSON has comparison_type + model_count")
except Exception:
    print(f"  ✗ FAIL: --compare 2-way didn't produce expected JSON")

exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} {MODELS[1]} {MODELS[2]} --dry-run --format markdown --quiet',
    expect_exit=0, timeout=60
)
if "Architectural comparison: 3 models" in stdout:
    ok("--compare 3-way markdown works")
if "| Field |" in stdout:
    ok("--compare 3-way markdown has table header")

exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[0]} --compare {MODELS[1]} {MODELS[2]} --dry-run --quiet',
    expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if parsed.get("model_count") == 3 and MODELS[0] in parsed.get("models", []):
        ok("--compare with source positional works")
except Exception:
    print(f"  ✗ FAIL: source positional + --combine failed")

# --compare with --only-different
exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} {MODELS[1]} {MODELS[2]} --dry-run --format markdown --only-different --quiet',
    expect_exit=0, timeout=60
)
if "Architectural comparison: 3 models" in stdout and "Identical across all models" not in stdout:
    ok("--only-different strips identical section from markdown")

# --compare with --only-different + JSON
exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} {MODELS[1]} {MODELS[2]} --dry-run --only-different --quiet',
    expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if all(f["all_same"] is False for f in parsed.get("fields", [])):
        ok("--only-different JSON only contains differing fields")
except Exception:
    print(f"  ✗ FAIL: --only-different JSON broken")

# --compare --fields
exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} {MODELS[1]} --dry-run --quiet --fields "hidden_size,model_type"',
    expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    field_names = [f["field"] for f in parsed.get("fields", [])]
    # hidden_size should be in there (varies), model_type should be in there (varies or same)
    if "hidden_size" in field_names:
        ok("--fields works in --compare mode")
except Exception:
    print(f"  ✗ FAIL: --fields in --compare didn't work")

# ============ --compare error cases ============
print("\n=== --compare error cases ===")
exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} --dry-run --quiet', expect_exit=1
)
if "at least 2 models" in stderr.lower():
    ok("--compare with single model exits 1 with clear error")

# ============ --compare --profile ============
print("\n=== --compare --profile ===")
exit, stdout, stderr = run(
    f'{CLI} arch --compare {MODELS[0]} {MODELS[1]} {MODELS[2]} --dry-run --profile --quiet',
    expect_exit=0, timeout=60
)
try:
    parsed = json.loads(stdout)
    if "profiles" in parsed and len(parsed["profiles"]) == 3:
        ok("--compare --profile emits profiles list")
except Exception:
    print(f"  ✗ FAIL: --compare --profile broken")

# ============ fail-fast on bad local path ============
print("\n=== fail-fast on bad paths ===")
exit, stdout, stderr = run(
    f'{CLI} arch /tmp/no-such-path-12345 --dry-run', expect_exit=1
)
if "local path does not exist" in stderr:
    ok("local path /tmp/no-such-path-12345 fails fast")

exit, stdout, stderr = run(
    f'{CLI} arch /tmp/no-such-path-12345 --dry-run --quiet', expect_exit=1
)
if "local path does not exist" in stderr:
    ok("fail-fast message visible even with --quiet")

exit, stdout, stderr = run(
    f'{CLI} arch /tmp/no-such-path-12345 {MODELS[0]} --diff {MODELS[1]}', expect_exit=1
)
if "local path does not exist" in stderr:
    ok("fail-fast works in --diff mode too")

# ============ --output to bad path ============
print("\n=== --output to bad path ===")
exit, stdout, stderr = run(
    f'{CLI} arch {MODELS[0]} --dry-run --quiet --output /tmp/nonexistent_dir/report.json', expect_exit=1
)
if "cannot write" in stderr.lower() or "no such file" in stderr.lower():
    ok("--output to bad path exits 1 with clear error")

# ============ version flag works with arch subcommand ============
print("\n=== --version on arch subcommand ===")
exit, stdout, stderr = run(f"{CLI} arch --version", expect_exit=0)
if stdout.strip().startswith("ssmforge "):
    ok("arch --version prints version")

# ============ help works for every subcommand ============
print("\n=== help output ===")
for sub in ["arch", "doctor"]:
    exit, stdout, stderr = run(f"{CLI} {sub} --help", expect_exit=0)
    if "usage:" in stdout.lower():
        ok(f"{sub} --help works")

print("\n=== Done. ===")
