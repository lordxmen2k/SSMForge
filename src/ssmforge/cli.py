"""Command-line interface for SSMForge.

Subcommands:
  arch MODEL      Inspect a HuggingFace model and report architectural quirks.
  doctor          Print ssmforge + environment info (cache, tokens, versions).

The `arch` subcommand supports:
  - --format {json,markdown,md}: report output format (default: json)
  - --output PATH / -o PATH: write to file; '-' means stdout
  - --quiet: suppress the human-readable summary on stderr
  - --diff OTHER: compare against another model and print a diff
  - --dry-run: fetch only config, estimate memory, skip weight download
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from pathlib import Path


@contextlib.contextmanager
def _arch_load_progress(model_id: str, dry_run: bool = False, revision: str | None = None):
    """Load a HuggingFace model for `ssmforge arch`, with progress on stderr.

    If `dry_run=True`, fetches only the config (no weight download).
    If `revision` is given, pins to that HF commit sha/tag/branch.

    Yields the loaded object. The resolved revision sha is stashed on
    `obj.ssmforge_revision` after load so the caller can read it.
    """
    if dry_run:
        print(f"Loading config for {model_id}... (dry-run, no weights)", file=sys.stderr)
    else:
        print(f"Loading {model_id}... (downloading if not cached)", file=sys.stderr)

    try:
        if dry_run:
            # Config only — no weight download. AutoConfig fetches config.json.
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, revision=revision)
            # Resolve the actual sha that was loaded
            try:
                resolved = getattr(config, "_name_or_path", None)
                rev = getattr(config, "revision", revision) or revision
                # Try to look up the actual sha from the HF hub cache
                if revision is None:
                    rev = _resolve_hf_revision(model_id)
                else:
                    rev = revision
                config.ssmforge_revision = rev
            except Exception:
                config.ssmforge_revision = revision
            yield config
        else:
            from transformers import AutoModelForCausalLM
            # `dtype=` is the new transformers ≥ 4.50 kwarg; `torch_dtype` is deprecated
            # and emits a warning on every load. Try `dtype` first, fall back if older.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype="auto", revision=revision
                )
            except TypeError:
                # transformers < 4.50
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype="auto", revision=revision
                )
            try:
                if revision is None:
                    model.ssmforge_revision = _resolve_hf_revision(model_id)
                else:
                    model.ssmforge_revision = revision
            except Exception:
                model.ssmforge_revision = revision
            yield model
    finally:
        pass


def _resolve_hf_revision(model_id: str) -> str | None:
    """Resolve the current HEAD sha for a HF model.

    Uses the huggingface_hub API. Returns None on failure.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id)
        return getattr(info, "sha", None)
    except Exception:
        return None


def _subset_report(report: dict, fields: str | None) -> dict:
    """Subset a report to only the comma-separated field names.

    Field names can be top-level keys (e.g. 'config', 'profile') or
    dotted paths (e.g. 'profile.family', 'quirks.attention_bias').
    Returns a new dict. If `fields` is None, returns the report as-is.
    If `fields == 'all'`, returns the full report.
    """
    if fields is None or fields == "all":
        return report

    requested = [f.strip() for f in fields.split(",") if f.strip()]
    out: dict = {}

    def _set_path(d, path, value):
        """Set d['a']['b'] = value, creating dicts as needed."""
        parts = path.split(".")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur.get(p), dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    for path in requested:
        # Walk the path in `report`, return value or None if any step is missing
        parts = path.split(".")
        cur = report
        ok = True
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok:
            _set_path(out, path, cur)
        else:
            print(f"Warning: field {path!r} not found in report", file=sys.stderr)
    return out


def _profile_only_report(report: dict) -> dict:
    """Return just the profile section of a report."""
    return {
        "model_id": report.get("model_id"),
        "model_type": report.get("model_type"),
        "profile": report.get("profile", {}),
    }


def _is_local_path(source: str) -> bool:
    """Return True if `source` looks like a local filesystem path.

    Heuristic: starts with `.`, `/`, `~`, or contains a path separator on Windows.
    HF model ids don't contain `/` at the start of a path component and have
    the form `org/name` (org has no leading dot).
    """
    if source.startswith(("./", "../", "/", "~")):
        return True
    # Windows absolute path like "C:\..." or "C:/..."
    if len(source) >= 3 and source[1] == ":" and source[2] in ("/", "\\"):
        return True
    return False


def _format_diff_markdown(report_a: dict, report_b: dict, diff: dict) -> str:
    """Render a diff between two reports as Markdown."""
    lines = []
    lines.append(f"# Architectural diff")
    lines.append("")
    lines.append(f"- **Model A:** `{report_a.get('model_id', '?')}`")
    lines.append(f"- **Model B:** `{report_b.get('model_id', '?')}`")
    lines.append("")
    if diff["identical"]:
        lines.append("**Identical** — architectures match exactly.")
        return "\n".join(lines)

    lines.append(f"**{len(diff['differences'])} differences found.**")
    lines.append("")
    if diff["added_quirks"]:
        lines.append(f"### Added quirks (in B but not A)")
        for q in diff["added_quirks"]:
            lines.append(f"- `{q}`")
        lines.append("")
    if diff["removed_quirks"]:
        lines.append(f"### Removed quirks (in A but not B)")
        for q in diff["removed_quirks"]:
            lines.append(f"- `{q}`")
        lines.append("")
    lines.append("### All differences")
    lines.append("")
    lines.append("| Field | A | B |")
    lines.append("|-------|---|---|")
    for d in diff["differences"]:
        lines.append(f"| `{d['field']}` | `{d['a']}` | `{d['b']}` |")
    return "\n".join(lines)


def _format_dry_run_summary(report: dict) -> str:
    """Format a dry-run report as a human-readable summary."""
    cfg = report.get("config", {})
    mem = report.get("memory_estimate", {})
    profile = report.get("profile", {})

    lines = []
    lines.append(f"SSMForge arch DRY-RUN: {report.get('model_id', '<unknown>')}")
    lines.append(f"  family:              {profile.get('family', '?')}")
    lines.append(f"  attention_type:      {profile.get('attention_type', '?')}")
    lines.append(f"  mlp_type:            {profile.get('mlp_type', '?')}")
    lines.append(f"  norm_type:           {profile.get('norm_type', '?')}")
    lines.append("")
    lines.append("Model config:")
    lines.append(f"  model_type:          {report.get('model_type', '?')}")
    lines.append(f"  hidden_size:         {cfg.get('hidden_size')}")
    lines.append(f"  num_hidden_layers:   {cfg.get('num_hidden_layers')}")
    lines.append(f"  num_attention_heads: {cfg.get('num_attention_heads')}")
    lines.append(f"  num_kv_heads:        {cfg.get('num_key_value_heads')}")
    lines.append(f"  intermediate_size:   {cfg.get('intermediate_size')}")
    lines.append(f"  vocab_size:          {cfg.get('vocab_size')}")
    lines.append(f"  max_position:        {cfg.get('max_position_embeddings')}")
    lines.append(f"  torch_dtype:         {cfg.get('torch_dtype')}")
    lines.append(f"  tie_word_embeddings: {cfg.get('tie_word_embeddings')}")
    lines.append(f"  attention_bias:      {cfg.get('attention_bias')}")
    lines.append("")
    if mem.get("estimated_params"):
        n_params = mem["estimated_params"]
        n_bytes = mem["estimated_bytes"]
        # Human-format
        if n_params >= 1e9:
            human = f"{n_params / 1e9:.2f}B"
        elif n_params >= 1e6:
            human = f"{n_params / 1e6:.2f}M"
        elif n_params >= 1e3:
            human = f"{n_params / 1e3:.2f}K"
        else:
            human = str(n_params)
        if n_bytes >= 1024 ** 3:
            sz = f"{n_bytes / (1024 ** 3):.2f} GB"
        elif n_bytes >= 1024 ** 2:
            sz = f"{n_bytes / (1024 ** 2):.2f} MB"
        elif n_bytes >= 1024:
            sz = f"{n_bytes / 1024:.2f} KB"
        else:
            sz = f"{n_bytes} B"
        lines.append(f"Memory estimate: ~{sz} in {mem.get('dtype', 'unknown')} ({human} params)")
        lines.append("")

    # Quirks we could detect from config
    lines.append("Detected quirks (config-only):")
    q = report.get("quirks", {})
    if q.get("grouped_attention"):
        lines.append("  ✓ GQA")
    if q.get("mqa"):
        lines.append("  ✓ MQA (kv_heads=1)")
    if q.get("moe"):
        lines.append(f"  ✓ MoE (num_experts={q.get('num_experts')}, top_k={q.get('moe_top_k')})")
    if q.get("sliding_window"):
        lines.append(f"  ✓ sliding_window={q['sliding_window']}")
    if q.get("soft_capping"):
        caps = q["soft_capping"]
        cap_str = ", ".join(f"{k}={v}" for k, v in caps.items())
        lines.append(f"  ✓ soft-capping: {cap_str}")
    if q.get("partial_rope_factor"):
        lines.append(f"  ✓ partial RoPE factor: {q['partial_rope_factor']}")
    if q.get("rope_theta") is not None:
        lines.append(f"  ✓ rope_theta={q['rope_theta']} (from {cfg.get('rope_theta_source', '?')})")

    # Quirks NOT detectable without state_dict
    lines.append("")
    lines.append("Quirks requiring --no-dry-run (state_dict inspection):")
    lines.append("  · attention_bias (config truth vs state_dict truth)")
    lines.append("  · tied_embeddings (lm_head tensor identity)")
    lines.append("  · fused_qkv (qkv_proj vs q/k/v split)")
    lines.append("  · fused_gate_up (gate_up_proj vs gate/up split)")
    lines.append("  · mlp_type / norm_type (state_dict structure)")
    lines.append("  · layer_scale (ls1/ls2 keys)")
    lines.append("")

    compat = report.get("compatibility", {})
    if compat.get("is_compatible"):
        lines.append("Compatibility: OK (no blockers)")
    else:
        lines.append(f"Compatibility: BLOCKED by: {', '.join(compat.get('blockers', []))}")

    return "\n".join(lines)


def _write_or_print(text: str, output_path: str | None) -> None:
    """Write text to a file if path given, else print to stdout.

    Special-case: if output_path is '-', print to stdout (Unix convention).
    """
    if output_path and output_path != "-":
        try:
            Path(output_path).write_text(text, encoding="utf-8")
        except (PermissionError, FileNotFoundError, IsADirectoryError, OSError) as e:
            print(f"Error: cannot write to {output_path!r}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(text)


def _check_local_path(source: str) -> None:
    """If `source` looks like a local path, fail fast if it doesn't exist.

    HF model ids pass through this unchanged.
    """
    if not _is_local_path(source):
        return
    expanded = os.path.expanduser(source)
    if not os.path.exists(expanded):
        print(f"Error: local path does not exist: {source}", file=sys.stderr)
        print(f"  expanded: {expanded}", file=sys.stderr)
        print(f"  Hint: if you meant a HuggingFace model id, omit any leading '/' or './'", file=sys.stderr)
        sys.exit(1)


def _check_ssmforge_on_path() -> tuple[bool, str | None]:
    """Best-effort detection of whether the `ssmforge` command is on PATH.

    Returns ``(on_path, fix_hint)``. Always returns successfully; never raises
    or exits. Used by `_cmd_doctor` and the main() pre-dispatch warning.

    Detection strategy:
    - On Windows / MINGW64, look for `%APPDATA%\\Python\\Python<ver>\\Scripts\\ssmforge.exe`
    - On POSIX, look for `which("ssmforge")` (which uses PATH + $PATHEXT)
    - If we got here via `python -m ssmforge.cli`, the script wrapper exists
      somewhere — we just don't know if it's on PATH.

    If we can detect a typical Windows user-site Scripts dir that *isn't* on
    PATH, we return a concrete fix string. Otherwise we just return the
    on/off state.
    """
    import shutil
    on_path = shutil.which("ssmforge") is not None

    if on_path:
        return True, None

    # On Windows the most common missing-PATH case is `%APPDATA%\Python\Scripts`
    if sys.platform.startswith("win"):
        py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
        candidates = [
            os.path.join(os.environ.get("APPDATA", ""), "Python", f"Python{py_ver}", "Scripts"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", f"Python{py_ver}", "Scripts"),
            os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "Python", f"Python{py_ver}", "Scripts"),
        ]
        for cand in candidates:
            if cand and os.path.isdir(cand):
                exe = os.path.join(cand, "ssmforge.exe")
                if os.path.isfile(exe):
                    fix = (
                        f"The 'ssmforge' script was installed to:\n"
                        f"    {exe}\n"
                        f"but that folder is not on PATH.\n\n"
                        f"  Quick fix (current shell):\n"
                        f"    export PATH=\"{cand}:$PATH\"\n\n"
                        f"  Permanent fix (PowerShell, restart shell after):\n"
                        f"    [Environment]::SetEnvironmentVariable(\"PATH\", \"{cand};\" + [Environment]::GetEnvironmentVariable(\"PATH\", \"User\"), \"User\")\n\n"
                        f"  Or just use: python -m ssmforge.cli\n"
                        f"  See: https://github.com/lordxmen2k/SSMForge#troubleshooting"
                    )
                    return False, fix
        return False, "ssmforge installed but 'ssmforge' command not on PATH. Use 'python -m ssmforge.cli' instead, or add Scripts to PATH."
    else:
        # On Linux/macOS, missing-on-PATH is rare; defer to `python -m`
        return False, "ssmforge installed but 'ssmforge' command not on PATH. Use 'python -m ssmforge.cli' instead."


def _warn_path_once(ctx: str) -> None:
    """Print a one-time warning if `ssmforge` is not on PATH.

    Called from main() before the actual subcommand dispatch. Suppressed
    when ``SSMFORGE_NO_PATH_WARN`` is set, or when the user is already
    invoking via `python -m ssmforge.cli` (which we can't easily detect
    from inside main(), so we just use a try/except to stay quiet).
    """
    if os.environ.get("SSMFORGE_NO_PATH_WARN"):
        return
    try:
        on_path, fix = _check_ssmforge_on_path()
    except Exception:
        return  # never break the CLI on a path check
    if on_path:
        return
    print(
        f"Note: 'ssmforge' is not on your PATH for `python -m ssmforge.cli` users.\n"
        f"      This is normal if you're running via 'python -m ssmforge.cli'.\n"
        f"      If 'ssmforge --version' fails, see: https://github.com/lordxmen2k/SSMForge#troubleshooting",
        file=sys.stderr,
    )


def _install_signal_handlers() -> None:
    """Make Ctrl+C print a clean message instead of a traceback."""
    def _sigint_handler(sig, frame):
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)  # 130 = SIGINT convention
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        # Not in main thread, or platform doesn't support it. Best-effort.
        pass


def _quiet_transformers_warnings() -> None:
    """Reduce noise from transformers' internal logging.

    The `torch_dtype` deprecation warning fires on every from_pretrained
    call. For a tool that's already migrated to `dtype=`, the warning
    is just clutter. We move the transformers logger to ERROR-level
    when --quiet is on, and leave WARNING-level visible otherwise so
    users still see real problems.

    Also silences the 'pad_token_id' / 'generation_config' warnings that
    fire on some small test models.
    """
    try:
        import logging
        import transformers
        # Don't downgrade below WARNING — we want to see actual errors.
        # But the deprecation flood is at INFO/DEBUG.
        for name in ("transformers", "transformers.modeling_utils",
                     "transformers.configuration_utils", "transformers.tokenization_utils_base"):
            logging.getLogger(name).setLevel(logging.ERROR)
        # The `[transformers] torch_dtype is deprecated` and
        # `[transformers] pad_token_id must be None` messages bypass
        # logging (they print directly to stderr via the `warnings`
        # module or via direct print()). Catch them via warnings too.
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")
        _warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        _warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `ssmforge` command."""
    _install_signal_handlers()

    # One-time PATH warning (suppressed by SSMFORGE_NO_PATH_WARN=1)
    # Skip for `ssmforge doctor --check-install` itself so its output stays clean.
    # When called via `python -m ssmforge.cli`, argv is None — fall back to
    # sys.argv[1:].
    effective_argv = argv if argv is not None else sys.argv[1:]
    if not (effective_argv and effective_argv[0] == "doctor"):
        _warn_path_once("main")

    # Show version on --version
    if argv and argv[0] in ("--version", "-V", "version") and (len(argv) == 1 or argv[1].startswith("-")):
        try:
            from ssmforge import __version__
            print(f"ssmforge {__version__}")
            sys.exit(0)
        except ImportError:
            print("ssmforge (unknown version)")
            sys.exit(0)

    parser = argparse.ArgumentParser(
        prog="ssmforge",
        description=(
            "Architecture analyzer for HuggingFace models.\n\n"
            "Subcommands:\n"
            "  arch MODEL      Inspect a HuggingFace model — reports quirks like\n"
            "                  attention biases, fused QKV, MoE, sliding window,\n"
            "                  LayerScale, soft-capping, partial RoPE, MLP type,\n"
            "                  norm type. Outputs JSON or Markdown.\n"
            "  doctor          Print ssmforge + environment info (cache dir, transformers version, etc.)\n\n"
            "Try: ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V", action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- arch subcommand ----
    arch_p = subparsers.add_parser(
        "arch",
        help=(
            "Inspect a HuggingFace model's architecture and report quirks. "
            "Detects attention biases, fused tensors, MoE, sliding window, "
            "LayerScale, soft-capping, partial RoPE, MLP type, and norm type."
        ),
    )
    arch_p.add_argument(
        "source", nargs="?", default=None,
        help="HF model id or local path. Optional when --compare is used (the "
             "first model can come from --compare instead).",
    )
    arch_p.add_argument(
        "--output", "-o", default=None,
        help="Write report to this file (use '-' for stdout). Format chosen by --format.",
    )
    arch_p.add_argument(
        "--format", "-f", default="json", choices=["json", "markdown", "md"],
        help="Output format (default: json). markdown/md is GitHub-friendly.",
    )
    arch_p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary; print the report only",
    )
    arch_p.add_argument(
        "--diff", metavar="OTHER_MODEL", default=None,
        help="Compare against another model. Loads both and prints a diff report.",
    )
    arch_p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch only the config (no weight download). Reports config-only quirks "
             "and memory estimate. Useful as a pre-flight check.",
    )
    arch_p.add_argument(
        "--compare", metavar="MODEL", nargs="+", default=None,
        help="Compare 2+ models side-by-side. The first model comes from the "
             "`source` positional (or --compare if source is omitted). Renders "
             "a multi-model comparison table.",
    )
    arch_p.add_argument(
        "--rev", "--revision", dest="revision", default=None,
        help="Pin to a specific HF revision (commit sha, tag, or branch). "
             "Otherwise loads the current HEAD of the repo's default branch.",
    )
    arch_p.add_argument(
        "--fields", metavar="NAME", default=None,
        help="Subset the output to comma-separated field names (works with both "
             "single-model and --compare). Field names are top-level keys from the "
             "report (config.*, quirks.*, profile.*, etc). Use 'all' for everything.",
    )
    arch_p.add_argument(
        "--only-different", action="store_true",
        help="In --compare mode, suppress the 'Identical across all models' "
             "section. Default already shows only differing fields in the table, "
             "this flag additionally excludes the identical summary.",
    )
    arch_p.add_argument(
        "--profile", action="store_true",
        help="Emit only the profile section (family, attention_type, mlp_type, "
             "norm_type, descriptors). Useful for quick eyeball checks.",
    )
    arch_p.add_argument(
        "--graph", action="store_true",
        help="Render a text-based decision-graph of the model's architecture "
             "in your terminal. Shows the flow from input → embed → attn + MLP + "
             "norms → final norm → lm_head, with each architectural choice "
             "(fused QKV, GQA, SwiGLU vs GeLU, MoE, tied embeddings, etc.) "
             "listed inline with its verdict. With --compare, renders an "
             "N-way decision table.",
    )

    # ---- doctor subcommand ----
    doctor_p = subparsers.add_parser(
        "doctor",
        help="Print ssmforge + environment info (cache dir, transformers version, etc.)",
    )
    doctor_p.add_argument(
        "--format", "-f", default="text", choices=["text", "json"],
        help="Output format (default: text)",
    )
    doctor_p.add_argument(
        "--check-install", action="store_true",
        help="Run only the installation checks (PATH, scripts, venv) and "
             "print the result. Exits non-zero if anything is wrong so you "
             "can use it in scripts.",
    )

    args = parser.parse_args(argv)

    if args.command == "arch":
        # Suppress transformers' torch_dtype deprecation noise when user wants clean output
        if getattr(args, "quiet", False):
            _quiet_transformers_warnings()
        return _cmd_arch(args)
    elif args.command == "doctor":
        return _cmd_doctor(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_arch(args) -> None:
    """Handle the `arch` subcommand."""
    from ssmforge.analyze import (
        build_report,
        build_report_from_config,
        compare_reports,
        diff_reports,
        format_compare_markdown,
        format_report_json,
        format_report_markdown,
    )
    from ssmforge.analyze.summary import render_summary

    # Resolve the list of models to compare / inspect.
    # If --compare is given, models = [source?] + args.compare (deduped, preserving order)
    # If only --diff is given, models = [source, args.diff]
    # Otherwise models = [source]
    if args.compare:
        # Build the full model list: source first (if given), then --compare args
        all_models = []
        if args.source:
            all_models.append(args.source)
        for m in args.compare:
            if m not in all_models:
                all_models.append(m)
        # Need at least 2 for compare
        if len(all_models) < 2:
            print(
                "Error: --compare requires at least 2 models. "
                "Pass model ids as arguments after --compare, or "
                "provide the first as a positional argument.",
                file=sys.stderr,
            )
            sys.exit(1)
        args._compare_models = all_models
    elif args.diff:
        args._compare_models = [args.source, args.diff]
    else:
        args._compare_models = [args.source] if args.source else []

    # Fail fast on missing local paths
    for m in args._compare_models:
        _check_local_path(m)

    # ---- Multi-model dispatch (compare, diff, single) ----
    models_to_load = args._compare_models
    n_models = len(models_to_load)

    # Load all reports
    reports = []
    for model_id in models_to_load:
        try:
            with _arch_load_progress(
                model_id, dry_run=args.dry_run, revision=args.revision
            ) as obj:
                if args.dry_run:
                    r = build_report_from_config(model_id=model_id, config=obj)
                else:
                    r = build_report(
                        model_id=model_id,
                        config=obj.config,
                        state_dict=dict(obj.state_dict()),
                    )
                # Stamp the resolved HF revision on the report for traceability
                rev = getattr(obj, "ssmforge_revision", None)
                if rev:
                    r["hf_revision"] = rev
            reports.append(r)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except MemoryError:
            print(f"Error: out of memory while loading {model_id}", file=sys.stderr)
            if not args.dry_run:
                print("  Try --dry-run to preview without loading weights.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            err_name = type(e).__name__
            msg = str(e)
            if not args.dry_run and ("memory allocation" in msg.lower() or "out of memory" in msg.lower()):
                print(f"Error: insufficient memory to load {model_id}", file=sys.stderr)
                print(f"  Detail: {msg}", file=sys.stderr)
                print(f"  Try --dry-run to preview without loading weights.", file=sys.stderr)
                sys.exit(1)
            print(f"Error analyzing {model_id}: {err_name}: {msg}", file=sys.stderr)
            sys.exit(1)

    # Single model
    if n_models == 1:
        report_a = reports[0]
        # Capture compatibility before any subsetting (so we still have it for exit code)
        is_compat = report_a.get("compatibility", {}).get("is_compatible", True)
        # --profile emits just the profile section
        if args.profile:
            output_text = json.dumps(_profile_only_report(report_a), indent=2)
            _write_or_print(output_text, args.output)
            sys.exit(0 if is_compat else 2)
        # --graph emits the architecture decision graph
        if args.graph:
            from ssmforge.analyze.graph import render_graph_text
            output_text = render_graph_text(report_a)
            _write_or_print(output_text, args.output)
            sys.exit(0 if is_compat else 2)
        # --fields subsets the report
        if args.fields:
            report_a = _subset_report(report_a, args.fields)
        if args.format in ("markdown", "md"):
            output_text = format_report_markdown(report_a)
        else:
            output_text = format_report_json(report_a)
            if not args.quiet and not args.output:
                if args.dry_run:
                    print(_format_dry_run_summary(report_a), file=sys.stderr)
                else:
                    print(render_summary(report_a), file=sys.stderr)

        _write_or_print(output_text, args.output)
        # Capture compatibility BEFORE subsetting (in case --fields removed it)
        is_compat = report_a.get("compatibility", {}).get("is_compatible", True)
        sys.exit(0 if is_compat else 2)

    # Two models (legacy --diff behavior, preserved for back-compat)
    if n_models == 2 and not args.compare:
        report_a, report_b = reports
        if args.profile:
            # Emit profile for both, side by side
            output = {
                "models": [args._compare_models[0], args._compare_models[1]],
                "profiles": [_profile_only_report(report_a)["profile"],
                             _profile_only_report(report_b)["profile"]],
            }
            _write_or_print(json.dumps(output, indent=2), args.output)
            sys.exit(0)
        if args.graph:
            from ssmforge.analyze.graph import render_graph_compare
            output_text = render_graph_compare(reports)
            _write_or_print(output_text, args.output)
            sys.exit(0)
        if args.fields:
            report_a = _subset_report(report_a, args.fields)
            report_b = _subset_report(report_b, args.fields)
        diff = diff_reports(report_a, report_b)
        if args.format in ("markdown", "md"):
            md = _format_diff_markdown(report_a, report_b, diff)
            _write_or_print(md, args.output)
        else:
            output = {
                "model_a": args._compare_models[0],
                "model_b": args._compare_models[1],
                "identical": diff["identical"],
                "differences": diff["differences"],
                "added_quirks": diff["added_quirks"],
                "removed_quirks": diff["removed_quirks"],
            }
            _write_or_print(json.dumps(output, indent=2), args.output)
        sys.exit(0)

    # Multi-model comparison (3+ models, or --compare with 2 models)
    # --profile: emit profile sections for all models
    if args.profile:
        output = {
            "model_count": len(reports),
            "profiles": [
                {"model_id": r.get("model_id"), "profile": r.get("profile", {})}
                for r in reports
            ],
        }
        _write_or_print(json.dumps(output, indent=2), args.output)
        sys.exit(0)

    # --graph: emit N-way architecture decision table
    if args.graph:
        from ssmforge.analyze.graph import render_graph_compare
        output_text = render_graph_compare(reports)
        _write_or_print(output_text, args.output)
        sys.exit(0)

    # --fields: subset each report before comparison
    if args.fields:
        reports = [_subset_report(r, args.fields) for r in reports]

    comparison = compare_reports(reports)
    if args.format in ("markdown", "md"):
        md = format_compare_markdown(comparison)
        if args.only_different:
            # Strip the "Identical across all models" section from markdown
            lines = md.split("\n")
            filtered = []
            in_identical = False
            for line in lines:
                if line.startswith("### Identical across all models"):
                    in_identical = True
                    continue
                if in_identical:
                    # Skip until next blank line + non-indented content
                    if line.strip() == "":
                        in_identical = False
                    continue
                filtered.append(line)
            md = "\n".join(filtered).rstrip() + "\n"
        _write_or_print(md, args.output)
    else:
        # Add a small header so JSON consumers know it's a comparison
        output = {
            "comparison_type": "multi_model",
            "model_count": len(comparison["models"]),
            "all_identical": comparison["all_identical"],
            "identical_field_count": comparison["identical_field_count"],
            "different_field_count": comparison["different_field_count"],
            "models": comparison["models"],
            "fields": comparison["fields"],
        }
        if args.only_different:
            output["fields"] = [f for f in comparison["fields"] if not f["all_same"]]
        _write_or_print(json.dumps(output, indent=2), args.output)
    sys.exit(0)


def _cmd_doctor(args) -> None:
    """Handle the `doctor` subcommand."""
    import os as _os
    try:
        from ssmforge import __version__ as ssmforge_version
    except ImportError:
        ssmforge_version = "(unknown)"

    info = {
        "ssmforge_version": ssmforge_version,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "hf_home": _os.environ.get("HF_HOME", "(not set — using ~/.cache/huggingface)"),
        "hf_hub_cache": _os.environ.get("HF_HUB_CACHE", "(not set — using HF_HOME/hub)"),
        "hf_token_set": bool(_os.environ.get("HF_TOKEN")),
    }
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "(not installed)"
    try:
        import huggingface_hub
        info["huggingface_hub_version"] = huggingface_hub.__version__
    except ImportError:
        info["huggingface_hub_version"] = "(not installed)"

    # --check-install: run the install checks only, exit non-zero on issues.
    if args.check_install:
        on_path, fix = _check_ssmforge_on_path()
        result = {
            "ssmforge_command_on_path": on_path,
            "fix_hint": fix,
            "platform": sys.platform,
            "python_executable": sys.executable,
            "ssmforge_version": info["ssmforge_version"],
        }
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            mark = "✓" if on_path else "✗"
            print(f"ssmforge doctor --check-install ({info['ssmforge_version']})")
            print("-" * 50)
            print(f"  {mark} ssmforge command on PATH: {on_path}")
            print(f"  python: {sys.executable}")
            print(f"  platform: {sys.platform}")
            if not on_path and fix:
                print()
                print(fix)
        sys.exit(0 if on_path else 1)

    if args.format == "json":
        print(json.dumps(info, indent=2))
    else:
        print("ssmforge doctor")
        print("-" * 40)
        for k, v in info.items():
            print(f"  {k}: {v}")
    sys.exit(0)


def _get_version() -> str:
    try:
        from ssmforge import __version__
        return __version__
    except ImportError:
        return "(unknown)"


if __name__ == "__main__":
    main()
