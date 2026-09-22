"""Pytest configuration for SSMForge tests.

Pins pytest basetemp (via --basetemp CLI option in pyproject.toml) and
points HF_HOME at a tests-only cache directory so test downloads don't
pollute the user's real HF cache.

Override behavior:
- SSMFORGE_TEST_HF_HOME=DIR  → use DIR as the test HF cache
- (unset)                    → use ./tests/.cache/ relative to repo root
- HF_HOME already set by user → respect it (don't override user config)
"""

import os
from pathlib import Path


# Resolve test HF cache directory
_explicit_test_hf_home = os.environ.get("SSMFORGE_TEST_HF_HOME")
_user_hf_home = os.environ.get("HF_HOME")

if _explicit_test_hf_home:
    # User explicitly asked for a test cache path
    _test_hf_home = _explicit_test_hf_home
elif _user_hf_home:
    # User has HF_HOME set in their env — respect it for tests too
    # (don't override what they configured for themselves)
    _test_hf_home = _user_hf_home
else:
    # Default: tests/.cache/ relative to repo root
    _test_hf_home = str(Path(__file__).resolve().parent.parent / "tests" / ".cache")

# Ensure directory exists and set env vars
Path(_test_hf_home).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = _test_hf_home
os.environ["TRANSFORMERS_CACHE"] = _test_hf_home
os.environ["HF_HUB_CACHE"] = _test_hf_home
