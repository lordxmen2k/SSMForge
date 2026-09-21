"""Result type returned by convert()."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


class ConversionResult(BaseModel):
    gguf_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    stats: dict[str, Any] = {}
