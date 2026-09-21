from pathlib import Path

import pytest
from ssmforge.distillation import CalibrationDataLoader
from ssmforge.exceptions import CalibrationDataError


def test_calibration_loads_text_file(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello world\nthis is a test\nthird line\n")
    loader = CalibrationDataLoader(source=str(f), max_samples=2)
    samples = loader.load()
    assert len(samples) == 2
    assert "hello" in samples[0]


def test_calibration_uses_builtin_when_source_is_none():
    loader = CalibrationDataLoader(source=None, max_samples=10)
    samples = loader.load()
    assert len(samples) == 10
    assert all(isinstance(s, str) for s in samples)


def test_calibration_handles_missing_file():
    loader = CalibrationDataLoader(source="/nonexistent/path/abc123", max_samples=5)
    with pytest.raises(CalibrationDataError):
        loader.load()


def test_calibration_handles_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    loader = CalibrationDataLoader(source=str(f), max_samples=5)
    with pytest.raises(CalibrationDataError):
        loader.load()


def test_calibration_respects_max_samples(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("\n".join(f"line {i}" for i in range(100)))
    loader = CalibrationDataLoader(source=str(f), max_samples=10)
    samples = loader.load()
    assert len(samples) == 10
