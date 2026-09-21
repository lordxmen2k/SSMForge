import pytest
from ssmforge.exceptions import (
    SSMForgeError,
    SourceModelError,
    ModelNotFoundError,
    UnsupportedArchitectureError,
    RecipeError,
    UnknownRecipeError,
    RecipeArchitectureMismatchError,
    ConversionError,
    WeightShapeError,
    LayerMappingError,
    RoundTripMismatchError,
    DistillationError,
    CalibrationDataError,
    TrainingDivergenceError,
    CheckpointError,
    ExportError,
    LlamaQuantizeNotFoundError,
    QuantizationFailedError,
    GGUFWriteError,
    VerificationError,
)


def test_all_subclasses_inherit_from_base():
    subclasses = [
        ModelNotFoundError, UnsupportedArchitectureError,
        UnknownRecipeError, RecipeArchitectureMismatchError,
        WeightShapeError, LayerMappingError, RoundTripMismatchError,
        CalibrationDataError, TrainingDivergenceError, CheckpointError,
        LlamaQuantizeNotFoundError, QuantizationFailedError, GGUFWriteError,
        VerificationError,
    ]
    for cls in subclasses:
        assert issubclass(cls, SSMForgeError)
        assert issubclass(
            cls,
            (SourceModelError, RecipeError, ConversionError, DistillationError, ExportError, VerificationError),
        )


def test_error_message_has_template():
    err = ModelNotFoundError(model_id="foo/bar")
    msg = str(err)
    assert "foo/bar" in msg
    assert "What you can do:" in msg
    assert "Docs:" in msg


def test_base_subclass_relationships():
    assert issubclass(SourceModelError, SSMForgeError)
    assert issubclass(RecipeError, SSMForgeError)
    assert issubclass(ConversionError, SSMForgeError)
    assert issubclass(DistillationError, SSMForgeError)
    assert issubclass(ExportError, SSMForgeError)
    assert issubclass(VerificationError, SSMForgeError)
