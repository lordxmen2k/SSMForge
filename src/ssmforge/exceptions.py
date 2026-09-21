"""Typed exception hierarchy for SSMForge.

Every error provides:
- A one-line summary
- Plain-English explanation
- Numbered list of next steps
- Link to relevant docs
- --debug flag mention for full traceback
"""

from __future__ import annotations


class SSMForgeError(Exception):
    """Base for all SSMForge errors. All other exceptions inherit from this."""

    error_type: str = "SSMForgeError"

    def __init__(self, **context):
        self.context = context
        super().__init__(self.format_message())

    def format_message(self) -> str:
        lines = [
            f"[SSMForge] {self.error_type}: {self.summary()}",
            f"  What happened: {self.explanation()}",
            "  What you can do:",
        ]
        for i, step in enumerate(self.next_steps(), 1):
            lines.append(f"    {i}. {step}")
        lines.append(f"  Docs: {self.docs_url()}")
        lines.append("  Run with --debug for full traceback.")
        return "\n".join(lines)

    def summary(self) -> str:
        return self.context.get("summary", "An error occurred")

    def explanation(self) -> str:
        return self.context.get("explanation", "Something went wrong.")

    def next_steps(self) -> list[str]:
        return self.context.get("next_steps", ["Check the error message above."])

    def docs_url(self) -> str:
        return self.context.get("docs_url", "https://github.com/lordxmen2k/SSMForge")


class SourceModelError(SSMForgeError):
    error_type = "SourceModelError"


class ModelNotFoundError(SourceModelError):
    error_type = "ModelNotFoundError"

    def summary(self):
        return f"Could not find model '{self.context.get('model_id')}'"

    def explanation(self):
        return (
            f"The model id '{self.context.get('model_id')}' could not be loaded. "
            "Either it doesn't exist on HuggingFace Hub, or the local path is wrong, "
            "or you need to authenticate."
        )

    def next_steps(self):
        steps = []
        if self.context.get("model_id", "").count("/") == 1:
            steps.append("Run `huggingface-cli login` if the model is gated.")
            steps.append("Verify the model id at https://huggingface.co/<model_id>")
        steps.append("If using a local path, verify the directory exists and contains config.json.")
        steps.append("Run with --debug to see the underlying error.")
        return steps


class UnsupportedArchitectureError(SourceModelError):
    error_type = "UnsupportedArchitectureError"

    def summary(self):
        return f"Architecture '{self.context.get('arch')}' not supported"

    def explanation(self):
        return (
            f"SSMForge v{self.context.get('version', '0.1.0')} does not support "
            f"the '{self.context.get('arch')}' architecture."
        )

    def next_steps(self):
        return [
            f"Supported architectures: {', '.join(self.context.get('supported', []))}",
            "See docs/architecture.md for adding new converters.",
            "Or open an issue: https://github.com/lordxmen2k/SSMForge/issues",
        ]


class RecipeError(SSMForgeError):
    error_type = "RecipeError"


class UnknownRecipeError(RecipeError):
    error_type = "UnknownRecipeError"

    def summary(self):
        return f"Recipe '{self.context.get('recipe')}' is not registered"

    def explanation(self):
        return "The recipe name you specified has not been registered with SSMForge."

    def next_steps(self):
        return [
            f"Registered recipes: {', '.join(self.context.get('registered', []))}",
            "See docs/recipes.md for the recipe catalog.",
            "Register a custom recipe via ssmforge.recipes.register_recipe().",
        ]


class RecipeArchitectureMismatchError(RecipeError):
    error_type = "RecipeArchitectureMismatchError"

    def summary(self):
        return f"Recipe '{self.context.get('recipe')}' not compatible with '{self.context.get('arch')}'"

    def explanation(self):
        return self.context.get("explanation", "This recipe requires architecture features the model lacks.")

    def next_steps(self):
        return self.context.get("next_steps", ["Try a different recipe.", "Check docs/recipes.md."])


class ConversionError(SSMForgeError):
    error_type = "ConversionError"


class WeightShapeError(ConversionError):
    error_type = "WeightShapeError"

    def summary(self):
        return (
            f"Cannot project weight '{self.context.get('weight_name')}' from "
            f"{self.context.get('src_shape')} to {self.context.get('tgt_shape')}"
        )

    def explanation(self):
        return "The attention weight shape cannot be projected to the corresponding Mamba2 weight shape."

    def next_steps(self):
        return [
            "Verify the model config (hidden_size, num_heads, etc.) matches expectations.",
            "Run with --debug-surgery for per-layer diagnostics.",
        ]


class LayerMappingError(ConversionError):
    error_type = "LayerMappingError"

    def summary(self):
        return f"Cannot satisfy layer plan: {self.context.get('reason')}"

    def explanation(self):
        return "The recipe's planned layer mapping cannot be applied to this model."

    def next_steps(self):
        return ["Try a different recipe.", "File an issue with the model config."]


class RoundTripMismatchError(ConversionError):
    error_type = "RoundTripMismatchError"

    def summary(self):
        return f"Forward-pass mismatch exceeds tolerance (max diff: {self.context.get('max_diff', 'unknown')})"

    def explanation(self):
        return "The hybrid model's forward pass deviates from the teacher beyond acceptable tolerance."

    def next_steps(self):
        return [
            "Run with --debug-surgery to inspect per-layer diffs.",
            "Verify the model loaded correctly (no missing weights).",
        ]


class DistillationError(SSMForgeError):
    error_type = "DistillationError"


class CalibrationDataError(DistillationError):
    error_type = "CalibrationDataError"

    def summary(self):
        return f"Calibration data unusable: {self.context.get('reason')}"

    def explanation(self):
        return "The calibration data source could not be loaded or produces empty/insufficient batches."

    def next_steps(self):
        return [
            "Check the data source path or HF dataset id.",
            "Use --calibration-data <path> to specify a custom source.",
            "Or omit the flag to use the built-in 1M-token default set.",
        ]


class TrainingDivergenceError(DistillationError):
    error_type = "TrainingDivergenceError"

    def summary(self):
        return f"Training diverged at step {self.context.get('step')} (loss: {self.context.get('loss')})"

    def explanation(self):
        return "The student model's loss exploded or NaN'd, indicating an unstable training configuration."

    def next_steps(self):
        return [
            "Reduce the learning rate (try 1e-5 instead of 5e-5).",
            "Check that calibration data is in the same distribution as the teacher's pretraining.",
            "Resume from the last good checkpoint with --resume-from-checkpoint.",
        ]


class CheckpointError(DistillationError):
    error_type = "CheckpointError"

    def summary(self):
        return f"Cannot {self.context.get('action')} checkpoint at {self.context.get('path')}"

    def next_steps(self):
        return ["Check disk space.", "Check write permissions on the output directory."]


class ExportError(SSMForgeError):
    error_type = "ExportError"


class LlamaQuantizeNotFoundError(ExportError):
    error_type = "LlamaQuantizeNotFoundError"

    def summary(self):
        return "Could not locate llama-quantize binary"

    def explanation(self):
        return "SSMForge shells out to llama-quantize for GGUF quantization, but no binary was found."

    def next_steps(self):
        return [
            "Install llama-cpp-python: pip install llama-cpp-python",
            "Or build llama.cpp from source and set LLAMA_QUANTIZE_BIN=/path/to/llama-quantize",
        ]


class QuantizationFailedError(ExportError):
    error_type = "QuantizationFailedError"

    def summary(self):
        return f"Quantization failed (exit code {self.context.get('exit_code')})"

    def explanation(self):
        return "The llama-quantize subprocess returned a non-zero exit code."

    def next_steps(self):
        return [
            f"stderr: {self.context.get('stderr', '<empty>')}",
            "Verify the F16 GGUF is valid by loading it with gguf-py.",
            "Try a different quantization type (F16 is a safe fallback).",
        ]


class GGUFWriteError(ExportError):
    error_type = "GGUFWriteError"

    def summary(self):
        return f"Failed to write GGUF: {self.context.get('reason')}"

    def next_steps(self):
        return ["Check disk space.", f"Output path: {self.context.get('path')}", "Check write permissions."]


class VerificationError(SSMForgeError):
    """Warning by default — does not fail the run unless --strict-verify."""

    error_type = "VerificationError"

    def summary(self):
        return f"Verification inconclusive: {self.context.get('reason')}"
