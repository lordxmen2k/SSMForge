"""Top-level pipeline orchestrator for SSMForge.

6-stage pipeline:
  1. Load       — HF model + tokenizer + state dict
  2. Recipe     — plan layer types per recipe
  3. Surgery    — state dict manipulation
  4. Distill    — KL divergence training (deferred to Phase 2)
  5. Export     — F16 GGUF + llama-quantize (deferred to Phase 3)
  6. Verify     — load GGUF, check forward pass (deferred to Phase 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from ssmforge.recipes import get_recipe
from ssmforge.converters import ArchitectureConverterRegistry
from ssmforge.result import ConversionResult
from ssmforge.exceptions import (
    ModelNotFoundError,
    UnsupportedArchitectureError,
    RecipeArchitectureMismatchError,
)


def _load_model(source: str) -> tuple[Any, dict]:
    """Load a HuggingFace model and its state dict.

    Returns: (model, state_dict)
    """
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype="auto")
    except Exception as e:
        msg = str(e).lower()
        if "repository not found" in msg or "404" in msg or "could not find" in msg:
            raise ModelNotFoundError(model_id=source) from e
        raise
    state_dict = dict(model.state_dict())
    state_dict["_hidden_size"] = model.config.hidden_size
    return model, state_dict


def _detect_architecture(model: Any) -> str:
    """Detect the model architecture string from HF config."""
    arch = getattr(model.config, "model_type", None)
    if arch is None:
        arch_list = getattr(model.config, "architectures", ["unknown"])
        arch = arch_list[0].lower()
    return arch


def _run_pipeline(
    source: str,
    recipe: str,
    quantize: str,
    output_dir: Path,
    calibration_data: Optional[str] = None,
    verify: bool = False,
    dry_run: bool = False,
    experimental: bool = False,
) -> ConversionResult:
    """Execute the pipeline (MVP: Stages 1-3 only)."""
    # Stage 1: Load
    model, state_dict = _load_model(source)
    arch = _detect_architecture(model)

    # Stage 2: Recipe plan
    recipe_obj = get_recipe(recipe)
    if recipe_obj.name == "pure-mamba" and not experimental:
        raise RecipeArchitectureMismatchError(
            recipe=recipe,
            arch=arch,
            explanation="The pure-mamba recipe is experimental. Pass --experimental to enable.",
            next_steps=[
                "Use 'hybrid-25' or 'hybrid-50' for production.",
                "Or pass --experimental to enable pure-mamba.",
            ],
        )
    plan = recipe_obj.plan(model)

    # Stage 3: Architecture surgery
    converter = ArchitectureConverterRegistry.get(arch)
    target_sd = converter.convert_state_dict(state_dict, plan)

    ssm_count = sum(1 for spec in plan if spec.layer_type.value == "ssm")
    stats = {
        "layer_count": len(plan),
        "ssm_count": ssm_count,
        "attention_count": len(plan) - ssm_count,
        "recipe": recipe,
        "quantize": quantize,
        "arch": arch,
        "dry_run": dry_run,
        "layer_mapping": [
            {"index": spec.index, "layer_type": spec.layer_type.value}
            for spec in plan
        ],
    }

    if dry_run:
        return ConversionResult(gguf_path=None, manifest_path=None, stats=stats)

    # Stages 4-6: not yet implemented in MVP
    return ConversionResult(
        gguf_path=None,
        manifest_path=None,
        stats={
            **stats,
            "note": (
                "Stages 1-3 complete. Stages 4-6 (distillation, export, verify) "
                "will be wired in subsequent phases. Use --dry-run to skip this notice."
            ),
        },
    )


def convert(
    source: str,
    recipe: str = "hybrid-25",
    quantize: str = "Q4_K_M",
    output_dir: Path | str = "./out",
    calibration_data: Optional[str] = None,
    verify: bool = False,
    dry_run: bool = False,
    experimental: bool = False,
) -> ConversionResult:
    """Convert a pretrained model to a hybrid SSM/attention model + quantized GGUF.

    See docs/quickstart.md for usage examples.
    """
    return _run_pipeline(
        source=source,
        recipe=recipe,
        quantize=quantize,
        output_dir=Path(output_dir),
        calibration_data=calibration_data,
        verify=verify,
        dry_run=dry_run,
        experimental=experimental,
    )
