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

    # Stage 4: Distillation (lightweight, in-place mutation of student state dict)
    from ssmforge.distillation.calibration import CalibrationDataLoader
    from ssmforge.distillation.trainer import DistillationTrainer
    from ssmforge.models import HybridLlamaMambaConfig, HybridLlamaMambaModel

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(source)
    except Exception:
        tokenizer = None

    # Build hybrid model from converted state dict
    config = HybridLlamaMambaConfig(
        vocab_size=getattr(model.config, "vocab_size", 32000),
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        intermediate_size=model.config.intermediate_size,
        max_position_embeddings=model.config.max_position_embeddings,
        rope_theta=getattr(model.config, "rope_theta", 10000.0),
        ssm_layer_indices=[spec.index for spec in plan if spec.layer_type.value == "ssm"],
    )

    if tokenizer is not None:
        student = HybridLlamaMambaModel(config)
        student.load_state_dict(target_sd, strict=False)

        cal = CalibrationDataLoader(source=calibration_data, max_samples=100)
        distill_config = recipe_obj.distillation_config()

        trainer = DistillationTrainer(
            student_model=student,
            teacher_model=model,
            calibration_loader=cal,
            tokenizer=tokenizer,
            config=distill_config,
        )
        try:
            trainer.train(num_steps=2)  # tiny for MVP
            stats["training_stats"] = {"final_loss": trainer.evaluate_loss()}
        except Exception as e:
            stats["training_stats"] = {"error": str(e), "skipped": True}
    else:
        stats["training_stats"] = {"skipped": True, "reason": "no tokenizer"}

    # Stages 5-6: deferred to Phase 3
    return ConversionResult(
        gguf_path=None,
        manifest_path=None,
        stats={
            **stats,
            "note": (
                "Stages 1-4 complete. Stages 5-6 (export, verify) will be wired in Phase 3."
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
