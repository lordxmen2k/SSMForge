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
    no_distill: bool = False,
) -> ConversionResult:
    """Execute the pipeline (MVP: Stages 1-3 only)."""
    # Stage 1: Load
    model, state_dict = _load_model(source)
    arch = _detect_architecture(model)

    # Determine device once (CUDA if available, else CPU). Explicit (no implicit defaults).
    import torch as _torch
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            model = model.to(device)
        except Exception as e:
            print(f"Warning: could not move model to {device}: {e}. Falling back to CPU.")
            device = "cpu"

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

    if no_distill:
        # Skip Stage 4 entirely — write a fresh state dict without distillation.
        # Useful when distillation OOMs or segfaults and we just want the GGUF.
        stats["training_stats"] = {"skipped": True, "reason": "no-distill flag"}
        # Fall through to Stage 5 (export) without running distillation.

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

    # Skip distillation entirely when there are no SSM layers to distill.
    # This applies to recipes like pure-attention where the converter passes
    # all attention layers through unchanged. Distillation would be wasted
    # compute and could even degrade the model.
    if ssm_count == 0:
        stats["training_stats"] = {"skipped": True, "reason": "no SSM layers (pure-attention recipe)"}
        # For pure-attention we bypass the student model entirely. Building a
        # HybridLlamaMambaModel with empty ssm_layer_indices produces a Llama-
        # shaped model, but the source might be Qwen2 / Mistral / etc. whose
        # weights won't load cleanly into Llama layers. Instead, we use the
        # converter's output (target_sd) directly — it already has the source
        # weights copied verbatim for all attention layers.
        # Set student = None so Stage 5 falls back to target_sd.
        student = None
    elif tokenizer is not None and not no_distill:
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
            device=device,
        )
        try:
            trainer.train(num_steps=2)  # tiny for MVP
            stats["training_stats"] = {"final_loss": trainer.evaluate_loss()}
        except Exception as e:
            stats["training_stats"] = {"error": str(e), "skipped": True}
    elif no_distill:
        # Build the student but skip training. Needed so Stage 5 (export) has a model.
        if tokenizer is not None:
            student = HybridLlamaMambaModel(config)
            student.load_state_dict(target_sd, strict=False)
        stats["training_stats"] = {"skipped": True, "reason": "no-distill flag"}
    else:
        stats["training_stats"] = {"skipped": True, "reason": "no tokenizer"}

    # Stage 5: Export
    from datetime import datetime
    from ssmforge.export.gguf_writer import write_f16_gguf
    from ssmforge.export.llama_quantize import LlamaQuantizer
    from ssmforge.export.manifest import Manifest, write_manifest, compute_file_sha

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = source.replace("/", "_").replace("\\", "_")[-64:]
    f16_path = output_dir / f"{safe_name}.{recipe.upper().replace('-','')}.{quantize}.f16.gguf"
    final_path = output_dir / f"{safe_name}.{recipe.upper().replace('-','')}.{quantize}.gguf"

    try:
        if 'student' in locals() and student is not None:
            final_sd = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        else:
            final_sd = {k: (v.detach().cpu() if hasattr(v, 'detach') else v) for k, v in target_sd.items()}
        write_f16_gguf(
            model_state_dict=final_sd,
            config=config,
            tokenizer=tokenizer,
            output_path=f16_path,
        )

        if quantize != "F16":
            try:
                quantizer = LlamaQuantizer()
                quantizer.quantize(f16_path, final_path, quantize)
                if f16_path.exists():
                    f16_path.unlink()
            except Exception as e:
                # Quantize binary may not be installed; fall back to F16
                final_path = f16_path
                stats["quantization_note"] = f"Quantize failed ({e}); kept F16 GGUF"
        else:
            final_path = f16_path

        output_sha = compute_file_sha(final_path) if final_path.exists() else "unknown"

        manifest = Manifest(
            ssmforge_version="0.2.0",
            source_model=source,
            source_revision="unknown",
            recipe=recipe,
            quant_type=quantize if final_path != f16_path else "F16",
            layer_mapping=[
                {"index": spec.index, "layer_type": spec.layer_type.value}
                for spec in plan
            ],
            calibration_data_sha=None,
            training_stats=stats.get("training_stats"),
            output_gguf_path=str(final_path),
            output_gguf_sha=output_sha,
            output_gguf_bytes=final_path.stat().st_size if final_path.exists() else 0,
            created_at=datetime.now(),
        )
        manifest_path = output_dir / f"{safe_name}.{recipe.upper().replace('-','')}.{quantize}.manifest.json"
        write_manifest(manifest, manifest_path)

        return ConversionResult(
            gguf_path=final_path,
            manifest_path=manifest_path,
            stats=stats,
        )
    except Exception as e:
        return ConversionResult(
            gguf_path=None,
            manifest_path=None,
            stats={**stats, "export_error": str(e)},
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
    no_distill: bool = False,
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
        no_distill=no_distill,
    )
