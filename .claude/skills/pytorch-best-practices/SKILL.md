---
name: pytorch-best-practices
description: Apply PyTorch best practices for fast, stable GPU training and inference (DataLoader tuning, pinned memory, AMP, torch.compile, profiling, reproducibility, checkpointing). Use when writing or reviewing PyTorch code, training loops, model optimization, or when the user mentions "GPU", "AMP", "torch.compile", "DataLoader", "pin_memory", "num_workers", "reproducibility", or "checkpoint".
---

# PyTorch Best Practices

This Skill guides you in writing correct and high-performance PyTorch code, focusing on GPU efficiency, stable training loops, and reproducibility.

## Sources (primary)

- PyTorch Performance Tuning Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- PyTorch Reproducibility notes: https://pytorch.org/docs/main/notes/randomness.html

## Quick start

When asked to write or optimize a PyTorch training loop:

1. Confirm target device(s) and PyTorch version.
2. Ensure data pipeline doesn’t bottleneck the GPU.
3. Use AMP where safe and measure speed/accuracy.
4. Avoid accidental CPU↔GPU sync points.
5. Add solid checkpointing + deterministic settings when required.

## Instructions

### 1) Device and dtype hygiene

- Move model and tensors to the same device explicitly.
- Avoid repeated `.to(device)` calls inside inner loops.
- Prefer creating tensors directly on the target device when possible.

### 2) Data loading for GPU throughput

- Use `DataLoader(..., pin_memory=True)` when training on CUDA.
- Tune `num_workers` per machine; avoid assuming a single “best” value.
- Use `persistent_workers=True` for long runs (when `num_workers>0`).
- Use `prefetch_factor` (when applicable) to reduce stalls.
- Transfer batches with `non_blocking=True` when using pinned memory.

### 3) Training loop correctness

- Call `model.train()` for training and `model.eval()` for evaluation.
- Use `optimizer.zero_grad(set_to_none=True)` for better performance.
- Clip gradients if training is unstable (configurable).
- Keep step logic explicit (forward → loss → backward → step).

### 4) Mixed precision (AMP)

- Use `torch.autocast(device_type="cuda", dtype=torch.float16 or torch.bfloat16)` where supported.
- Use `torch.cuda.amp.GradScaler` for FP16 training stability.
- Validate numerics (NaNs/inf) and final metrics; keep AMP optional via config.

### 5) Performance pitfalls to avoid

- Avoid `.item()` / `.cpu()` / converting tensors to Python numbers inside the hot path.
- Avoid frequent `print` in the step loop.
- Accumulate metrics in tensors and reduce/log periodically.

### 6) Compilation and profiling

- Consider `torch.compile(model)` for speedups (validate correctness and warm-up).
- Use `torch.profiler` to identify CPU stalls, data-transfer bottlenecks, and kernel inefficiencies.

### 7) Reproducibility (when requested)

- Set seeds for Python/NumPy/PyTorch and document them.
- Enable deterministic algorithms only when needed (can slow down).
- Record versions (PyTorch/CUDA/cuDNN) in checkpoints/metadata.

### 8) Checkpointing and resume

- Save:
  - `model.state_dict()`
  - `optimizer.state_dict()`
  - scaler state (if AMP)
  - scheduler state (if used)
  - RNG states (when reproducibility matters)
- Make resume path a first-class workflow.

## Output format

When applying this Skill, produce:

1. A short checklist of enabled performance features (pinned memory, AMP, compile, etc.).
2. A minimal, correct training loop (or a diff-based review).
3. A benchmarking/profiling plan (what to measure, how to detect bottlenecks).
4. A checkpoint/resume snippet consistent with the project.
