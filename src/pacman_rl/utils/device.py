from __future__ import annotations

import torch


def cuda_compatibility() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "cuda_not_available"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
    except Exception as e:
        return False, f"cuda_capability_error={e}"

    arch_list = None
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            arch_list = set(torch.cuda.get_arch_list())
        except Exception:
            arch_list = None

    if arch_list is not None and arch not in arch_list:
        return False, f"unsupported_arch={arch} supported={sorted(arch_list)}"

    try:
        x = torch.empty((1,), device="cuda")
        x.add_(1)
    except Exception as e:
        return False, f"cuda_runtime_error={e}"

    return True, "ok"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        ok, _ = cuda_compatibility()
        return torch.device("cuda" if ok else "cpu")
    return torch.device(device)
