from __future__ import annotations

from pathlib import Path


def pick_device(requested: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    def _has_nvidia_device_files() -> bool:
        return Path("/dev/nvidiactl").exists() or Path("/dev/nvidia0").exists()

    def _cuda_works() -> bool:
        try:
            if not _has_nvidia_device_files():
                return False
            if not torch.cuda.is_available():
                return False
            _ = torch.cuda.device_count()
            x = torch.randn((8, 8), device="cuda", requires_grad=True)
            y = (x @ x).sum()
            y.backward()
            return True
        except Exception:
            return False

    def _mps_works() -> bool:
        try:
            if not hasattr(torch.backends, "mps"):
                return False
            if not torch.backends.mps.is_available():
                return False
            _ = torch.tensor([0.0], device="mps")
            return True
        except Exception:
            return False

    if requested == "cuda":
        return "cuda" if _cuda_works() else "cpu"
    if requested == "mps":
        return "mps" if _mps_works() else "cpu"
    if requested == "cpu":
        return "cpu"

    if _cuda_works():
        return "cuda"
    if _mps_works():
        return "mps"
    return "cpu"
