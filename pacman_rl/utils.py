from __future__ import annotations

from pathlib import Path


def pick_device(requested: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    def _has_nvidia_device_files() -> bool:
        return Path("/dev/nvidiactl").exists() or Path("/dev/nvidia0").exists()

    def _cuda_conv_works() -> bool:
        try:
            if not _has_nvidia_device_files():
                return False
            if not torch.cuda.is_available():
                return False
            _ = torch.cuda.device_count()

            import torch.nn as nn

            x = torch.randn((1, 4, 84, 84), device="cuda", dtype=torch.float32)
            conv = nn.Conv2d(4, 32, kernel_size=8, stride=4).to(device="cuda", dtype=torch.float32)
            _ = conv(x)
            return True
        except Exception as e:
            msg = str(e)
            if "unable to find an engine" in msg.lower():
                try:
                    torch.backends.cudnn.enabled = False
                    import torch.nn as nn

                    x = torch.randn((1, 4, 84, 84), device="cuda", dtype=torch.float32)
                    conv = nn.Conv2d(4, 32, kernel_size=8, stride=4).to(device="cuda", dtype=torch.float32)
                    _ = conv(x)
                    print("CUDA conv engine failed; running with cuDNN disabled.", flush=True)
                    return True
                except Exception:
                    return False
            return False

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
            return _cuda_conv_works()
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


def parse_int_tuple(text: str) -> tuple[int, ...]:
    raw = str(text or "").replace(";", ",").replace(" ", ",")
    out: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            continue
    return tuple(out)


def set_global_seeds(seed: int) -> None:
    try:
        import random

        random.seed(int(seed))
    except Exception:
        pass

    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
