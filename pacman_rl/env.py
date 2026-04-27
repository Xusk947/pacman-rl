from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable


def make_pacman_env(env_id: str, *, seed: int, render_mode: str = "rgb_array") -> Callable[[], "gym.Env"]:
    def _make() -> "gym.Env":
        try:
            import gymnasium as gym
        except Exception as e:
            raise RuntimeError("gymnasium is required") from e

        if env_id.startswith("ALE/"):
            try:
                import ale_py

                gym.register_envs(ale_py)
                _ensure_ale_roms(ale_py)
            except Exception as e:
                raise RuntimeError("ALE environments are not available. Install ale-py and ROMs (AutoROM).") from e

        try:
            from stable_baselines3.common.monitor import Monitor
        except Exception as e:
            raise RuntimeError("stable-baselines3 is required") from e

        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed)
        return Monitor(env)

    return _make


def _ensure_ale_roms(ale_py_module: object) -> None:
    ale_py_roms_dir = Path(getattr(ale_py_module, "roms").__file__).parent
    pacman_bin = ale_py_roms_dir / "pacman.bin"
    if pacman_bin.exists():
        return

    try:
        import AutoROM
    except Exception:
        return

    autorom_roms_dir = Path(AutoROM.__file__).parent / "roms"
    if not autorom_roms_dir.exists():
        return

    bins = list(autorom_roms_dir.glob("*.bin"))
    if not bins:
        return

    ale_py_roms_dir.mkdir(parents=True, exist_ok=True)
    for src in bins:
        dst = ale_py_roms_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
