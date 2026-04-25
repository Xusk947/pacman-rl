from __future__ import annotations

import unittest

import torch

from pacman_rl.config import EnvConfig
from pacman_rl.env.torch_env import TorchPacmanEnv
from pacman_rl.layouts.parser import parse_layout_text


class TestEnvFrightened(unittest.TestCase):
    def test_power_pellet_makes_ghost_eatable(self) -> None:
        lay = parse_layout_text(
            "\n".join(
                [
                    "#####",
                    "#0oB#",
                    "#P I#",
                    "#C..#",
                    "#####",
                ]
            ),
            name="t",
        )
        cfg = EnvConfig(frightened_steps=5, reward_power=5.0, reward_ghost_eat=10.0, max_steps=50)
        env = TorchPacmanEnv([lay], batch_size=1, device=torch.device("cpu"), cfg=cfg)
        env.reset(seed=0)

        actions = torch.tensor([[3, 2, 4, 4, 4]], dtype=torch.int64)
        out = env.step(actions)

        self.assertFalse(bool(out.done.item()))
        self.assertEqual(int(out.info["pac_dead"].item()), 0)
        self.assertGreaterEqual(float(out.reward[0, 0].item()), 14.0)

        ghost_b = env.ghosts[0, 0].tolist()
        home_b = env.ghosts_home[0, 0].tolist()
        self.assertEqual(ghost_b, home_b)


if __name__ == "__main__":
    unittest.main()

