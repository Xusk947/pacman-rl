# Pacman RL (Multi-Agent, Torch-Vectorized)

This project implements a multi-agent Pacman environment where:

- Pacman is controlled by an RL agent.
- Each ghost is controlled by an RL agent (shared policy by default).
- The environment is batched and tensorized with PyTorch to run efficiently on GPU.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .

python -m pacman_rl.train --layout-dir layouts --device auto
```

## Telegram notifications

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then run with `--telegram`:

```bash
python -m pacman_rl.train --layout-dir layouts --device auto --telegram
```

The trainer sends a “training started” message and a final summary message. Every `--report-every` updates it also sends a `bundle.zip` with checkpoint + telemetry + a recorded game.

## Project layout

```text
src/pacman_rl/
  env/
  layouts/
  models/
  rl/
  train.py
layouts/
  basic.txt
```
