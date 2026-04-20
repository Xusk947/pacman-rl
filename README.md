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

Telegram sends periodic status and artifacts (see `--report-every`, `--telegram-send-recordings`).

Demo recording can be capped via `--record-max-steps` and it will stop early if the game is idle for too long (`--record-idle-steps`).

## Postgres (Neon) logging

You can stream episode-level telemetry into Postgres (e.g. Neon) using a connection string (`postgresql://...`).

1. Install a Postgres driver:

```bash
pip install psycopg[binary]
```

2. Provide the connection string as a secret or environment variable:

- Local: set `DATABASE_URL` in `.env`
- Kaggle: add a secret named `DATABASE_URL` in the Notebook "Secrets" UI

3. Run with:

```bash
python -m pacman_rl.train --layout-dir layouts --device auto --postgres-url-env DATABASE_URL
```

If you want to force-disable Postgres logging even when `DATABASE_URL` is set, use `--no-postgres`.

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
