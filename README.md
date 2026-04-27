# pacman-rl

CLI that trains `gymnasium.make("ALE/Pacman-v5")` with 3 algorithms (PPO, A2C, DQN) and logs episode + training metrics into SQLite.

## Kaggle

1. Create a Kaggle Notebook with Internet enabled and (optionally) GPU enabled.
2. Add secrets or environment variables:
   - `BOT_TOKEN` (Telegram bot token)
   - `USER_ID` (your chat id)
3. In a notebook cell:

```bash
!git clone https://github.com/Xusk947/pacman-rl.git
%cd pacman-rl
!make runenv
```

If `BOT_TOKEN` and `USER_ID` are present, the training will send an editable progress message to Telegram and, after finishing, send videos + SQLite DB.

## Install

Atari environments require ROM support. The simplest path is:

On Arch/CachyOS you also need the HDF5 runtime library for `ale-py`:

```bash
sudo pacman -S --needed hdf5 vtk ffmpeg
```

```bash
pip install -r requirements.txt
pip install "gymnasium[atari]>=0.29.1,<1.3.0" "autorom[accept-rom-license]"
AutoROM --accept-license
```

PyTorch GPU wheels are installed via the official PyTorch instructions for your CUDA version.

## Run

```bash
pacman-rl-train --db runs.sqlite --total-timesteps 200000 --algos ppo a2c dqn
```

`ALE/Pacman-v5` rewards are scaled (pellet reward is typically `1.0`), so the default `--win-score-threshold` is `500`.

The database will contain:

- `runs`: one row per algorithm run
- `episode_metrics`: per-episode metrics (return, win, pellets/power-pellets/ghosts, percent-cleared)
- `training_metrics`: periodic training metrics snapshots as JSON

Timestamps are stored as ISO-8601 UTC strings with timezone (example: `2026-04-27T12:34:56.789Z`).

## Progress output

The trainer prints a short progress line every 5% by default:

```bash
pacman-rl-train --print-every-percent 5 --stats-window-episodes 100
```

## Visualization

You can record MP4 videos during training:

```bash
pacman-rl-train --record-video-dir videos --video-trigger-steps 50000 --video-length 1800
```

Rendering live (`render_mode="human"`) is possible but slows training a lot and requires a working desktop/SDL.

### After training (playback)

Live:

```bash
pacman-rl-play --model models/<run_id>_ppo.zip --render human --device cuda
```

Headless (no window):

```bash
pacman-rl-play --model models/<run_id>_ppo.zip --render rgb_array --episodes 1 --max-steps 2000
```

Record video during playback:

```bash
pacman-rl-play --model models/<run_id>_ppo.zip --render rgb_array --record-video-dir videos_play
```

### Report (videos + plots)

After training you can generate one playback video per finished run and plot PNG graphs from `training_metrics`:

```bash
pacman-rl-report --db runs.sqlite --models-dir models --out-dir artifacts --device cuda
```
