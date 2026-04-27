VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
TRAIN := $(VENV)/bin/pacman-rl-train
PLAY := $(VENV)/bin/pacman-rl-play
REPORT := $(VENV)/bin/pacman-rl-report
MODEL ?=
ARTIFACTS ?= artifacts

.PHONY: venv install runenv runvenv test clean
.PHONY: sysdeps roms
.PHONY: play playrgb

venv: $(PY)

sysdeps:
	sudo pacman -S --needed hdf5 vtk ffmpeg

$(PY):
	python3 -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel

install: venv
	$(PIP) install -e .
	$(PIP) install -U "gymnasium[atari]>=0.29.1,<1.3.0"
	$(PIP) install -U "autorom[accept-rom-license]"
	$(PIP) install -U "moviepy>=1.0.3"
	$(PIP) install -U "matplotlib>=3.8"
	$(VENV)/bin/AutoROM --accept-license

runenv: install
	mkdir -p $(ARTIFACTS)
	$(TRAIN) --db runs.sqlite --total-timesteps 200000 --algos ppo a2c dqn --device cuda --print-every-percent 5 --stats-window-episodes 100 --record-video-dir $(ARTIFACTS)/train_videos --video-trigger-steps 50000 --video-length 1800
	$(REPORT) --db runs.sqlite --models-dir models --out-dir $(ARTIFACTS) --device cuda

runvenv: runenv

play: install
	$(PLAY) --model $(MODEL) --render human --device cuda

playrgb: install
	$(PLAY) --model $(MODEL) --render rgb_array --device cuda --episodes 1 --max-steps 2000

test: venv
	$(PY) -m unittest discover -s tests -p 'test_*.py' -q

roms: install
	$(VENV)/bin/AutoROM --accept-license

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache
