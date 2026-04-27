VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
TRAIN := $(VENV)/bin/pacman-rl-train
PLAY := $(VENV)/bin/pacman-rl-play
REPORT := $(VENV)/bin/pacman-rl-report
PY_SYS ?= python3
PIP_SYS ?= $(PY_SYS) -m pip
MODEL ?=
ARTIFACTS ?= artifacts
TOTAL_TIMESTEPS ?= 2000000
STEPS ?= $(TOTAL_TIMESTEPS)
ALGOS ?= ppo a2c dqn
DEVICE ?= cuda

.PHONY: venv install runenv runvenv test clean
.PHONY: sysdeps roms
.PHONY: play playrgb
.PHONY: kaggle-install kaggle-runenv

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
	$(PIP) install -U "requests>=2.31"
	$(VENV)/bin/AutoROM --accept-license

runenv: install
	mkdir -p $(ARTIFACTS)
	$(TRAIN) --db runs.sqlite --total-timesteps $(STEPS) --algos $(ALGOS) --device $(DEVICE) --print-every-percent 5 --stats-window-episodes 100
	$(REPORT) --db runs.sqlite --models-dir models --out-dir $(ARTIFACTS) --device $(DEVICE)

runvenv: runenv

play: install
	$(PLAY) --model $(MODEL) --render human --device cuda

playrgb: install
	$(PLAY) --model $(MODEL) --render rgb_array --device cuda --episodes 1 --max-steps 2000

test: venv
	$(PY) -m unittest discover -s tests -p 'test_*.py' -q

roms: install
	$(VENV)/bin/AutoROM --accept-license

kaggle-install:
	$(PIP_SYS) install -U pip setuptools wheel
	$(PIP_SYS) install -e .
	$(PIP_SYS) install -U "gymnasium[atari]>=0.29.1,<1.3.0"
	$(PIP_SYS) install -U "autorom[accept-rom-license]"
	$(PIP_SYS) install -U "moviepy>=1.0.3"
	$(PIP_SYS) install -U "matplotlib>=3.8"
	$(PIP_SYS) install -U "requests>=2.31"
	AutoROM --accept-license

kaggle-runenv: kaggle-install
	mkdir -p $(ARTIFACTS)
	$(PY_SYS) -m pacman_rl.cli --db runs.sqlite --total-timesteps $(STEPS) --algos $(ALGOS) --device $(DEVICE) --print-every-percent 5 --stats-window-episodes 100
	$(PY_SYS) -m pacman_rl.report --db runs.sqlite --models-dir models --out-dir $(ARTIFACTS) --device $(DEVICE)

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache
