# Pacman Shared-Policy PPO Spec

## Why
Нужно обучить Pacman и 4 призраков с одной общей нейросетью на классической карте, с поддержкой CUDA (Kaggle P100) и с устойчивой телеметрией (SQLite + Telegram) без спама.

## What Changes
- Добавить ASCII-парсер карты и валидатор легенды символов.
- Реализовать среду Pacman (grid world) с дискретными действиями и пошаговой симуляцией.
- Реализовать обучение PPO на PyTorch с одной общей policy/value сетью для Pacman и призраков (общий trunk + роль через agent_id).
- Добавить логирование метрик в SQLite и периодическую отправку прогресса/файла БД в Telegram с throttling и редактированием сообщения.
- Добавить конфигурацию через аргументы CLI и переменные окружения (секреты вне репозитория).

## Impact
- Affected specs: RL training loop, multi-agent environment, telemetry pipeline.
- Affected code: новый пакет Python (env, model, ppo, storage, telegram), CLI entrypoint, тесты.

## ADDED Requirements

### Requirement: ASCII Map Format
Система SHALL поддерживать загрузку карты из ASCII-строки/файла и валидировать её структуру.

#### Legend
- `#` — стена (непроходимо)
- `.` — еда/пеллета (съедаемо)
- ` ` (space) — пустой проход
- `0` — стартовая позиция Pacman (ровно 1)
- `B` — Blinky (ровно 1)
- `P` — Pinky (ровно 1)
- `I` — Inky (ровно 1)
- `C` — Clyde (ровно 1)

#### Scenario: Success case
- **WHEN** пользователь передаёт ASCII карту
- **THEN** парсер строит grid `H x W`, находит стартовые позиции агентов, проверяет прямоугольность строк, и возвращает структуру карты.

#### Scenario: Validation failure
- **WHEN** в карте отсутствует один из 5 агентов, либо встречаются неизвестные символы, либо строки разной длины
- **THEN** система возвращает ошибку с контекстом (какой символ/строка/условие нарушены).

### Requirement: Pacman Environment (Multi-Agent)
Система SHALL предоставлять среду пошаговой симуляции Pacman на grid-карте с 5 агентами: Pacman и 4 призрака.

#### Environment API
- `reset(seed)` возвращает начальные наблюдения для всех агентов, и метаданные эпизода.
- `step(actions)` принимает действие для каждого агента на текущем шаге и возвращает:
  - `obs_by_agent`, `reward_by_agent`, `done`, `info`
- `done` устанавливается при завершении эпизода (Pacman съеден, либо все пеллеты съедены, либо достигнут лимит шагов).

#### Actions
- Дискретные действия: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`.
- Перемещение в стену не меняет позицию.

#### Collisions and Rules
- Если позиция Pacman совпадает с позицией любого призрака после применения действий на шаге — эпизод завершается (Pacman проиграл).
- Если Pacman попадает на клетку с `.` — пеллета исчезает.

#### Scenario: Success case
- **WHEN** выполняется `step` с валидными действиями
- **THEN** позиции обновляются, пеллеты собираются, коллизии учитываются, и возвращаются корректные награды и флаг завершения.

### Requirement: Observations (Full Grid One-Hot)
Система SHALL генерировать наблюдение как полный one-hot тензор по сетке для каждого агента.

#### Observation Tensor
- Формат: `C x H x W` (PyTorch tensor, float32)
- Каналы минимум:
  - стены
  - пеллеты
  - Pacman
  - 4 призрака (по отдельному каналу на каждого)
  - `agent_id` как отдельный канал(ы), позволяющий одной сети различать роли (Pacman vs конкретный призрак)

#### Scenario: Success case
- **WHEN** среда возвращает наблюдение для агента
- **THEN** tensor соответствует текущему состоянию и согласован по каналам/размерам.

### Requirement: Shared Network (Single Model for All Agents)
Система SHALL использовать одну общую нейросеть для всех 5 агентов.

#### Model Contract
- Одна модель принимает `(obs, agent_id)` и возвращает распределение по действиям и value.
- Модель использует общий trunk (например CNN по `H x W`) и различает агентов через `agent_id` вход (в каналах или через embedding, конкатенируемый к features).

#### Scenario: Success case
- **WHEN** выполняется инференс для Pacman и любого призрака
- **THEN** используется один и тот же объект модели и одни и те же параметры (shared weights).

### Requirement: PPO Training (CUDA-capable)
Система SHALL обучать общую policy/value сеть методом PPO в PyTorch с поддержкой CUDA.

#### CUDA Requirements
- **WHEN** доступен `torch.cuda.is_available()`
- **THEN** модель и батчи перемещаются на `cuda`, и обучение выполняется на GPU (в т.ч. Kaggle P100).
- **WHEN** CUDA недоступна
- **THEN** обучение корректно работает на CPU без изменения кода пользователя (кроме конфигов производительности).

#### PPO Loop (High Level)
- Rollout собирается синхронно из среды на фиксированное число шагов.
- Считаются advantages (GAE) и returns.
- Выполняются несколько epochs оптимизации на одном rollout.

#### Scenario: Success case
- **WHEN** запускается тренировочный CLI с PPO
- **THEN** метрики (reward/loss/entropy/kl/step/sec) обновляются, модель обучается, и процесс устойчиво продолжается.

### Requirement: SQLite Logging
Система SHALL логировать метрики и события обучения в SQLite (без `SELECT *` в критичных запросах и с явными типами).

#### Storage Requirements
- SQLite файл создаётся/используется по пути из конфигурации.
- Записи добавляются транзакционно и короткими транзакциями.
- Схема содержит первичные ключи и `NOT NULL` где уместно.

#### Minimum Logged Fields
- `global_step` (integer, unique or indexed)
- `episode` (integer)
- `timestamp` (integer unix or ISO string)
- `reward_mean` (real) и/или reward per agent
- `loss_policy`, `loss_value`, `entropy`, `approx_kl` (real, nullable если не рассчитано)
- `fps` (real)

#### Scenario: Success case
- **WHEN** идёт обучение
- **THEN** каждые N шагов (конфиг, по умолчанию 1) метрики добавляются в SQLite.

### Requirement: Telegram Reporting (Anti-Spam)
Система SHALL отправлять прогресс обучения и SQLite-файл в Telegram по расписанию, минимизируя спам через редактирование сообщения.

#### Configuration
- Секреты/идентификаторы задаются через env vars/CLI:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
  - опционально `TELEGRAM_TOPIC_ID`/`MESSAGE_THREAD_ID` (если используется)

#### Progress Message Policy
- Каждые 100 шагов система пытается обновить (edit) последнее “прогресс”-сообщение.
- Сообщение считается “активным” в течение 1 часа с момента отправки:
  - если прошло > 1 часа — отправляется новое прогресс-сообщение и оно становится активным на следующий час
  - если редактирование невозможно (например, Telegram отказал) — отправляется новое и оно становится активным

#### SQLite Send Policy
- Каждые 1000 шагов система отправляет краткий отчёт (или обновляет активное сообщение, если доступно) и гарантирует, что метрики в SQLite записаны.
- Каждые 2000–3000 шагов (конфиг по умолчанию 2500) система отправляет текущий файл SQLite как документ в Telegram чат.

#### Scenario: Success case
- **WHEN** обучение идёт тысячи шагов
- **THEN** Telegram получает редактируемый прогресс без частого спама, и периодически получает файл SQLite.

## MODIFIED Requirements
Нет.

## REMOVED Requirements
Нет.

