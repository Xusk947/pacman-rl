---
name: skill-writer
description: Guide users through creating Agent Skills for Claude Code. Use when the user wants to create, write, author, or design a new Skill, or needs help with SKILL.md files, frontmatter, or skill structure.
---

# Skill Writer

Этот Skill помогает создавать хорошо структурированные Agent Skills для Claude Code, которые соответствуют лучшим практикам и требованиям валидации.

## Когда использовать

Используй этот Skill когда нужно:

- Создать новый Agent Skill
- Написать или обновить SKILL.md
- Спроектировать структуру Skill и YAML frontmatter
- Починить проблемы с обнаружением Skill
- Превратить существующий промпт или workflow в Skill

## Инструкции

### Шаг 1: Определи область Skill

Сначала уточни, что именно должен делать Skill:

1. Задай уточняющие вопросы:
   - Какую конкретную возможность должен давать Skill?
   - Когда Claude должен использовать этот Skill?
   - Какие инструменты/ресурсы нужны?
   - Это для личного использования или для команды?
2. Держи фокус: один Skill = одна способность
   - Хорошо: “PDF form filling”, “Excel data analysis”
   - Слишком широко: “Document processing”, “Data tools”

### Шаг 2: Выбери расположение Skill

Определи, куда создавать Skill:

Личные Skills ( ~/.claude/skills/ ):
- индивидуальные workflows и предпочтения
- экспериментальные Skills
- личная продуктивность

Проектные Skills ( .claude/skills/ ):
- командные workflows и конвенции
- проектно-специфичная экспертиза
- шаринг внутри репозитория (коммитится в git)

### Шаг 3: Создай структуру Skill

Создай директорию и файлы:

```bash
# Personal
mkdir -p ~/.claude/skills/skill-name

# Project
mkdir -p .claude/skills/skill-name
```

Для multi-file Skills:

```text
skill-name/
├── SKILL.md (required)
├── reference.md (optional)
├── examples.md (optional)
├── scripts/
│   └── helper.py (optional)
└── templates/
    └── template.txt (optional)
```

### Шаг 4: Напиши YAML frontmatter в SKILL.md

```yaml
---
name: skill-name
description: Brief description of what this does and when to use it
---
```

Требования к полям:

- name:
  - только lowercase, цифры и дефисы
  - максимум 64 символа
  - должен совпадать с названием директории
- description:
  - максимум 1024 символа
  - включает “что делает” и “когда использовать”
  - содержит триггерные слова, которые пользователь реально пишет

Опционально:

```yaml
allowed-tools: Read, Grep, Glob
```

### Шаг 5: Пиши “активируемые” описания

Формула: что делает + когда использовать + триггеры

Хорошо:

```yaml
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
```

Плохо:

```yaml
description: Helps with documents
```

### Шаг 6: Структурируй контент Skill

Рекомендуемая структура:

```markdown
# Skill Name

Короткое описание.

## Quick start

Минимальный пример.

## Instructions

1. Первый шаг
2. Второй шаг
3. Edge cases

## Examples

Примеры.

## Best practices

- Конвенции
- Подводные камни
```

### Шаг 7: Добавь supporting files (опционально)

- reference.md: подробная документация/опции
- examples.md: расширенные примеры
- scripts/: скрипты-хелперы
- templates/: шаблоны

Ссылайся на них из SKILL.md:

```markdown
Для advanced usage см. [reference.md](reference.md).
```

### Шаг 8: Валидация Skill

Проверь:

- SKILL.md существует в правильной папке
- директория совпадает с name во frontmatter
- YAML валиден (без табов, корректные отступы)
- name соответствует правилам
- description специфичен и короче 1024 символов

### Шаг 9: Тест Skill

1) Перезапусти Claude Code, чтобы Skill подхватился

2) Спроси что-то по триггеру:

```text
Помоги создать новый Skill для обработки PDF
```

3) Убедись, что Claude использует инструкции Skill

### Шаг 10: Дебаг, если не активируется

- Сделай description более конкретным (добавь триггеры/расширения файлов)
- Проверь путь к SKILL.md
- Проверь YAML
- Запусти debug-режим (если доступно в твоей среде)

## Output format

Когда этот Skill используется, результат должен включать:

1) Уточняющие вопросы о целях и ограничениях
2) Предложение имени и расположения Skill
3) Создание SKILL.md с корректным frontmatter
4) Чёткие пошаговые инструкции и примеры
5) Supporting files (если нужно)
6) Инструкции по тестированию
7) Чеклист валидации
