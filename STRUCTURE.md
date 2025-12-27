# Структура проекта

## Организация по функциональности

```
Voice_Settings/
├── bin/
│   └── analyze-audio          # Главный исполняемый скрипт
│
├── cli/
│   └── main.py                # CLI интерфейс: парсинг аргументов
│
├── src/
│   ├── extractors/            # Сбор метрик
│   │   ├── praat.py           # Извлечение данных из Praat
│   │   └── verify_pitch.py    # Верификация pitch (pyin)
│   │
│   ├── reporters/             # Упаковка вывода
│   │   ├── visual.py          # Генерация визуализаций (ReportGenerator)
│   │   └── create_report.py   # Legacy: прямой вызов генерации отчетов
│   │
│   ├── dsp/                   # Обработка сигналов
│   │   └── dsp.py             # STFT, спектрограммы, фильтры
│   │
│   ├── core/                  # Основной пайплайн
│   │   ├── pipeline.py        # Orchestration: AudioAnalyzer
│   │   └── selftest.py        # Автотесты
│   │
│   └── config/                # Настройки и схемы
│       ├── settings.py        # Параметры анализа
│       └── schema.py          # Валидация данных
│
├── src/                       # Приватные исходные файлы (игнорируется git)
│   └── ...
│
├── LICENSE
├── README.md
└── .gitignore
```

## Описание модулей

### `bin/analyze-audio`
Точка входа. Вызывает `cli.main.main()`.

### `cli/main.py`
CLI интерфейс: парсинг аргументов, валидация, вызов `AudioAnalyzer`.

### `src/extractors/`
**Сбор метрик:**
- `praat.py` — извлечение данных из Praat (pitch, formants, LTAS, метрики)
- `verify_pitch.py` — верификация pitch вторым методом (pyin)

### `src/reporters/`
**Упаковка вывода:**
- `visual.py` — генерация отчетов (HTML, PDF, PNG) через `ReportGenerator`
- `create_report.py` — legacy модуль для прямого вызова

### `src/dsp/`
**Обработка сигналов:**
- `dsp.py` — STFT, спектрограммы, интерполяция, фильтрация

### `src/core/`
**Основной пайплайн:**
- `pipeline.py` — `AudioAnalyzer`: orchestration всего процесса
- `selftest.py` — автотесты (синус, гармоники, тишина, шум)

### `src/config/`
**Настройки:**
- `settings.py` — все параметры анализа (STFT, pitch, formants, LTAS)
- `schema.py` — схемы данных, валидация, `manifest.json` структура

## Поток данных

```
Input file
    ↓
[cli/main.py] - парсинг аргументов
    ↓
[core/pipeline.py] - AudioAnalyzer
    ↓
[dsp/dsp.py] - обработка аудио (STFT, конвертация)
    ↓
[extractors/praat.py] - извлечение метрик (Praat)
    ↓
[extractors/verify_pitch.py] - верификация pitch
    ↓
[reporters/visual.py] - генерация отчетов (HTML/PDF/PNG)
    ↓
Output: analysis_YYYYMMDD_HHMMSS/
```

