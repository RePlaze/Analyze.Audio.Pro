# Структура проекта

## Организация по функциональности

```
Voice_Settings/
├── bin/
│   └── analyze-audio          # Главный исполняемый скрипт
│
├── cli/
│   └── main.py                # CLI интерфейс
│
├── src/
│   ├── metrics/               # Сбор метрик
│   │   ├── praat.py           # Извлечение данных из Praat
│   │   └── verify_pitch.py    # Верификация pitch
│   │
│   ├── output/                # Генерация отчетов
│   │   ├── reports.py         # Генерация визуализаций
│   │   └── create_report.py   # Legacy: прямой вызов
│   │
│   ├── audio/                 # Обработка аудио
│   │   └── audio.py           # STFT, спектрограммы, фильтры
│   │
│   ├── main/                  # Основной пайплайн
│   │   ├── analyzer.py        # AudioAnalyzer
│   │   └── tests.py           # Автотесты
│   │
│   └── settings/              # Настройки
│       ├── settings.py        # Параметры анализа
│       └── validation.py      # Валидация данных
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

### `src/metrics/`
**Сбор метрик:**
- `praat.py` — извлечение данных из Praat (pitch, formants, LTAS, метрики)
- `verify_pitch.py` — верификация pitch вторым методом (pyin)

### `src/output/`
**Генерация отчетов:**
- `reports.py` — генерация отчетов (HTML, PDF, PNG) через `ReportGenerator`
- `create_report.py` — legacy модуль для прямого вызова

### `src/audio/`
**Обработка аудио:**
- `audio.py` — STFT, спектрограммы, интерполяция, фильтрация

### `src/main/`
**Основной пайплайн:**
- `analyzer.py` — `AudioAnalyzer`: orchestration всего процесса
- `tests.py` — автотесты (синус, гармоники, тишина, шум)

### `src/settings/`
**Настройки:**
- `settings.py` — все параметры анализа (STFT, pitch, formants, LTAS)
- `validation.py` — схемы данных, валидация, `manifest.json` структура

## Поток данных

```
Input file
    ↓
[cli/main.py] - парсинг аргументов
    ↓
[main/analyzer.py] - AudioAnalyzer
    ↓
[audio/audio.py] - обработка аудио (STFT, конвертация)
    ↓
[metrics/praat.py] - извлечение метрик (Praat)
    ↓
[metrics/verify_pitch.py] - верификация pitch
    ↓
[output/reports.py] - генерация отчетов (HTML/PDF/PNG)
    ↓
Output: analysis_YYYYMMDD_HHMMSS/
```

