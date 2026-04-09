# pyconfplate — AI Guide
<!-- last human review: 2026 Mar 24 -->
<!-- last ai update: 2026 Apr 09 -->

Quick reference for AI assistants working in this codebase.
See `vibe-code-rule.yaml` for project rules.

---

## Project Purpose

Boilerplate for a Python CLI program that reads a YAML config file and runs a pipeline of executors orchestrated by a Processor. Each executor has a single duty, returns a typed result, and passes it to the next executor in the chain.

---

## Project Structure

```
pyconfplate/
├── vibe-code-rule.yaml              # AI instruction manifest — read first
├── pyproject.toml                   # dependencies and build config (only place for deps)
├── .ai/
│   └── AI-PYTHON-GUIDE.md           # this file
└── program/
    ├── app/
    │   └── main.py                  # CLI entry point — do not add logic here
    ├── models/
    │   ├── config_models.py         # pydantic config models (AppConfig and sub-models)
    │   └── result_models.py         # pydantic result models passed between executors
    ├── processor/
    │   ├── processor.py             # orchestrator: config load + executor chain
    │   ├── <duty>_executor.py       # one file per executor
    │   └── ...
    ├── global_config.py             # reserved — do not add models here
    ├── config_templates/
    │   └── config.yaml              # YAML template showing valid config structure
    └── test_suits/
        └── global_test_config.py    # shared test fixtures/constants
```

---

## How to Look Things Up

| What you need | Where to look |
|---|---|
| CLI argument parsing | `program/app/main.py` |
| Executor chain and flow | `program/processor/processor.py` |
| Config structure (pydantic) | `program/models/config_models.py` |
| Inter-executor result types | `program/models/result_models.py` |
| YAML config structure | `program/config_templates/config.yaml` |
| Shared test fixtures | `program/test_suits/global_test_config.py` |
| Dependencies | `pyproject.toml` |

---

## Architecture: Processor + Executor Chain

```
YAML Config File
      ↓
  Processor
  ├─→ __init__: load + validate config → self.config
  └─→ execute():
        ├─→ FirstExecutor(config).execute()            → ResultA
        ├─→ SecondExecutor(result_a, config).execute() → ResultB
        └─→ ThirdExecutor(result_b, config).execute()  → (side effects / final output)
```

**Rules:**
- Processor holds all executor instances and their results as `self.*`
- Each executor receives the full config — it extracts only what it needs
- Each executor returns a typed pydantic model (never a plain dict or tuple)
- Executors do not call each other — Processor manages the chain

---

## Coding Conventions

- **Indentation:** 2 spaces — never 4 spaces or tabs
- **Error prefix:** `❌` for all fatal errors
- **Exit:** `sys.exit(1)` on any fatal error

---

## Core Patterns

### 1. CLI Entry Point (`main.py`)

No logic here — only wires CLI args to Processor.

```python
import argparse
from processor.processor import Processor

def main():
  parser = argparse.ArgumentParser(description="<program description>")
  parser.add_argument("-c", "--config", required=True, help="Path to config Yaml file")
  args = parser.parse_args()

  processor = Processor(args.config)
  processor.execute()

if __name__ == "__main__":
  main()
```

---

### 2. Processor (`processor.py`)

Loads config, chains executors. No business logic.

```python
import yaml, sys
from pathlib import Path
from pydantic import ValidationError
from models.config_models import AppConfig
from processor.first_executor import FirstExecutor
from processor.second_executor import SecondExecutor

class Processor:
  def __init__(self, config_path: str):
    self.config = self._load_and_validate_config(config_path)

  def _load_and_validate_config(self, config_path: str) -> AppConfig:
    if not Path(config_path).exists():
      print(f"❌ Config file not found: {config_path}")
      sys.exit(1)
    try:
      with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
      return AppConfig(**yaml_data)
    except ValidationError as e:
      print(f"❌ Invalid config file:")
      for error in e.errors():
        print(f"   - {error['loc']}: {error['msg']}")
      sys.exit(1)

  def execute(self):
    self.first_executor = FirstExecutor(self.config)
    self.first_result = self.first_executor.execute()

    self.second_executor = SecondExecutor(self.first_result, self.config)
    self.second_result = self.second_executor.execute()
```

---

### 3. Executor (`<duty>_executor.py`)

One duty per executor. Receives previous result + full config. Returns typed model.

```python
from models.config_models import AppConfig
from models.result_models import FirstResult, SecondResult

class SecondExecutor:
  def __init__(self, first_result: FirstResult, config: AppConfig):
    self.first_result = first_result
    # extract only the config section this executor needs
    self.task_config = config.second_task_config

  def execute(self) -> SecondResult:
    data = self._process(self.first_result)
    return SecondResult(...)

  def _process(self, input: FirstResult) -> ...:
    ...
```

**Naming:** `<duty>_executor.py` — e.g., `data_prepare_executor.py`, `training_executor.py`

---

### 4. Config Models (`models/config_models.py`)

Top-level model validated on startup. Sub-models scoped to each executor's duty.

```python
from pydantic import BaseModel
from typing import Optional

class FirstTaskConfig(BaseModel):
  source: str
  timeout: int = 30

class SecondTaskConfig(BaseModel):
  output_dir: str

class AppConfig(BaseModel):
  first_task: FirstTaskConfig
  second_task: SecondTaskConfig
```

---

### 5. Result Models (`models/result_models.py`)

Typed outputs passed between executors by Processor.

```python
from pydantic import BaseModel

class FirstResult(BaseModel):
  data: list
  count: int

class SecondResult(BaseModel):
  output_path: str
  success: bool
```

---

### 6. YAML Config Template (`config_templates/config.yaml`)

Mirror the pydantic model structure. Use `REQUEST` as placeholder for values the user must fill in.

```yaml
first_task:
  source: REQUEST
  timeout: 30

second_task:
  output_dir: REQUEST
```

---

## Error Handling Convention

- File not found → `❌ Config file not found: <path>` + `sys.exit(1)`
- Pydantic validation error → `❌ Invalid config file:` + per-field errors + `sys.exit(1)`
- Always use `❌` prefix for fatal errors
- Never swallow exceptions silently

---

## Testing Patterns

Tests live in `program/test_suits/`. File names must be prefixed with `test_`.

```python
# program/test_suits/test_processor.py
import pytest
from processor.processor import Processor

def test_config_not_found():
  with pytest.raises(SystemExit):
    Processor("nonexistent.yaml")
```

Shared fixtures and constants go in `global_test_config.py`.

Run tests with:
```bash
pytest program/test_suits/
```

---

## Dependencies

Managed in `pyproject.toml` only — never `requirements.txt`.

Current dependencies:
- `pydantic==2.12.5` — config and result model validation
- `PyYAML==6.0.3` — YAML parsing
- `requests>=2.31.0` — HTTP (available for use in executors)
- `pytest==9.0.2` — testing
