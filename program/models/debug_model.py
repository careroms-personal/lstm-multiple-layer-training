import inspect

from pathlib import Path
from typing import Any
from pydantic import BaseModel

class DebugBase(BaseModel):
  def log(self, enabled: bool, data: Any = None) -> None:
    if not enabled:
      return
    frame = inspect.currentframe().f_back
    func = frame.f_code.co_name
    parts = Path(frame.f_code.co_filename).parts
    try:
      idx = parts.index("program")
      file_path = "/".join(parts[idx:])
    except ValueError:
      file_path = Path(frame.f_code.co_filename).name
    print(f"[DEBUG] {file_path} :: {func}")
    if data is not None:
      print(data)

class DatasetPreparationDebug(DebugBase):
  main_executor: bool = False
  sub_executor: bool = False

class ModelPreparationDebug(DebugBase):
  main_executor: bool = False

class Debug(BaseModel):
  dataset_preparation: DatasetPreparationDebug
  model_preparation: ModelPreparationDebug
