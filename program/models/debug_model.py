import inspect

from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, PrivateAttr

class DebugBase(BaseModel):
  _log_file: Optional[str] = PrivateAttr(default=None)

  def set_log_file(self, log_file: str) -> None:
    self._log_file = log_file

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

    lines = [f"[DEBUG] {file_path} :: {func}"]

    if data is not None:
      lines.append(str(data))

    message = "\n".join(lines)
    
    print(message)

    if self._log_file:
      with open(self._log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

  def write(self, message: str) -> None:
    print(message)
    if self._log_file:
      with open(self._log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

class DatasetPreparationDebug(DebugBase):
  main_executor: bool = False
  sub_executor: bool = False

class ModelPreparationDebug(DebugBase):
  main_executor: bool = False

class ModelTrainingDebug(DebugBase):
  main_executor: bool = False
  model_build_executor: bool = False
  model_test_executor: bool = False

class Debug(BaseModel):
  dataset_preparation: DatasetPreparationDebug = DatasetPreparationDebug()
  model_preparation: ModelPreparationDebug = ModelPreparationDebug()
  model_training: ModelTrainingDebug = ModelTrainingDebug()

  def configure_file_output(self, logs_path: str, name: str) -> None:
    log_file = str(Path(logs_path) / f"{name}.log")
    Path(logs_path).mkdir(parents=True, exist_ok=True)
    self.dataset_preparation.set_log_file(log_file)
    self.model_preparation.set_log_file(log_file)
    self.model_training.set_log_file(log_file)