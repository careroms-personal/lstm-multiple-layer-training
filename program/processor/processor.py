import yaml, sys
import torch, random
import numpy as np

from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from models.training_config import TrainingConfig

from .executors.dataset_preparation.main_executor import DatasetPreparationExecutor
from .executors.model_preparation.main_executor import ModelPreparationExecutor
from .executors.model_training.main_executor import ModelTrainingExecutor
from .executors.model_ensemble.main_executor import ModelEnsembleExecutor
from .executors.model_exporter.main_executor import ModelExporterExecutor

class Processor:
  def __init__(self, config_path: str):
    self._load_and_validate_config(config_path=config_path)
    self._set_seed(self.training_config.training_setting.seed)

    if self.training_config.training_setting.use_gpu:
      self._configure_gpu()

  def _set_seed(self, seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

  def _configure_gpu(self):
    if torch.cuda.is_available():
      print(f"[OK] GPU enabled: {torch.cuda.get_device_name(0)}")
    else:
      print("[WARNING] No GPU found - using CPU")

  def _load_and_validate_config(self, config_path: str):
    if not Path(config_path).exists():
      print(f"[ERROR] Config file not found: {config_path}")
      sys.exit(1)
    
    try:
      with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

      self.training_config = TrainingConfig(**yaml_data)
      self.training_config.config_base_dir = str(Path(config_path).parent.resolve())
        
    except ValidationError as e:
      print(f"[ERROR] Invalid config file:")

      for error in e.errors():
        print(f"   - {error['loc']}: {error['msg']}")
      
      sys.exit(1)

  def _resolve_logs_path(self) -> Optional[str]:
    write_output = self.training_config.output.write_output
    if not write_output or not write_output.enabled:
      return None
    return str(Path(self.training_config.config_base_dir) / write_output.logs_path)

  def _clear_logs(self, logs_path: str):
    logs_dir = Path(logs_path)
    if logs_dir.exists():
      for log_file in logs_dir.glob("*.log"):
        log_file.unlink()

  def execute(self):
    logs_path = self._resolve_logs_path()

    if logs_path:
      self._clear_logs(logs_path)

    if logs_path:
      self.training_config.debug.configure_file_output(logs_path, "dataset_preparation")

    dataset_preparation_executor = DatasetPreparationExecutor(self.training_config)
    model_training_dataset = dataset_preparation_executor.executor()

    if logs_path:
      self.training_config.debug.configure_file_output(logs_path, "model_preparation")

    model_preparation = ModelPreparationExecutor(self.training_config, model_training_dataset)
    training_models = model_preparation.execute()

    model_training = ModelTrainingExecutor(self.training_config, training_models, logs_path=logs_path)
    trained_models = model_training.execute()

    if self.training_config.ensemble.enabled:
      model_ensemble = ModelEnsembleExecutor(training_config=self.training_config, model_trained_results=trained_models)
      ensemble_model = model_ensemble.execute()

      model_export = ModelExporterExecutor(training_config=self.training_config, ensembled_result=ensemble_model)
      model_export.execute()