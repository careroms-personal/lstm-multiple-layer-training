import yaml, sys
import torch

from pathlib import Path
from pydantic import ValidationError

from models.training_config import TrainingConfig

from .executors.dataset_preparation.main_executor import DatasetPreparationExecutor
from .executors.model_preparation.main_executor import ModelPreparationExecutor
from .executors.model_training.main_executor import ModelTrainingExecutor

class Processor:
  def __init__(self, config_path: str):
    self._load_and_validate_config(config_path=config_path)

    if self.training_config.training_setting.use_gpu:
      self._configure_gpu()

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

  def execute(self):
    dataset_preparation_executor = DatasetPreparationExecutor(self.training_config)
    model_training_dataset = dataset_preparation_executor.executor()

    model_preparation = ModelPreparationExecutor(self.training_config, model_training_dataset)
    training_models = model_preparation.execute()

    model_training = ModelTrainingExecutor(self.training_config,training_models)
    model_training.execute()