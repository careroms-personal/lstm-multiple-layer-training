import yaml, sys

from pathlib import Path
from pydantic import ValidationError

from models.training_config import TrainingConfig

from executors.dataset_preparation.main_executor import DatasetPreparationExecutor

class Processor:
  def __init__(self, config_path: str):
    self._load_and_validate_config(config_path=config_path)

  def _load_and_validate_config(self, config_path: str):
    if not Path(config_path).exists():
      print(f"❌ Config file not found: {config_path}")
      sys.exit(1)
    
    try:
      with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

      self.training_config = TrainingConfig(**yaml_data)
      self.training_config.config_base_dir = str(Path(config_path).parent.resolve())
        
    except ValidationError as e:
      print(f"❌ Invalid config file:")

      for error in e.errors():
        print(f"   - {error['loc']}: {error['msg']}")
      
      sys.exit(1)

  def execute(self):
    dataset_preparation_executor = DatasetPreparationExecutor(self.training_config)
    training_dataset, validation_dataset, test_dataset = dataset_preparation_executor.executor