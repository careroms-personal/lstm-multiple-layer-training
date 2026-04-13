from pydantic import BaseModel, model_validator
from typing import Optional, List, Any, Literal
from enum import StrEnum

from .datasources_config_model import Files
from .debug_model import Debug

class DatasourceType(StrEnum):
  Files = "files"

_DATASOURCE_CONFIG_MAP = {
  DatasourceType.Files: Files,
}

class Datasource(BaseModel):
  type: DatasourceType = DatasourceType.Files
  config: Any

  @model_validator(mode='before')
  @classmethod
  def validate_config_matches_type(cls, values):
    type_ = values.get('type', DatasourceType.Files)
    config = values.get('config')
    config_model = _DATASOURCE_CONFIG_MAP.get(DatasourceType(type_))
    if config_model is None:
      raise ValueError(f"Unknown datasource type: '{type_}'")
    if isinstance(config, dict):
      values['config'] = config_model(**config)
    elif not isinstance(config, config_model):
      raise ValueError(f"datasource config must be {config_model.__name__} for type '{type_}'")
    return values

class TrainingData(BaseModel):
  validation_data_ratio: float
  test_data_ratio: float
  timeseries_column: str
  target_columns: List[str] = []
  feature_columns: List[str] = []

class LSTMModel(BaseModel):
  name: str
  window_size: int
  units: List[int]
  dropout: float
  float_type: Literal["float32"]

class TrainingSetting(BaseModel):
  epochs: int = 50
  batch_size: int = 32
  patience: int = 10
  optimizer: Literal["adam"] = "adam"
  loss: Literal["mse"] = "mse"

class Ensemble(BaseModel):
  method: Literal["stacking"]
  meta_learner: Literal["linear_regression"]

class PrintOutput(BaseModel):
  training_data: bool = False
  training_logs: bool = False
  model_summary: bool = False
  ensemble_weights: bool = False

class WriteOutput(BaseModel):
  training_data_path: str = ""
  logs_path: str = ""

class Output(BaseModel):
  model_output_path: str
  print_output: Optional[PrintOutput] = None
  write_output: Optional[WriteOutput] = None

class TrainingConfig(BaseModel):
  config_base_dir: Optional[str] = "" # Set in processing when validate model
  datasource: Datasource
  training_data: TrainingData
  lstm_models: List[LSTMModel]
  training_setting: TrainingSetting
  ensemble: Ensemble
  output: Output
  debug: Debug