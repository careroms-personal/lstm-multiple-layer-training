from pydantic import BaseModel
from typing import Optional, List, Any, Literal
from enum import StrEnum

from .datasources_config import *

class DatasourceType(StrEnum):
  Files = "files"

class Datasource(BaseModel):
  type: DatasourceType = DatasourceType.Files
  config: Any

class TrainingData(BaseModel):
  validation_data_ratio: float
  test_data_ratio: float
  target_columns: List[str] = []
  feature_columns: List[str] = []

class LSTMModel(BaseModel):
  name: str
  window_size: int
  units: List[int]
  dropout: float
  float_type: Literal["float32"]

class Training(BaseModel):
  epochs: 50
  batch_size: 32
  patience: 10
  optimizer: Literal["adam"]
  loss: Literal["mse"]

class Ensemble(BaseModel):
  method: Literal["stacking"]
  meta_learner: Literal["linear_regression"]

class TrainingConfig(BaseModel):
  config_base_dir: Optional[str] = "" # Set in processing when validate model
