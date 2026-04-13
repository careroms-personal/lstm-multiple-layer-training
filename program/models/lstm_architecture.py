import pandas as pd

from sklearn.preprocessing import MinMaxScaler  # type: ignore

from typing import List, Optional, Union
from pydantic import BaseModel
from pydantic.config import ConfigDict

from .optimizer_model import AdamConfig, SGDConfig

class CustomBaseModel(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

class ModelTrainingDataset(CustomBaseModel):
  training_dataset: pd.DataFrame
  validation_dataset: pd.DataFrame
  test_dataset: pd.DataFrame

  timeseries_column: str
  target_columns: List[str]
  feature_columns: Optional[List[str]]

class ModelArchitectureConfig(CustomBaseModel):
  name: str
  window_size: int
  units: List[int]
  dropout: float
  float_type: str

  epochs: int
  batch_size: int
  patience: int
  optimizer: Union[AdamConfig, SGDConfig]
  loss: str

  timeseries_column: str
  target_columns: List[str]
  feature_columns: Optional[List[str]]

  normalize_training_dataset: pd.DataFrame
  normalize_validation_dataset: pd.DataFrame
  normalize_test_dataset: pd.DataFrame

  scaler: MinMaxScaler
  gradient_clip: float