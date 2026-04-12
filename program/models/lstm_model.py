import pandas as pd

from sklearn.preprocessing import MinMaxScaler  # type: ignore

from typing import List, Optional
from pydantic import BaseModel
from pydantic.config import ConfigDict

class CustomBaseModel(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

class ModelTrainingDataset(CustomBaseModel):
  training_dataset: pd.DataFrame
  validation_dataset: pd.DataFrame
  test_dataset: pd.DataFrame

  timeseries_column: str
  target_columns: List[str]
  feature_columns: Optional[List[str]]

class ModelTrainingConfig(CustomBaseModel):
  name: str
  window_size: int
  units: List[int]
  dropout: float
  float_type: str


class ModelTrainingConfig(CustomBaseModel):
  name: str
  window_size: int
  units: List[int]
  dropout: float
  float_type: str

  epochs: int
  batch_size: int
  patience: int
  optimizer: str
  loss: str

  timeseries_column: str
  target_columns: List[str]
  feature_columns: Optional[List[str]]

  normalize_training_dataset: pd.DataFrame
  normalize_validation_dataset: pd.DataFrame
  normalize_test_dataset: pd.DataFrame

  scaler: MinMaxScaler