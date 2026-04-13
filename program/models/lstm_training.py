import torch.nn as nn

from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from typing import List, Union

from .optimizer_model import AdamConfig, SGDConfig

@dataclass
class ModelTrainingConfig:
  name: str
  model: nn.Module  
  training_dataset: DataLoader
  val_dataset: DataLoader
  test_dataset: DataLoader
  scaler: MinMaxScaler
  target_columns: List[str]

  epochs: int
  batch_size: int
  patience: int
  optimizer: Union[AdamConfig, SGDConfig]
  loss: str
  gradient_clip: float