import io
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset  # type: ignore

from models.debug_model import Debug
from models.lstm_architecture import ModelArchitectureConfig
from models.lstm_training import ModelTrainingConfig
from typing import List

class LSTMModel(nn.Module):
  def __init__(self, units: List[int], dropout: float, n_features: int, n_targets: int, name: str):
    super().__init__()

    self.name = name
    self.lstm_layers = nn.ModuleList()
    self.dropout_layers = nn.ModuleList()

    input_size = n_features

    for unit in units:
      self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=unit, batch_first=True))
      self.dropout_layers.append(nn.Dropout(dropout))
      input_size = unit

    self.dense = nn.Linear(units[-1], n_targets)

  def forward(self, x):
    for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
      x, _ = lstm(x)
      x = dropout(x)
    
    x = x[:, -1, :]

    return self.dense(x)

class ModelBuildExecutor:
  def __init__(self, model_config: ModelArchitectureConfig, debug_mode: Debug):
    self.model_config = model_config
    self.debug_mode = debug_mode.model_training

  def _build_dataset(self, df: pd.DataFrame):
    dtype = torch.float32 if self.model_config.float_type == "float32" else torch.float64
    n_targets = len(self.model_config.target_columns)
    data = torch.tensor(df.values, dtype=dtype)

    if len(data) <= self.model_config.window_size:
      raise ValueError(
        f"Dataset too small for window_size! "
        f"data={len(data)} rows, "
        f"window_size={self.model_config.window_size}. "
        f"Need at least {self.model_config.window_size + 1} rows."
      )

    X, y = [], []
    
    for i in range(len(data) - self.model_config.window_size):
      X.append(data[i:i + self.model_config.window_size])
      y.append(data[i + self.model_config.window_size, :n_targets])

    X = torch.stack(X)  # (n_samples, window_size, n_features)
    y = torch.stack(y)  # (n_samples, n_targets)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=self.model_config.batch_size, shuffle=False)

    self.debug_mode.log(self.debug_mode.main_executor, loader)

    return loader

  def _build_model_architecture(self) -> nn.Module:
    n_targets = len(self.model_config.target_columns)
    n_features = len(self.model_config.feature_columns or []) + n_targets

    model = LSTMModel(
      units=self.model_config.units,
      dropout=self.model_config.dropout,
      n_features=n_features,
      n_targets=n_targets,
      name=self.model_config.name
    )
      
    if self.debug_mode.model_build_executor:
      self.debug_mode.log(self.debug_mode.model_build_executor, model)

    return model, n_features

  def execute(self):
    model, n_features = self._build_model_architecture()
    train_ds = self._build_dataset(self.model_config.normalize_training_dataset)
    val_ds = self._build_dataset(self.model_config.normalize_validation_dataset)
    test_ds = self._build_dataset(self.model_config.normalize_test_dataset)

    model_training_config = ModelTrainingConfig(
      name=self.model_config.name,
      model=model,
      training_dataset=train_ds,
      val_dataset=val_ds,
      test_dataset=test_ds,
      scaler=self.model_config.scaler,
      target_columns=self.model_config.target_columns,
      unit=self.model_config.units,
      dropout=self.model_config.dropout,
      n_features=n_features,
      windows_size=self.model_config.window_size,
      epochs=self.model_config.epochs,
      batch_size=self.model_config.batch_size,
      patience=self.model_config.patience,
      optimizer=self.model_config.optimizer,
      loss=self.model_config.loss,
      gradient_clip=self.model_config.gradient_clip,
    )

    return model_training_config