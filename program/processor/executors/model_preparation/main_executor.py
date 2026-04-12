import pandas as pd

from typing import List
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from models.training_config import TrainingConfig
from models.lstm_model import ModelTrainingDataset, ModelTrainingConfig

class ModelPreparationExecutor:
  def __init__(self, training_config: TrainingConfig, model_training_dataset: ModelTrainingDataset):
    self.lstm_models = training_config.lstm_models
    self.training_setting = training_config.training_setting

    self.model_training_dataset = model_training_dataset

  def _set_min_max_scale(self):
    scaler = MinMaxScaler()

    combined_dataset = pd.concat([
      self.model_training_dataset.training_dataset,
      self.model_training_dataset.validation_dataset,
    ])

    scaler.fit(combined_dataset[self.model_training_dataset.target_columns])

    return scaler
  
  def _normalize_training_data(self, dataset: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    target_columns = self.model_training_dataset.target_columns
    
    normalized = scaler.transform(dataset[target_columns])
    
    return pd.DataFrame(normalized, columns=target_columns)
  
  def _prepare_model_traing_config(self) -> List[ModelTrainingConfig]:
    scaler = self._set_min_max_scale()

    normalize_dataset = ModelTrainingDataset(
      training_dataset=self._normalize_training_data(self.model_training_dataset.training_dataset, scaler),
      validation_dataset=self._normalize_training_data(self.model_training_dataset.validation_dataset, scaler),
      test_dataset=self._normalize_training_data(self.model_training_dataset.test_dataset, scaler),
      timeseries_column=self.model_training_dataset.timeseries_column,
      target_columns=self.model_training_dataset.target_columns,
      feature_columns=self.model_training_dataset.feature_columns,
    )

    model_training_configs = []

    for model in self.lstm_models:
      mtc = ModelTrainingConfig(
        name=model.name,
        window_size=model.window_size,
        units=model.units,
        dropout=model.dropout,
        float_type=model.float_type,
        epochs=self.training_setting.epochs,
        batch_size=self.training_setting.batch_size,
        patience=self.training_setting.patience,
        optimizer=self.training_setting.optimizer,
        loss=self.training_setting.loss,
        timeseries_column=normalize_dataset.timeseries_column,
        target_columns=normalize_dataset.target_columns,
        feature_columns=normalize_dataset.feature_columns,
        normalize_training_dataset=normalize_dataset.training_dataset,
        normalize_validation_dataset=normalize_dataset.validation_dataset,
        normalize_test_dataset=normalize_dataset.test_dataset,
        scaler=scaler
      )

      model_training_configs.append(mtc)

    return model_training_configs

  def execute(self) -> List[ModelTrainingConfig]:
    return self._prepare_model_traing_config()