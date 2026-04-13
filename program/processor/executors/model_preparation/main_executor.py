import pandas as pd

from typing import List
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from models.training_config import TrainingConfig
from models.lstm_architecture import ModelTrainingDataset, ModelArchitectureConfig

class ModelPreparationExecutor:
  def __init__(self, training_config: TrainingConfig, model_training_dataset: ModelTrainingDataset):
    self.debug_mode = training_config.debug.model_preparation
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

    self.debug_mode.log(self.debug_mode.main_executor, scaler)

    return scaler

  def _normalize_training_data(self, dataset: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    target_columns = self.model_training_dataset.target_columns
    feature_columns = self.model_training_dataset.feature_columns or []

    # normalize target columns only
    normalized_targets = scaler.transform(dataset[target_columns])
    normalized_df = pd.DataFrame(normalized_targets, columns=target_columns)

    # keep feature columns as-is
    if feature_columns:
      feature_df = dataset[feature_columns].reset_index(drop=True)
      result = pd.concat([feature_df, normalized_df], axis=1)  # features first, targets last
    else:
      result = normalized_df

    self.debug_mode.log(self.debug_mode.main_executor, result)

    return result

  def _prepare_model_traing_config(self) -> List[ModelArchitectureConfig]:
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
      mtc = ModelArchitectureConfig(
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
        gradient_clip=self.training_setting.gradient_clip,
        timeseries_column=normalize_dataset.timeseries_column,
        target_columns=normalize_dataset.target_columns,
        feature_columns=normalize_dataset.feature_columns,
        normalize_training_dataset=normalize_dataset.training_dataset,
        normalize_validation_dataset=normalize_dataset.validation_dataset,
        normalize_test_dataset=normalize_dataset.test_dataset,
        scaler=scaler
      )

      model_training_configs.append(mtc)

    self.debug_mode.log(self.debug_mode.main_executor, model_training_configs)

    return model_training_configs

  def execute(self) -> List[ModelArchitectureConfig]:
    result = self._prepare_model_traing_config()
    self.debug_mode.log(self.debug_mode.main_executor, result)
    return result
