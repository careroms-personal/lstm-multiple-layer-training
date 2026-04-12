import pandas as pd

from models.training_config import TrainingConfig
from models.lstm_model import ModelTrainingDataset

from .csv_dataset_executor import CSVDatasetExecutor

class DatasetPreparationExecutor:
  def __init__(self, training_config: TrainingConfig):
    self.datasources = training_config.datasource
    self.training_data = training_config.training_data

    self.files_executor = {
      "csv": CSVDatasetExecutor
    }

  def _load_datasource(self) -> pd.DataFrame:
    match self.datasources.type:
      case "files":
        datasource_executor = self.files_executor[self.datasources.config.get("format")](self.datasources.config)

      case _:
        raise ValueError(f"Unsupported datasource type: {self.datasources.type}")

    return datasource_executor.executor()
  
  def _split_test_data(self, raw_dataset: pd.DataFrame):
    df = raw_dataset.sort_values(self.training_data.timeseries_column).reset_index(drop=True)

    test_size = int(len(df) * self.training_data.test_data_ratio)
    val_size = int(len(df) * self.training_data.validation_data_ratio)

    training_dataset = df.iloc[:-(test_size + val_size)].reset_index(drop=True)
    validation_dataset = df.iloc[-(test_size + val_size):-test_size].reset_index(drop=True)
    test_dataset = df.iloc[-test_size:].reset_index(drop=True)

    return ModelTrainingDataset(
      training_dataset=training_dataset, 
      validation_dataset=validation_dataset, 
      test_dataset=test_dataset,
      timeseries_column=self.training_data.timeseries_column,
      target_columns=self.training_data.target_columns,
      feature_columns=self.training_data.feature_columns,
    )
  
  def _prepare_column(self, df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = (
      self.training_data.feature_columns + self.training_data.target_columns
    )

    return df[ordered_columns]
    
  def executor(self) -> ModelTrainingDataset:
    raw_dataset = self._load_datasource()
    model_training_dataset = self._split_test_data(raw_dataset)

    model_training_dataset.training_dataset = self._prepare_column(model_training_dataset.training_dataset)
    model_training_dataset.validation_dataset = self._prepare_column(model_training_dataset.validation_dataset)
    model_training_dataset.test_dataset = self._prepare_column(model_training_dataset.test_dataset)

    return model_training_dataset