import pytest
import yaml

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from models.datasources_config_model import Files
from models.training_config import TrainingConfig
from models.lstm_architecture import ModelArchitectureConfig
from processor.executors.dataset_preparation.csv_dataset_executor import CSVDatasetExecutor
from processor.executors.dataset_preparation.main_executor import DatasetPreparationExecutor
from processor.executors.model_preparation.main_executor import ModelPreparationExecutor

TEST_CONFIG_PATH = Path(__file__).parent.parent / "test_config" / "config.yaml"

TRAIN_SIZE = 18145
VAL_SIZE   = 5184
TEST_SIZE  = 2592


@pytest.fixture
def training_config():
  with open(TEST_CONFIG_PATH) as f:
    yaml_data = yaml.safe_load(f)
  config = TrainingConfig(**yaml_data)
  config.config_base_dir = str(TEST_CONFIG_PATH.parent.resolve())
  return config


@pytest.fixture
def model_training_dataset(training_config):
  files_config = training_config.datasource.config
  resolved_dir = Path(training_config.config_base_dir) / files_config.file_dir
  resolved_config = Files(
    file_dir=str(resolved_dir),
    file_list=files_config.file_list,
    format=files_config.format,
  )
  raw = CSVDatasetExecutor(resolved_config).executor()

  dataset_executor = DatasetPreparationExecutor(training_config)
  split = dataset_executor._split_test_data(raw)
  split.training_dataset = dataset_executor._prepare_column(split.training_dataset)
  split.validation_dataset = dataset_executor._prepare_column(split.validation_dataset)
  split.test_dataset = dataset_executor._prepare_column(split.test_dataset)
  return split


@pytest.fixture
def executor(training_config, model_training_dataset):
  return ModelPreparationExecutor(training_config, model_training_dataset)


# --- _set_min_max_scale ---

class TestSetMinMaxScale:

  def test_returns_fitted_scaler(self, executor):
    scaler = executor._set_min_max_scale()
    assert isinstance(scaler, MinMaxScaler)
    assert hasattr(scaler, "data_min_")

  def test_scaler_fitted_on_train_and_val_only(self, executor, model_training_dataset):
    scaler = executor._set_min_max_scale()
    target = model_training_dataset.target_columns

    train_val_min = min(
      model_training_dataset.training_dataset[target].min().min(),
      model_training_dataset.validation_dataset[target].min().min(),
    )
    train_val_max = max(
      model_training_dataset.training_dataset[target].max().max(),
      model_training_dataset.validation_dataset[target].max().max(),
    )

    assert scaler.data_min_[0] == pytest.approx(train_val_min)
    assert scaler.data_max_[0] == pytest.approx(train_val_max)



# --- _normalize_training_data ---

class TestNormalizeTrainingData:

  def test_returns_only_target_columns(self, executor, model_training_dataset):
    scaler = executor._set_min_max_scale()
    result = executor._normalize_training_data(model_training_dataset.training_dataset, scaler)
    assert list(result.columns) == model_training_dataset.target_columns

  def test_train_values_are_in_zero_one_range(self, executor, model_training_dataset):
    scaler = executor._set_min_max_scale()
    result = executor._normalize_training_data(model_training_dataset.training_dataset, scaler)
    assert result.min().min() >= 0.0
    assert result.max().max() <= 1.0

  def test_row_count_unchanged(self, executor, model_training_dataset):
    scaler = executor._set_min_max_scale()
    result = executor._normalize_training_data(model_training_dataset.training_dataset, scaler)
    assert len(result) == TRAIN_SIZE

  def test_index_is_zero_based(self, executor, model_training_dataset):
    scaler = executor._set_min_max_scale()
    result = executor._normalize_training_data(model_training_dataset.training_dataset, scaler)
    assert result.index[0] == 0
    assert result.index[-1] == TRAIN_SIZE - 1


# --- execute ---

class TestExecute:

  def test_returns_one_config_per_lstm_model(self, executor, training_config):
    result = executor.execute()
    assert len(result) == len(training_config.lstm_models)

  def test_each_result_is_model_training_config(self, executor):
    result = executor.execute()
    assert all(isinstance(c, ModelArchitectureConfig) for c in result)

  def test_architecture_params_match_config(self, executor, training_config):
    result = executor.execute()
    for config, lstm_model in zip(result, training_config.lstm_models):
      assert config.name == lstm_model.name
      assert config.window_size == lstm_model.window_size
      assert config.units == lstm_model.units
      assert config.dropout == lstm_model.dropout

  def test_training_params_match_config(self, executor, training_config):
    result = executor.execute()
    ts = training_config.training_setting
    for config in result:
      assert config.epochs == ts.epochs
      assert config.batch_size == ts.batch_size
      assert config.patience == ts.patience
      assert config.optimizer == ts.optimizer
      assert config.loss == ts.loss

  def test_all_configs_share_same_scaler(self, executor):
    result = executor.execute()
    assert result[0].scaler is result[1].scaler

  def test_normalized_dataset_row_counts(self, executor):
    result = executor.execute()
    for config in result:
      assert len(config.normalize_training_dataset) == TRAIN_SIZE
      assert len(config.normalize_validation_dataset) == VAL_SIZE
      assert len(config.normalize_test_dataset) == TEST_SIZE
