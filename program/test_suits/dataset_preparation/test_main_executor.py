import pytest
import yaml

from pathlib import Path
from models.datasources_config_model import Files
from models.training_config import TrainingConfig
from models.lstm_model import ModelTrainingDataset
from processor.executors.dataset_preparation.csv_dataset_executor import CSVDatasetExecutor
from processor.executors.dataset_preparation.main_executor import DatasetPreparationExecutor

TEST_CONFIG_PATH = Path(__file__).parent.parent / "test_config" / "config.yaml"

TOTAL_ROWS = 25921
TEST_SIZE  = int(TOTAL_ROWS * 0.1)                    # 2592
VAL_SIZE   = int(TOTAL_ROWS * 0.2)                    # 5184
TRAIN_SIZE = TOTAL_ROWS - TEST_SIZE - VAL_SIZE         # 18145


@pytest.fixture
def training_config():
  with open(TEST_CONFIG_PATH) as f:
    yaml_data = yaml.safe_load(f)
  config = TrainingConfig(**yaml_data)
  config.config_base_dir = str(TEST_CONFIG_PATH.parent.resolve())
  return config


@pytest.fixture
def raw_dataset(training_config):
  files_config = training_config.datasource.config
  resolved_dir = Path(training_config.config_base_dir) / files_config.file_dir
  resolved_config = Files(
    file_dir=str(resolved_dir),
    file_list=files_config.file_list,
    format=files_config.format,
  )
  return CSVDatasetExecutor(resolved_config).executor()


@pytest.fixture
def executor(training_config):
  return DatasetPreparationExecutor(training_config)


# --- _split_test_data ---

class TestSplitTestData:

  def test_returns_model_training_dataset(self, executor, raw_dataset):
    result = executor._split_test_data(raw_dataset)
    assert isinstance(result, ModelTrainingDataset)

  def test_training_row_count(self, executor, raw_dataset):
    result = executor._split_test_data(raw_dataset)
    assert len(result.training_dataset) == TRAIN_SIZE

  def test_validation_row_count(self, executor, raw_dataset):
    result = executor._split_test_data(raw_dataset)
    assert len(result.validation_dataset) == VAL_SIZE

  def test_test_row_count(self, executor, raw_dataset):
    result = executor._split_test_data(raw_dataset)
    assert len(result.test_dataset) == TEST_SIZE

  def test_total_rows_preserved(self, executor, raw_dataset):
    result = executor._split_test_data(raw_dataset)
    total = len(result.training_dataset) + len(result.validation_dataset) + len(result.test_dataset)
    assert total == TOTAL_ROWS

  def test_splits_are_chronological(self, executor, raw_dataset, training_config):
    result = executor._split_test_data(raw_dataset)
    ts = training_config.training_data.timeseries_column
    assert result.training_dataset[ts].max() < result.validation_dataset[ts].min()
    assert result.validation_dataset[ts].max() < result.test_dataset[ts].min()

  def test_each_split_index_starts_at_zero(self, executor, raw_dataset):
    result = executor._split_test_data(raw_dataset)
    assert result.training_dataset.index[0] == 0
    assert result.validation_dataset.index[0] == 0
    assert result.test_dataset.index[0] == 0

  def test_metadata_passed_to_result(self, executor, raw_dataset, training_config):
    result = executor._split_test_data(raw_dataset)
    td = training_config.training_data
    assert result.timeseries_column == td.timeseries_column
    assert result.target_columns == td.target_columns
    assert result.feature_columns == td.feature_columns


# --- _prepare_column ---

class TestPrepareColumn:

  def test_returns_only_target_columns_when_no_features(self, executor, raw_dataset, training_config):
    split = executor._split_test_data(raw_dataset)
    result = executor._prepare_column(split.training_dataset)
    assert list(result.columns) == training_config.training_data.target_columns

  def test_excludes_non_target_columns(self, executor, raw_dataset, training_config):
    split = executor._split_test_data(raw_dataset)
    result = executor._prepare_column(split.training_dataset)
    non_target = [c for c in raw_dataset.columns if c not in training_config.training_data.target_columns]
    assert all(c not in result.columns for c in non_target)

  def test_row_count_unchanged_after_column_prep(self, executor, raw_dataset):
    split = executor._split_test_data(raw_dataset)
    result = executor._prepare_column(split.training_dataset)
    assert len(result) == TRAIN_SIZE
