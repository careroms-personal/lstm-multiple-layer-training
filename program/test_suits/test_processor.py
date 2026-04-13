import pytest
import yaml

from pathlib import Path
from models.training_config import TrainingConfig
from processor.processor import Processor

TEST_CONFIG_PATH = Path(__file__).parent / "test_config" / "config.yaml"


@pytest.fixture
def valid_config_path():
  return str(TEST_CONFIG_PATH)


@pytest.fixture
def absolute_config_path(tmp_path):
  """Test config with file_dir resolved to absolute path — safe to run from any CWD."""
  with open(TEST_CONFIG_PATH) as f:
    yaml_data = yaml.safe_load(f)

  sample_dir = TEST_CONFIG_PATH.parent / "sample-dataset"
  yaml_data["datasource"]["config"]["file_dir"] = str(sample_dir.resolve())
  yaml_data["output"]["model_output_path"] = str(tmp_path / "models")

  config_file = tmp_path / "config.yaml"
  with open(config_file, "w") as f:
    yaml.dump(yaml_data, f)

  return str(config_file)


# --- Config loading ---

class TestLoadAndValidateConfig:

  def test_loads_valid_config(self, valid_config_path):
    processor = Processor(valid_config_path)
    assert isinstance(processor.training_config, TrainingConfig)

  def test_config_base_dir_is_set_to_config_parent(self, valid_config_path):
    processor = Processor(valid_config_path)
    expected = str(TEST_CONFIG_PATH.parent.resolve())
    assert processor.training_config.config_base_dir == expected

  def test_config_fields_loaded_correctly(self, valid_config_path):
    processor = Processor(valid_config_path)
    tc = processor.training_config
    assert tc.datasource.type == "files"
    assert tc.training_data.timeseries_column == "elapsed_s"
    assert tc.training_data.target_columns == ["disk_gb"]
    assert len(tc.lstm_models) == 2

  def test_exits_when_config_file_not_found(self, tmp_path):
    with pytest.raises(SystemExit):
      Processor(str(tmp_path / "nonexistent.yaml"))

  def test_exits_when_config_has_missing_required_fields(self, tmp_path):
    invalid = tmp_path / "config.yaml"
    invalid.write_text("datasource:\n  type: files\n")

    with pytest.raises(SystemExit):
      Processor(str(invalid))

  def test_exits_when_config_has_wrong_field_type(self, tmp_path):
    with open(TEST_CONFIG_PATH) as f:
      yaml_data = yaml.safe_load(f)

    yaml_data["training_data"]["validation_data_ratio"] = "not_a_float"

    invalid = tmp_path / "config.yaml"
    with open(invalid, "w") as f:
      yaml.dump(yaml_data, f)

    with pytest.raises(SystemExit):
      Processor(str(invalid))


# --- execute ---

class TestExecute:

  def test_execute_completes_without_error(self, absolute_config_path):
    processor = Processor(absolute_config_path)
    processor.execute()
