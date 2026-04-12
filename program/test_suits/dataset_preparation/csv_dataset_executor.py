import pytest
import pandas as pd

from pathlib import Path
from models.datasources_config import Files
from processor.executors.dataset_preparation.csv_dataset_executor import CSVDatasetExecutor

SAMPLE_DIR = Path(__file__).parent.parent / "test_config" / "sample-dataset"
SAMPLE_FILE = "disk_sim_5min.csv"
SAMPLE_ROW_COUNT = 25921
SAMPLE_COLUMNS = ["step", "elapsed_s", "elapsed_days", "disk_gb"]


class TestCSVDatasetExecutor:

  # --- Happy paths ---

  def test_load_single_file_from_file_list(self):
    config = Files(file_dir=str(SAMPLE_DIR), file_list=[SAMPLE_FILE])
    result = CSVDatasetExecutor(config).executor()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == SAMPLE_ROW_COUNT
    assert list(result.columns) == SAMPLE_COLUMNS

  def test_load_all_csvs_from_dir_when_file_list_is_empty(self):
    config = Files(file_dir=str(SAMPLE_DIR), file_list=[])
    result = CSVDatasetExecutor(config).executor()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == SAMPLE_ROW_COUNT
    assert list(result.columns) == SAMPLE_COLUMNS

  def test_load_multiple_files_from_file_list_are_concatenated(self):
    config = Files(file_dir=str(SAMPLE_DIR), file_list=[SAMPLE_FILE, SAMPLE_FILE])
    result = CSVDatasetExecutor(config).executor()

    assert len(result) == SAMPLE_ROW_COUNT * 2

  def test_concat_resets_index(self):
    config = Files(file_dir=str(SAMPLE_DIR), file_list=[SAMPLE_FILE, SAMPLE_FILE])
    result = CSVDatasetExecutor(config).executor()

    assert result.index[0] == 0
    assert result.index[-1] == (SAMPLE_ROW_COUNT * 2) - 1

  # --- Error paths ---

  def test_exits_when_file_dir_not_found(self, tmp_path):
    config = Files(file_dir=str(tmp_path / "nonexistent"), file_list=[])

    with pytest.raises(SystemExit):
      CSVDatasetExecutor(config).executor()

  def test_exits_when_no_csv_files_in_dir(self, tmp_path):
    config = Files(file_dir=str(tmp_path), file_list=[])

    with pytest.raises(SystemExit):
      CSVDatasetExecutor(config).executor()

  def test_exits_when_file_in_file_list_not_found(self, tmp_path):
    config = Files(file_dir=str(tmp_path), file_list=["missing.csv"])

    with pytest.raises(SystemExit):
      CSVDatasetExecutor(config).executor()

  def test_exits_when_one_of_multiple_files_in_file_list_not_found(self, tmp_path):
    (tmp_path / "exists.csv").write_text("col\n1\n")
    config = Files(file_dir=str(tmp_path), file_list=["exists.csv", "missing.csv"])

    with pytest.raises(SystemExit):
      CSVDatasetExecutor(config).executor()
