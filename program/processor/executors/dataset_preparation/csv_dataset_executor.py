import sys
import pandas as pd

from pathlib import Path
from typing import Optional
from models.datasources_config_model import Files
from models.debug_model import DatasetPreparationDebug

class CSVDatasetExecutor:
  def __init__(self, datasource_config: Files, debug_mode: Optional[DatasetPreparationDebug] = None):
    self.debug_mode = debug_mode or DatasetPreparationDebug()
    self.file_dir = Path(datasource_config.file_dir)
    self.file_list = datasource_config.file_list

  def _load_csv_to_dataframe(self) -> pd.DataFrame:
    if not self.file_dir.exists():
      print(f"[ERROR] file_dir not found: {self.file_dir}")
      sys.exit(1)

    if self.file_list:
      paths = [self.file_dir / f for f in self.file_list]
    else:
      paths = sorted(self.file_dir.glob("*.csv"))

    if not paths:
      print(f"[ERROR] No CSV files found in: {self.file_dir}")
      sys.exit(1)

    missing = [p for p in paths if not p.exists()]

    if missing:
      for p in missing:
        print(f"[ERROR] CSV file not found: {p}")
      sys.exit(1)

    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

  def executor(self) -> pd.DataFrame:
    result = self._load_csv_to_dataframe()
    self.debug_mode.log(self.debug_mode.sub_executor, result)
    return result
