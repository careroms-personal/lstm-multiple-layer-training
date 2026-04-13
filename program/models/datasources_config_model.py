from pydantic import BaseModel
from typing import List, Literal

class Files(BaseModel):
  file_dir: str
  file_list: List[str] = []
  format: Literal["csv", "parquet", "json"] = "csv"