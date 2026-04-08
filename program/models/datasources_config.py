from pydantic import BaseModel
from typing import List

class Files(BaseModel):
  file_dir: str
  file_list: List[str] = []