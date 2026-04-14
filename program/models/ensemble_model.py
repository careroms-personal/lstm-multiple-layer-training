from dataclasses import dataclass
from typing import Any, List

from .lstm_training import ModelTrainedResult

@dataclass
class EnsembleResult:
  method: str
  ensemble_model: Any
  model_trained_results: List[ModelTrainedResult]