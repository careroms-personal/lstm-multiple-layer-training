from models.training_config import TrainingConfig
from models.lstm_training import ModelTrainedResult
from models.ensemble_model import EnsembleResult
from typing import List

from .stacking_ensemble_executor import StackingEnsembleExecutor

class ModelEnsembleExecutor:
  def __init__(self, training_config: TrainingConfig, model_trained_results: List[ModelTrainedResult]):
    self.ensemble = training_config.ensemble
    self.model_trained_results = model_trained_results
    
  def _ensemble_model(self):
    match self.ensemble.method.lower():
      case "stacking":
        ensemble_executor = StackingEnsembleExecutor(model_trained_results=self.model_trained_results, ensemble=self.ensemble)

      case _:
        raise ValueError(f"Unsupported method: {self.ensemble.method}")
      
    return ensemble_executor.execute()

  def execute(self):
    ensemble_model_result = self._ensemble_model()
    
    ensemble_result = EnsembleResult(
      method=self.ensemble.method,
      ensemble_model=ensemble_model_result,
      model_trained_results=self.model_trained_results,
    )

    return ensemble_result