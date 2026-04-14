import torch
import numpy as np

from sklearn.linear_model import LinearRegression

from models.training_config import TrainingConfig
from models.lstm_training import ModelTrainedResult

from typing import List

class ModelEnsembleExecutor:
  def __init__(self, training_config: TrainingConfig, model_trained_results: List[ModelTrainedResult]):
    self.ensemble = training_config.ensemble
    self.model_trained_results = model_trained_results
    

  def _ensemble_model(self) -> LinearRegression:
    pass

  def execute(self):
    pass