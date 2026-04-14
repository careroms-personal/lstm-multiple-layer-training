import torch
import numpy as np

from typing import List, Tuple
from sklearn.linear_model import LinearRegression  # type: ignore

from functions.print_analyze_metrics import print_metrics
from models.training_config import Ensemble, StackingConfig
from models.lstm_training import ModelTrainedResult

class StackingEnsembleExecutor:
  def __init__(self, model_trained_results: List[ModelTrainedResult], ensemble: Ensemble):
    self.model_trained_results = model_trained_results
    self.ensemble = ensemble
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def _get_meta_learner(self):
    stacking: StackingConfig = self.ensemble.stacking
    match stacking.meta_learner.lower():
      case "linear_regression":
        return LinearRegression()
      case _:
        raise ValueError(f"Unsupported meta_learner: {stacking.meta_learner}")

  def _get_predictions(self, trained_result: ModelTrainedResult, dataset) -> Tuple[np.ndarray, np.ndarray]:
    all_preds = []
    all_acts = []

    with torch.no_grad():
      for X_batch, y_batch in dataset:
        X_batch = X_batch.to(self.device)
        predictions = trained_result.model(X_batch)
        all_preds.append(predictions.cpu().numpy())
        all_acts.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    acts  = np.concatenate(all_acts, axis=0)

    preds = trained_result.scaler.inverse_transform(preds)
    acts  = trained_result.scaler.inverse_transform(acts)

    return preds, acts

  def _stacking_ensemble_model(self):
    all_val_predictions  = []
    all_test_predictions = []
    val_actuals  = None
    test_actuals = None

    for trained_result in self.model_trained_results:
      trained_result.model.eval()

      # val predictions → train meta-learner
      val_preds, val_acts = self._get_predictions(trained_result, trained_result.val_dataset)
      all_val_predictions.append(val_preds)
      if val_actuals is None:
        val_actuals = val_acts

      # test predictions → evaluate ensemble
      test_preds, test_acts = self._get_predictions(trained_result, trained_result.test_dataset)
      all_test_predictions.append(test_preds)
      if test_actuals is None:
        test_actuals = test_acts

    min_val_size  = min(p.shape[0] for p in all_val_predictions)
    min_test_size = min(p.shape[0] for p in all_test_predictions)

    all_val_predictions  = [p[-min_val_size:]  for p in all_val_predictions]
    all_test_predictions = [p[-min_test_size:] for p in all_test_predictions]

    val_actuals  = val_actuals[-min_val_size:]
    test_actuals = test_actuals[-min_test_size:]
    

    # stack predictions as features
    meta_X_val  = np.column_stack(all_val_predictions)
    meta_X_test = np.column_stack(all_test_predictions)

    # train meta-learner
    meta_learner = self._get_meta_learner()
    meta_learner.fit(meta_X_val, val_actuals)

    # evaluate ensemble
    ensemble_predictions = meta_learner.predict(meta_X_test)
    print_metrics(ensemble_predictions, test_actuals)

    return meta_learner

  def execute(self):
    return self._stacking_ensemble_model()
