import torch, time
import numpy as np

from pathlib import Path

from models.lstm_training import ModelTrainedResult, ModelPredictionResult
from models.debug_model import ModelTrainingDebug

from typing import List, Optional

class ModelTestExecutor:
  def __init__(self, model_trained_results: List[ModelTrainedResult], debug_mode: ModelTrainingDebug, logs_path: Optional[str] = None):
    self.model_trained_results = model_trained_results
    self.debug_mode = debug_mode
    self.logs_path = logs_path
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def _run_test_prediction(self):
    total_start = time.time()
    prediction_results = []
    epsilon = 1e-8

    for trained_result in self.model_trained_results:
      if self.logs_path:
        log_file = str(Path(self.logs_path) / f"{trained_result.name}.log")
        self.debug_mode.set_log_file(log_file)

      self.debug_mode.write(f"[INFO] Running test prediction: {trained_result.name}")

      trained_result.model.eval()

      all_predictions = []
      all_actuals = []

      with torch.no_grad():
        for X_batch, y_batch in trained_result.test_dataset:
          X_batch = X_batch.to(self.device)

          prediction = trained_result.model(X_batch)

          all_predictions.append(prediction.cpu().numpy())
          all_actuals.append(y_batch.numpy())

      predictions_normalized = np.concatenate(all_predictions, axis=0)
      actuals_normalized = np.concatenate(all_actuals, axis=0)

      predictions_real = trained_result.scaler.inverse_transform(predictions_normalized)
      actuals_real = trained_result.scaler.inverse_transform(actuals_normalized)

      self.debug_mode.write(f"  Samples predicted: {len(predictions_real)}")

      for i, col in enumerate(trained_result.target_columns):
        col_pred = predictions_real[:, i]
        col_actual = actuals_real[:, i]

        mae       = np.mean(np.abs(col_pred - col_actual))
        rmse      = np.sqrt(np.mean((col_pred - col_actual) ** 2))
        mape      = np.mean(np.abs((col_actual - col_pred) / (col_actual + epsilon))) * 100
        max_error = np.max(np.abs(col_pred - col_actual))
        min_error = np.min(np.abs(col_pred - col_actual))

        self.debug_mode.write(f"  [{col}]")
        self.debug_mode.write(f"    Predicted → min: {col_pred.min():.4f}  max: {col_pred.max():.4f}")
        self.debug_mode.write(f"    Actual    → min: {col_actual.min():.4f}  max: {col_actual.max():.4f}")
        self.debug_mode.write(f"    MAE:       {mae:.4f}")
        self.debug_mode.write(f"    RMSE:      {rmse:.4f}")
        self.debug_mode.write(f"    MAPE:      {mape:.2f}%")
        self.debug_mode.write(f"    Max error: {max_error:.4f}")
        self.debug_mode.write(f"    Min error: {min_error:.4f}")

      result = ModelPredictionResult(
        name=trained_result.name,
        model=trained_result.model,
        scaler=trained_result.scaler,
        target_columns=trained_result.target_columns,
        predictions=predictions_real,
        actuals=actuals_real,
      )

      self.debug_mode.log(self.debug_mode.model_test_executor, result)
      prediction_results.append(result)

    total_elapsed = time.time() - total_start
    self.debug_mode.write(f"[INFO] Total prediction time: {total_elapsed:.1f}s")

    self.debug_mode.log(self.debug_mode.model_test_executor, prediction_results)

    return prediction_results

  def execute(self):
    return self._run_test_prediction()