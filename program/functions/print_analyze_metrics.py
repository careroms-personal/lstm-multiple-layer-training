import numpy as np

def print_metrics(predictions: np.ndarray, actuals: np.ndarray):
  epsilon = 1e-8
  mae  = np.mean(np.abs(predictions - actuals))
  rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
  mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100

  print(f"[INFO] Ensemble metrics:")
  print(f"  MAE:  {mae:.4f}")
  print(f"  RMSE: {rmse:.4f}")
  print(f"  MAPE: {mape:.2f}%")