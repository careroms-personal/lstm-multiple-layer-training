import torch, time
import torch.nn as nn

from typing import List, Optional

from models.training_config import TrainingConfig
from models.lstm_architecture import ModelArchitectureConfig
from models.lstm_training import ModelTrainingConfig, ModelTrainedResult
from models.optimizer_model import AdamConfig, SGDConfig

from .model_build_executor import ModelBuildExecutor
from .model_test_executor import ModelTestExecutor

class ModelTrainingExecutor:
  def __init__(self, training_config: TrainingConfig, model_configs: List[ModelArchitectureConfig], logs_path: Optional[str] = None):
    self.debug_mode = training_config.debug.model_training
    self.debug = training_config.debug
    self.ensemble = training_config.ensemble
    self.training_setting = training_config.training_setting
    self.model_configs = model_configs
    self.logs_path = logs_path
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def _get_optimizer(self, model: nn.Module, opt_config) -> torch.optim.Optimizer:

    match opt_config.type.lower():
      case "adam":
        opt_config: AdamConfig = opt_config

        return torch.optim.Adam(
          model.parameters(),
          lr=opt_config.learning_rate,
          betas=(opt_config.beta1, opt_config.beta2),
          eps=opt_config.eps,
          weight_decay=opt_config.weight_decay,
        )
      
      case "sgd":
        opt_config: SGDConfig = opt_config

        return torch.optim.SGD(
          model.parameters(),
          lr=opt_config.learning_rate,
          momentum=opt_config.momentum,
          weight_decay=opt_config.weight_decay,
        )
      
      case _:
        raise ValueError(f"Unsupported optimizer: {opt_config.type}")
  
  def _get_loss(self, loss_name: str) -> nn.Module:
    match loss_name.lower():
      case "mse":
        return nn.MSELoss()
      
      case "mae":
        return nn.L1Loss()
      
      case _:
        raise ValueError(f"Unsupported loss: {loss_name}")

  def _run_training(self, model_results: List[ModelTrainingConfig]) -> List[ModelTrainedResult]:
    total_start = time.time()
    model_trained_results = []

    for model_result in model_results:
      if self.logs_path:
        self.debug.configure_file_output(self.logs_path, model_result.name)

      self.debug_mode.write(f"[INFO] Training model: {model_result.name}")

      model = model_result.model.to(self.device)
      optimizer = self._get_optimizer(model=model, opt_config=model_result.optimizer)
      loss_fn = self._get_loss(loss_name=model_result.loss)

      best_val_loss = float("inf")
      patience_counter = 0
      best_weights = None

      for epoch in range(model_result.epochs):

        # training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in model_result.training_dataset:
          X_batch = X_batch.to(self.device)
          y_batch = y_batch.to(self.device)

          optimizer.zero_grad()
          predictions = model(X_batch)
          loss = loss_fn(predictions, y_batch)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_result.gradient_clip)
          optimizer.step()

          train_loss += loss.item()

        train_loss /= len(model_result.training_dataset)

        # validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
          for X_batch, y_batch in model_result.val_dataset:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            val_loss += loss.item()

        val_loss /= len(model_result.val_dataset)

        self.debug_mode.write(f"  Epoch {epoch + 1}/{model_result.epochs} — loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        # early stopping
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          patience_counter = 0
          best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
          patience_counter += 1
          if patience_counter >= model_result.patience:
            self.debug_mode.write(f"  [EarlyStopping] Stopped at epoch {epoch + 1}")
            break
      
      if best_weights:
        model.load_state_dict(best_weights)

      model.eval()

      trained_model = ModelTrainedResult(
        name=model_result.name,
        model=model,
        scaler=model_result.scaler,
        target_columns=model_result.target_columns,
        unit=model_result.unit,
        dropout=model_result.dropout,
        n_features=model_result.n_features,
        batch_size=model_result.batch_size,
        windows_size=model_result.windows_size,
        test_dataset=model_result.test_dataset,
        val_dataset=model_result.val_dataset,
      )

      model_trained_results.append(trained_model)

    total_elapsed = time.time() - total_start
    self.debug_mode.write(f"[INFO] Total training time: {total_elapsed:.1f}s")

    self.debug_mode.log(self.debug_mode.main_executor, model_trained_results)

    return model_trained_results

  def execute(self) -> List[ModelTrainedResult]:
    model_results = []

    for model_config in self.model_configs:
      build_executor = ModelBuildExecutor(model_config, self.debug)
      result = build_executor.execute()
      model_results.append(result)

    trained_results = self._run_training(model_results=model_results)
    
    test_executor = ModelTestExecutor(model_trained_results=trained_results, debug_mode=self.debug_mode, logs_path=self.logs_path)
    test_executor.execute()

    return trained_results