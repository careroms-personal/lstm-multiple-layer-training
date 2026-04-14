import torch
import json
import pickle

from pathlib import Path
from models.training_config import TrainingConfig
from models.ensemble_model import EnsembleResult

class ModelExporterExecutor:
  def __init__(self, training_config: TrainingConfig, ensembled_result: EnsembleResult):
    self.config_base_dir = training_config.config_base_dir
    self.model_output_path = training_config.output.model_output_path
    self.ensembled_result = ensembled_result

  def _export_model(self):
    output_path = Path(self.config_base_dir) / self.model_output_path
    output_path.mkdir(parents=True, exist_ok=True)

    models_path = output_path / "models"
    models_path.mkdir(exist_ok=True)

    # 1. save each LSTM model
    models_metadata = []
    for trained_result in self.ensembled_result.model_trained_results:
      model_file = models_path / f"{trained_result.name}.pth"

      torch.save({
        "state_dict": trained_result.model.state_dict(),
        "scaler": trained_result.scaler,
        "target_columns": trained_result.target_columns,
      }, model_file)

      print(f"[INFO] Saved model: {model_file}")
      models_metadata.append(trained_result.name)

    # 2. save ensemble_model
    ensemble_file = output_path / "ensemble_model.pk1"
    with open(ensemble_file, "wb") as f:
      pickle.dump(self.ensembled_result.ensemble_model, f)

    print(f"[INFO] Saved ensemble model: {ensemble_file}")

    metadata = {
      "method": self.ensembled_result.method,
      "models": models_metadata,
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
      json.dump(metadata, f , indent=2)

    print(f"[INFO] Saved metadata: {metadata_file}")

  def execute(self):
    print(f"[INFO] Exporting model to: {self.model_output_path}")
    self._export_model()
    print(f"[INFO] Export complete!")