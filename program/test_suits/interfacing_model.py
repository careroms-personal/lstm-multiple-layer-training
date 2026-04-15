import pickle, json, torch
import pandas as pd
import numpy as np

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load metadata
_TRAINED_MODEL_DIR = Path("./test_config/training/model")

with open(_TRAINED_MODEL_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

print(f"[INFO] Ensemble method: {metadata['method']}")
print(f"[INFO] Models: {metadata['models']}")

# 2. Load Ensemble
with open(_TRAINED_MODEL_DIR / "ensemble_model.pkl", "rb") as f:  # ✅ pkl
    ensemble_model = pickle.load(f)

print(f"[INFO] Ensemble model loaded: {type(ensemble_model)}")

# 3. Load models
lstm_models = {}

for model_name in metadata['models']:
    model_dir  = _TRAINED_MODEL_DIR / "models" / model_name  # ✅ no comma, models
    model      = torch.jit.load(str(model_dir / "model.pt"))
    checkpoint = torch.load(model_dir / "metadata.pth", weights_only=False)

    lstm_models[model_name] = {
        "model":          model,
        "scaler":         checkpoint["scaler"],   # ✅ scaler not scalar
        "target_columns": checkpoint['target_columns'],
        "window_size":    checkpoint['window_size'],
    }

    print(f"[INFO] Loaded: {model_name} window_size={checkpoint['window_size']}")

# 4. Load test data
_TEST_INTERFACE_DATASET = Path("./test_config/test-sample-dataset/disk_sim_for_test_5min.csv")

df = pd.read_csv(_TEST_INTERFACE_DATASET)
df = df.sort_values("elapsed_s").reset_index(drop=True)

strip_ratio = 0.2
n = int(len(df) * strip_ratio)
test_df = df.iloc[n:-n]

print(f"[INFO] Dataset loaded: {len(test_df)} rows")

# 5. Generate predictions
all_predictions = []
min_samples = None

for model_name, config in lstm_models.items():
    model       = config["model"].to(device)  # ✅ move model to GPU
    scaler      = config["scaler"]
    window_size = config["window_size"]
    target_cols = config["target_columns"]

    data_scaled = scaler.transform(test_df[target_cols])
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    X = []
    for i in range(len(data_tensor) - window_size):
        X.append(data_tensor[i:i + window_size])

    X = torch.stack(X)  # ✅ CPU only!

    model.eval()
    all_preds = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size].to(device)  # ✅ batch to GPU
            pred = model(X_batch)
            all_preds.append(pred.cpu())               # ✅ back to CPU

    preds = torch.cat(all_preds, dim=0)
    preds_real = scaler.inverse_transform(preds.numpy())
    all_predictions.append(preds_real)

    print(f"[INFO] {model_name} predicted: {len(preds_real)} samples")

    if min_samples is None or len(preds_real) < min_samples:
        min_samples = len(preds_real)

last_target_cols = target_cols

# 6. Trim and ensemble
all_predictions = [p[-min_samples:] for p in all_predictions]
meta_X = np.column_stack(all_predictions)
final_predictions = ensemble_model.predict(meta_X)

# 7. Validate against actuals
max_window_size = max(config["window_size"] for config in lstm_models.values())
actuals = test_df[last_target_cols].values[max_window_size:]
actuals_trimmed = actuals[-min_samples:]

# 8. Build comparison DataFrame
epsilon = 1e-8

compare_df = pd.DataFrame({
    "actual":      actuals_trimmed.flatten(),
    "predicted":   final_predictions.flatten(),
})

compare_df["error"]   = np.abs(compare_df["actual"] - compare_df["predicted"])
compare_df["error_%"] = np.abs((compare_df["actual"] - compare_df["predicted"]) / (compare_df["actual"] + epsilon)) * 100

# 9. Print summary
print(f"\n[INFO] Comparison Summary:")
print(f"  Samples:    {len(compare_df)}")
print(f"  MAE:        {compare_df['error'].mean():.4f}")
print(f"  RMSE:       {np.sqrt((compare_df['error'] ** 2).mean()):.4f}")
print(f"  MAPE:       {compare_df['error_%'].mean():.2f}%")
print(f"  Max error:  {compare_df['error'].max():.4f}")
print(f"  Min error:  {compare_df['error'].min():.4f}")

# 10. Print table sample
print(f"\n[INFO] Sample Comparison (first 10):")
print(compare_df.head(10).to_string(index=True))

# 11. Export to CSV
_OUTPUT_CSV = Path("./test_config/comparison_output.csv")
compare_df.to_csv(_OUTPUT_CSV, index=True, float_format="%.4f")
print(f"\n[INFO] Saved comparison: {_OUTPUT_CSV}")