# Training Flow

End-to-end flow from CLI to trained ensemble model.

---

## 1. Entry — `main.py`

```
$ python main.py -c path/to/config.yaml
```

- Parses `-c` / `--config` argument
- Creates `Processor(config_path)` and calls `.execute()`

---

## 2. Config Load — `Processor._load_and_validate_config()`

- Checks config file exists → ❌ exit if not
- `yaml.safe_load()` → raw dict
- `TrainingConfig(**yaml_data)` → pydantic validates all fields
  - `Datasource.model_validator` → coerces `config` dict into correct model (`Files`, etc.)
- Sets `config_base_dir` = resolved parent directory of the config file
- ❌ exit on any validation error with per-field messages

---

## 3. Dataset Preparation — `DatasetPreparationExecutor`

### 3a. Load raw data
- Dispatches to `CSVDatasetExecutor` based on `datasource.type` + `datasource.config.format`
- If `file_list` is set → loads those files only
- If `file_list` is empty → globs all `*.csv` in `file_dir`
- Concatenates all files into one `pd.DataFrame`

### 3b. Chronological split
Sorts by `timeseries_column`, then cuts from the tail:

```
|←——————— training (70%) ———————→|←— validation (20%) —→|←— test (10%) —→|
                                                           earliest ——————→ latest
```

Ratios are controlled by `validation_data_ratio` and `test_data_ratio` in config.

### 3c. Column ordering
Reorders columns to: `[feature_columns..., target_columns...]`

### Output → `ModelTrainingDataset`
```
training_dataset    pd.DataFrame
validation_dataset  pd.DataFrame
test_dataset        pd.DataFrame
```

---

## 4. Model Preparation — `ModelPreparationExecutor`

### 4a. Fit scaler
- Fits `MinMaxScaler` on `target_columns` of **train + validation** data only
- Test set is never seen during fitting

### 4b. Normalize
- Applies fitted scaler to `target_columns` in all three splits (train, validation, test)

### 4c. Build per-model configs
For each model defined in `lstm_models`, creates a `ModelTrainingConfig`:

```
ModelTrainingConfig
├── Architecture  →  from LSTMModel config (name, window_size, units, dropout, float_type)
├── Hyperparams   →  from TrainingSetting (epochs, batch_size, patience, optimizer, loss)
└── Datasets      →  normalized train / validation / test DataFrames
```

### Output → `List[ModelTrainingConfig]`
One entry per LSTM model defined in config.

---

## 5. LSTM Training — `[ planned ]`

For each `ModelTrainingConfig`:
- Build LSTM layers from `units` list
- Apply `dropout` between layers
- Compile with `optimizer` and `loss`
- Train with `EarlyStopping(patience=patience)`
- Evaluate on validation set per epoch

---

## 6. Ensemble — `[ planned ]`

- Collect predictions from all trained LSTM models on the **validation set**
- Stack predictions as input features to meta-learner
- Fit `linear_regression` meta-learner (`ensemble.meta_learner`)
- Evaluate stacked ensemble on **test set**

---

## 7. Output — `[ planned ]`

Controlled by `output` config block:

| Flag | What it does |
|---|---|
| `print_output.training_logs` | Print epoch-by-epoch loss to console |
| `print_output.model_summary` | Print LSTM architecture summary |
| `print_output.ensemble_weights` | Print meta-learner coefficients |
| `print_output.training_data` | Print dataset shape/stats |
| `write_output.training_data_path` | Save train/val/test splits to disk |
| `write_output.logs_path` | Save training logs to disk |
| `model_output_path` | Save trained models to disk |

All paths are relative to the config file's directory.
