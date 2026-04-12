# lstm-multiple-layer-training — Project Overview
<!-- last ai update: 2026 Apr 12 -->

Quick reference for AI assistants to understand this project's purpose, structure, and data flow.
See `vibe-code-rule.yaml` for coding rules. See `AI-PYTHON-GUIDE.md` for patterns.

---

## Purpose

Train multiple LSTM models on time-series data (e.g. disk usage over time), then combine them into a stacking ensemble. Driven entirely by a YAML config file — no code changes needed to run different scenarios.

---

## Project Structure

```
lstm-multiple-layer-trainning/
├── vibe-code-rule.yaml
├── pyproject.toml
├── TRAINING-FLOW.md                         # human-readable program flow tracker
├── .ai/
│   ├── AI-PROJECT-OVERVIEW.md               # this file
│   ├── AI-PRINCIPLE-GUIDE.md
│   └── AI-PYTHON-GUIDE.md
└── program/
    ├── app/
    │   └── main.py                          # CLI entry point
    ├── models/
    │   ├── datasources_config.py            # Files datasource model
    │   ├── training_config.py               # AppConfig (TrainingConfig) + all sub-models
    │   └── lstm_model.py                    # Runtime models: ModelTrainingDataset, ModelTrainingConfig
    ├── processor/
    │   ├── processor.py                     # Orchestrator
    │   └── executors/
    │       ├── dataset_preparation/
    │       │   ├── main_executor.py         # DatasetPreparationExecutor
    │       │   └── csv_dataset_executor.py  # CSVDatasetExecutor
    │       └── model_preparation/
    │           └── main_executor.py         # ModelPreparationExecutor
    ├── global_config.py                     # Reserved — do not add models here
    ├── config_templates/
    │   └── config.yaml                      # YAML template with all valid fields
    └── test_suits/
        ├── global_test_config.py
        └── test_config/
            ├── config.yaml                  # Test config: disk depletion scenario
            └── sample-dataset/
                ├── disk_sim_5min.csv        # Generated mock data (5-min intervals, 90 days)
                └── db_depletion_mock.py     # Script that generated the mock CSV
```

---

## Config Model Structure (`training_config.py`)

```
TrainingConfig
├── config_base_dir: str          # Set by Processor after load — base dir of the config file
├── datasource: Datasource
│   ├── type: DatasourceType      # "files" (more types planned)
│   └── config: Files             # validated by model_validator against type
│       ├── file_dir: str
│       ├── file_list: List[str]  # empty = load all *.csv in file_dir
│       └── format: "csv"|"parquet"|"json"
├── training_data: TrainingData
│   ├── validation_data_ratio: float
│   ├── test_data_ratio: float
│   ├── timeseries_column: str    # column used to sort chronologically
│   ├── target_columns: List[str]
│   └── feature_columns: List[str]
├── lstm_models: List[LSTMModel]
│   └── name, window_size, units, dropout, float_type
├── training_setting: TrainingSetting
│   └── epochs, batch_size, patience, optimizer, loss
├── ensemble: Ensemble
│   └── method: "stacking", meta_learner: "linear_regression"
└── output: Output
    ├── model_output_path: str
    ├── print_output: Optional[PrintOutput]   # training_data, training_logs, model_summary, ensemble_weights
    └── write_output: Optional[WriteOutput]   # training_data_path, logs_path
```

---

## Runtime Model Structure (`lstm_model.py`)

```
ModelTrainingDataset (CustomBaseModel)       # passed from DatasetPreparation → ModelPreparation
├── training_dataset: pd.DataFrame
├── validation_dataset: pd.DataFrame
└── test_dataset: pd.DataFrame

ModelTrainingConfig (CustomBaseModel)        # one per LSTMModel, output of ModelPreparation
├── name, window_size, units, dropout, float_type   # from LSTMModel config
├── epochs, batch_size, patience, optimizer, loss   # from TrainingSetting
└── normalize_training/validation/test_dataset: pd.DataFrame
```

---

## Executor Chain

```
Processor
├── DatasetPreparationExecutor(training_config)
│   ├── CSVDatasetExecutor           → raw pd.DataFrame
│   ├── _split_test_data()           → train / validation / test DataFrames (chronological)
│   ├── _prepare_column()            → reorder cols: [feature_columns + target_columns]
│   └── returns: ModelTrainingDataset
│
└── ModelPreparationExecutor(training_config, model_training_dataset)
    ├── _set_min_max_scale()         → fits MinMaxScaler on train+validation target cols
    ├── _normalize_training_data()   → normalizes target cols in all 3 splits
    ├── _prepare_model_training_config() → builds ModelTrainingConfig per LSTMModel
    └── returns: List[ModelTrainingConfig]
```

---

## Key Design Decisions

- **All paths in config are relative to the config file's directory** — `config_base_dir` is injected by Processor after loading.
- **Chronological splits only** — data is sorted by `timeseries_column` before splitting. No shuffling.
- **Scaler fitted on train+validation only** — test set is never seen during normalization.
- **One `ModelTrainingConfig` per LSTM model** — enables each model to have a different `window_size` for the same dataset (e.g. day-scale vs week-scale predictions).
- **`Datasource.config` is dynamically validated** — `model_validator` maps `type` → config model class via `_DATASOURCE_CONFIG_MAP`.

---

## Test Scenario

Config: `program/test_suits/test_config/config.yaml`
- Data: disk usage (GB) sampled every 5 minutes over 90 days (~25,920 rows)
- Target: `disk_gb`
- Two models: `day_prediction` (window=288 steps = 1 day), `week_prediction` (window=2016 steps = 1 week)
- Splits: 70% train / 20% validation / 10% test
