class Config:
    SEED = 42
    BATCH_SIZE = 64
    MAX_EPOCHS = 6
    MAX_SEQ_LEN = 512
    MODEL_NAME = "prajjwal1/bert-tiny"
    DATASET_NAME = "climatebert/environmental_claims"
    HYPERPARAMETER_CONFIGS = [
        {"lr": 3e-5, "freeze_layers": 0, "weight_decay": 0.01},
        {"lr": 5e-5, "freeze_layers": 2, "weight_decay": 0.00},
        {"lr": 2e-5, "freeze_layers": 0, "weight_decay": 0.01},
        {"lr": 3e-5, "freeze_layers": 4, "weight_decay": 0.00},
    ]
    EARLY_STOP_PATIENCE = 2
    EARLY_STOP_METRIC = "val_f1"
    EARLY_STOP_MODE = "max"
    WARMUP_RATIO = 0.1
    LOG_EVERY_N_STEPS = 10
