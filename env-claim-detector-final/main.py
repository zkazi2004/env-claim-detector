import logging
import pytorch_lightning as pl
from src.config import Config
from src.data_processor import DataProcessor
from src.model import EnvironmentalClaimClassifier
from src.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate(config: Config):
    pl.seed_everything(config.SEED, workers=True)
    processor = DataProcessor(config)
    train_ds, val_ds = processor.prepare_datasets()
    train_loader, val_loader = processor.create_dataloaders(train_ds, val_ds)
    class_weights = processor.calculate_class_weights(train_ds)
    logger.info(f"Class weights: {class_weights}")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=config.EARLY_STOP_METRIC,
        mode=config.EARLY_STOP_MODE,
        patience=config.EARLY_STOP_PATIENCE
    )

    trainer = pl.Trainer(
        accelerator="gpu" if pl.utilities.device_parser.num_cuda_devices() > 0 else "cpu",
        devices=1,
        max_epochs=config.MAX_EPOCHS,
        deterministic=True,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        callbacks=[early_stopping],
        enable_checkpointing=False
    )

    results, confusion_matrices = [], []
    for i, hp in enumerate(config.HYPERPARAMETER_CONFIGS):
        logger.info(f"▶️ Run {i + 1}: {hp}")
        model = EnvironmentalClaimClassifier(config.MODEL_NAME, class_weights=class_weights, **hp)
        trainer.fit(model, train_loader, val_loader)
        eval_res = Evaluator.evaluate_model(model, val_loader)
        results.append({**hp, "acc": eval_res["acc"], "f1": eval_res["f1"]})
        confusion_matrices.append(eval_res["cm"])
        print(eval_res["report"])

    import pandas as pd
    import numpy as np
    results_df = pd.DataFrame(results)
    best_idx = results_df.f1.idxmax()
    best_config = results[best_idx]
    best_cm = confusion_matrices[best_idx]

    print("\n=== Results Summary ===")
    print(results_df)
    print(f"\nBest run: {best_config}")

    Evaluator.plot_results(results_df, best_cm)

if __name__ == "__main__":
    config = Config()
    train_and_evaluate(config)
