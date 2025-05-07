import torch
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class Evaluator:
    @staticmethod
    def evaluate_model(model, val_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("label").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch)
                preds = logits.argmax(1)
                y_true.extend(labels.cpu())
                y_pred.extend(preds.cpu())

        report = classification_report(y_true, y_pred, digits=3, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = float(re.search(r"accuracy\s+([0-9.]+)", report).group(1))
        f1 = float(re.search(r"macro avg\s+[0-9.\s]+\s([0-9.]+)\s+[0-9.]+", report).group(1))
        return {"report": report, "cm": cm, "acc": acc, "f1": f1}

    @staticmethod
    def plot_results(results_df, best_cm):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(results_df))
        width = 0.35
        ax.bar(x - width/2, results_df["acc"], width, label="Accuracy")
        ax.bar(x + width/2, results_df["f1"], width, label="F1-Score")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Run {i}" for i in x])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Experiment Run")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_title("Validation Metrics by Run")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(best_cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Claim", "Claim"])
        ax.set_yticklabels(["No Claim", "Claim"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Best Model Confusion Matrix")
        for i in range(2):
            for j in range(2):
                color = "white" if best_cm[i, j] > best_cm.max() / 2 else "black"
                ax.text(j, i, best_cm[i, j], ha="center", va="center", color=color)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
