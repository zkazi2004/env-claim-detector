import torch
import logging
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class EnvironmentalClaimClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes=2, lr=3e-5, freeze_layers=0, weight_decay=0.01, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

        if freeze_layers > 0:
            logger.info(f"Freezing {freeze_layers} encoder layers")
            for param in self.model.bert.encoder.layer[:freeze_layers].parameters():
                param.requires_grad_(False)

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, **x):
        return self.model(**x).logits

    def _shared_step(self, batch):
        labels = batch.pop("label")
        logits = self(**batch)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def training_step(self, batch, _):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, labels = self._shared_step(batch)
        self.acc.update(preds, labels)
        self.f1.update(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", self.f1, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.config.WARMUP_RATIO * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
