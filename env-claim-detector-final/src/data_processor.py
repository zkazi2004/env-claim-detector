import re
import logging
import numpy as np
import torch
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def load_dataset(self):
        logger.info(f"Loading dataset: {self.config.DATASET_NAME}")
        env_ds = datasets.load_dataset(self.config.DATASET_NAME)
        return env_ds["train"], env_ds["validation"]

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s\.,!?']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_batch(self, batch):
        return {"text": self.clean_text(batch["text"])}

    def tokenize_batch(self, batch):
        return self.tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=self.config.MAX_SEQ_LEN
        )

    def prepare_datasets(self):
        train_ds, val_ds = self.load_dataset()
        train_ds = train_ds.map(self.clean_batch)
        val_ds = val_ds.map(self.clean_batch)
        train_ds = train_ds.map(self.tokenize_batch, batched=True, remove_columns=["text"])
        val_ds = val_ds.map(self.tokenize_batch, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")
        val_ds.set_format("torch")
        return train_ds, val_ds

    def create_dataloaders(self, train_ds, val_ds):
        train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.BATCH_SIZE, pin_memory=True)
        return train_loader, val_loader

    def calculate_class_weights(self, train_ds):
        cls_counts = np.bincount(train_ds["label"])
        weights = torch.tensor(cls_counts.sum() / (len(cls_counts) * cls_counts), dtype=torch.float)
        return weights
