# Environmental Claim Detector

This project fine-tunes a TinyBERT model using PyTorch Lightning to classify environmental claims in corporate reports, using the `climatebert/environmental_claims` dataset.

## 🚀 Features

- TinyBERT (prajjwal1/bert-tiny) for fast training
- Handles class imbalance with dynamic weighting
- 4 hyperparameter search runs
- Best result: **76.6% accuracy**, **0.739 macro F1**

## 📁 Folder Structure

- `src/` – Python modules for config, model, preprocessing, and evaluation
- `results/` – Experiment summary and classification report
- `main.py` – Launches training, evaluation, and plotting

## 📦 Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python main.py
```

## 📝 Results Summary

See `results/summary.csv` and `results/classification_report.txt`

## 📄 License

MIT License  
© 2025 Zayd Kazi
