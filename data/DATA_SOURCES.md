# 📦 Sample Data

This folder contains sample and helper scripts for working with the BTC/USD dataset used in TPS Core.

## 📊 Files

- `btcusd.csv` — ~10K candles for quick testing and demos  
- `download_full_dataset.py` — Script to download and prepare the full dataset

## 🧪 Sample vs Full Dataset

- The sample file (`btcusd.csv`) is small and suitable for quick tests.
- The **full dataset (~7.3M candles)** is downloaded automatically from Kaggle using the script below.

## ⬇️ How to Download the Full Dataset

From the `data/` directory, run:

```bash
python download_full_dataset.py
