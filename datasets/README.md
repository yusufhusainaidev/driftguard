# Datasets Folder

Place your raw CSV files here before running the pipeline.

## Required Files

| File | Source |
|---|---|
| `loan_approval_dataset.csv` | Primary training data |
| `BankChurners.csv` | Drift injection source |

## How to Use

1. Copy your CSV files into this folder
2. Run the data pipeline:
   ```bash
   python scripts/prepare_real_data.py
   ```
3. This generates 6 monthly batches in `data/`
4. Then train the model:
   ```bash
   python scripts/train_baseline.py
   ```
