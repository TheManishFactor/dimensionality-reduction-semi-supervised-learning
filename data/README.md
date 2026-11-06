# Data directory

This directory holds datasets and derived artifacts. The contents of `raw/`
and `processed/` are git-ignored and must not be committed.

## Current status

| Dataset | Source | Local location | Status |
|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | Kaggle `uciml/breast-cancer-wisconsin-data` | `data/raw/breast_cancer/data.csv` | Downloaded and verified |
| MNIST Digit Recognizer | Kaggle Digit Recognizer competition files | `data/raw/mnist/train.csv`, `data/raw/mnist/test.csv` | Downloaded and verified |
| UCI HAR | UCI 240 via UCI-owned Kaggle copy `uciml/human-activity-recognition-with-smartphones` | `data/raw/har/train.csv`, `data/raw/har/test.csv` | Downloaded and verified |

No data were fabricated. The loader reads only the files described below and
raises a clear error when an expected file is missing.

## Layout

```text
data/
├── raw/
│   ├── breast_cancer/
│   │   └── data.csv
│   ├── mnist/
│   │   ├── train.csv        # labelled: 42,000 rows, 784 pixels + label
│   │   └── test.csv         # unlabelled: 28,000 rows, 784 pixels
│   └── har/
│       ├── train.csv        # 561 features + subject + Activity
│       └── test.csv         # 561 features + subject + Activity
├── processed/
└── README.md
```

## 1. Breast Cancer Wisconsin (Diagnostic)

Source: <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>

The Kaggle dataset archive contains a single `data.csv`. It can be downloaded
without authentication through Kaggle's public API:

```bash
curl -L --fail -o breast_cancer.zip \
  "https://www.kaggle.com/api/v1/datasets/download/uciml/breast-cancer-wisconsin-data"
unzip -o breast_cancer.zip data.csv -d data/raw/breast_cancer
```

If the download fails, download the archive from the Kaggle page in a browser,
extract `data.csv`, and place it at `data/raw/breast_cancer/data.csv`.

Schema notes:

- Target column: `diagnosis` (`B` = benign, `M` = malignant).
- Non-feature columns: `id` (patient identifier) and `diagnosis`.
- `Unnamed: 32` is a fully empty artifact column and is dropped by the loader.
- Numeric features: 30.
- Rows: 569.

## 2. MNIST Digit Recognizer

Official source: <https://www.kaggle.com/competitions/digit-recognizer/data>

The official competition download requires a Kaggle account, acceptance of the
competition rules, and Kaggle API authentication. It **cannot** be downloaded
anonymously.

### Secure Kaggle authentication

Never place credentials in source code or commit them. Use one of:

1. **Token file (recommended):** on Kaggle, open *Settings -> API -> Create New
   Token*. Move the downloaded `kaggle.json` to:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Restrict permissions (Linux/macOS): `chmod 600 ~/.kaggle/kaggle.json`
2. **Environment variables:** set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your
   shell or secret manager.

`kaggle.json` and `.kaggle/` are excluded by `.gitignore`.

### Official download

Accept the rules on the competition *Data* page, then run:

```bash
pip install kaggle
kaggle competitions download -c digit-recognizer -p data/raw/mnist
unzip -o data/raw/mnist/digit-recognizer.zip -d data/raw/mnist
```

Expected files: `train.csv` (42,000 labelled rows, 785 columns) and `test.csv`
(28,000 unlabelled rows, 784 columns).

### Current staged files

Because no Kaggle credentials were configured in this environment, the
competition download endpoint returned HTTP 401. To avoid blocking Phase 2
without concealment, the staged files were retrieved from the public Kaggle
dataset copy `khotijahs1/digitrecognizer`:

```bash
curl -L --fail -o mnist_digit_recognizer.zip \
  "https://www.kaggle.com/api/v1/datasets/download/khotijahs1/digitrecognizer"
unzip -o mnist_digit_recognizer.zip train.csv test.csv -d data/raw/mnist
```

This copy contains `train.csv` with 42,000 samples, 784 pixel columns, and the
ten digit classes 0-9, plus a 28,000-row unlabelled `test.csv`. It matches the
competition layout. To use the official competition files instead, follow the
authenticated download above and replace `data/raw/mnist/train.csv` (the loader
requires no code changes).

## 3. Human Activity Recognition Using Smartphones

Source: UCI Machine Learning Repository dataset 240, obtained through the
UCI-owned Kaggle copy:
<https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones>

The direct UCI archive transfer was unreliable in this environment (it reset
mid-download), so the UCI-owned Kaggle copy was used. It preserves the original
561 sensor features and subject identifiers. This substitution is explicit, not
silent.

```bash
curl -L --fail -o har_kaggle.zip \
  "https://www.kaggle.com/api/v1/datasets/download/uciml/human-activity-recognition-with-smartphones"
unzip -o har_kaggle.zip train.csv test.csv -d data/raw/har
```

Schema notes:

- Columns: 563 = 561 sensor features + `subject` + `Activity`.
- Features: 561.
- Activities: 6: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING,
  STANDING, LAYING.
- Samples: 7,352 train + 2,947 test = 10,299.
- Subjects: preserved in the `subject` column. The loader exposes them as
  `DatasetSplit.subjects`. Do not discard them, because a later phase requires
  subject-wise train/test separation.

The loader also supports the original UCI text layout (`features.txt`,
`activity_labels.txt`, `train/X_train.txt`, `test/X_test.txt`, and
`subject_*.txt`) if you prefer the direct UCI archive. Place it directly under
`data/raw/har/` or inside `data/raw/har/UCI HAR Dataset/`; no code changes are
needed.

## Verification

From the project root:

```bash
python scripts/check_datasets.py
```

The script reports dimensions, feature counts, class distributions, missing
values, duplicate rows, subject counts, and overall usability. It exits with
code 0 only when all datasets are usable.

## Rules

- Do not modify files in `raw/`; keep them immutable.
- Derived or cached data belong in `processed/`.
- Datasets are not committed to version control.
- Never commit Kaggle credentials or other secrets.
