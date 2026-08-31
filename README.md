# loss-stability

Reproducible code and archived analysis outputs for the manuscript:

**A Repeated-Run Evaluation of Soft-F2 and Weighted BCE in Edge-Based Image Classification under Class Imbalance**

This repository contains the analysis pipeline used to compare a differentiable Soft-F2 surrogate loss with weighted binary cross-entropy (BCE) under class imbalance. The primary experiment uses BeeSpotter Apis-versus-Bombus images; an additional controlled-imbalance CIFAR-10 Bird-versus-Frog experiment is used as an external public-image stress test.

## What is reproduced

The authoritative analysis is `run_analysis.py`. It implements the settings reported in the manuscript:

- 10 repeated stratified train/validation/test splits with seeds `42, 73, 101, 137, 211, 307, 401, 509, 613, 719`
- 56% training, 14% validation, and 30% held-out test data
- grayscale conversion, resize to 200 × 200, 3 × 3 median filtering, Sobel gradient magnitude, and per-image normalization
- flattened 40,000-dimensional edge representation
- compact neural network: `40,000 -> 64 (ReLU) -> 1`
- Adam optimizer, learning rate 0.001, batch size 256, maximum 30 epochs, no weight decay
- early stopping on validation loss with patience 5 and minimum improvement `1e-5`
- Soft-F2 with beta = 2, calculated per training minibatch
- weighted BCE using PyTorch `BCEWithLogitsLoss`; `pos_weight = n_negative / n_positive`, calculated from the training subset only for each split
- validation-only threshold selection over 0.05–0.95 in increments of 0.01
- Brier score, ECE (10 bins), log loss, ROC-AUC, average precision, MCC, class-specific F-scores, threshold sensitivity, and collapse diagnostics
- controlled CIFAR-10 Bird-versus-Frog subsampling with seed 2026, matching the BeeSpotter class counts (827 positive/minority and 3,142 negative/majority)
- paired seed-level Wilcoxon comparisons and 10,000-resample bootstrap confidence intervals for mean paired differences

## Repository structure

```text
loss-stability/
├── run_analysis.py                 # Authoritative end-to-end analysis
├── requirements.txt                # Python dependencies
├── results/                        # Archived outputs from the final analysis run
├── supplementary/
│   ├── Supplementary_Table_S1.csv
│   └── Supplementary_Table_S1.pdf
├── CITATION.cff
├── LICENSE
├── .gitignore
└── README.md
```

The earlier modular scripts used during development are intentionally not included in the final repository because they contained obsolete architecture and threshold-selection code that does not correspond to the final manuscript.

## Data

The image data are not redistributed in this repository.

### BeeSpotter

The analysis expects a ZIP archive containing:

```text
train.csv
train_images/
    <id>.jpg
    ...
```

`train.csv` must contain at least the columns `id` and `genus`. The final analysis verifies the source-data class counts before training:

- raw `genus = 0`: 827 images, mapped to **Apis = 1** (positive/minority)
- raw `genus = 1`: 3,142 images, mapped to **Bombus = 0** (negative/majority)

Place the archive at the repository root as `image_data.zip`, or set the environment variable `BEESPOTTER_ZIP` to its location.

### CIFAR-10

The Bird-versus-Frog benchmark is constructed automatically from the public CIFAR-10 training partition through `torchvision`. Bird is the positive/minority class and Frog the negative/majority class. Deterministic subsampling uses seed 2026 and reproduces the BeeSpotter class counts exactly.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the analysis

From the repository root:

```bash
python run_analysis.py
```

By default, generated files are written to `analysis_outputs/` and cached image features to `analysis_cache/`. These paths can be overridden with `LOSS_STABILITY_OUTPUT_DIR` and `LOSS_STABILITY_CACHE_DIR`.

The complete run is computationally intensive because every image is converted to a 40,000-dimensional representation and both losses are trained across 10 repeated splits on two datasets. GPU execution is recommended.

## Archived final outputs

The `results/` directory contains the tables and figures produced by the final analysis run used for the manuscript, including:

- repeated-run metric summaries
- split audits and training histories
- stability and collapse-rate summaries
- paired statistical comparisons
- representative-seed threshold and probability tables
- mean ROC/precision–recall and calibration figures
- the full analysis configuration

The manuscript reports mean ± standard deviation for primary repeated-run summaries. Paired Wilcoxon tests and bootstrap confidence intervals are treated as supplementary evidence because repeated partitions come from the same underlying datasets and multiple metrics are examined.

## Supplementary Table S1

`supplementary/Supplementary_Table_S1.csv` and `.pdf` contain the paired seed-level comparisons reported as Supplementary Table S1, including mean paired differences, 10,000-resample bootstrap 95% confidence intervals, and two-sided Wilcoxon signed-rank test results.

## Reproducibility note

The final pipeline resets the random seed before training each loss within a split so that Soft-F2 and weighted BCE begin from the same model initialization. Decision thresholds are selected from validation data only and are then applied once to the held-out test set.

## Citation

Citation metadata are provided in `CITATION.cff`. A permanent archived DOI will be added after archival.

## License

Code is released under the MIT License. Dataset use remains subject to the terms of the original data providers.
