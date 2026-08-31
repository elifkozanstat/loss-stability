# ============================================================
# Reproducible analysis code for:
# "A Repeated-Run Evaluation of Soft-F2 and Weighted BCE in
#  Edge-Based Image Classification under Class Imbalance"
#
# This script reproduces the manuscript analysis:
# 1) BeeSpotter label mapping: Apis=1 (positive/minority),
#    Bombus=0 (negative/majority)
# 2) Ten repeated stratified train/validation/test splits
# 3) Identical compact neural architecture for both losses
# 4) Validation-only threshold selection and untouched test evaluation
# 5) Calibration diagnostics (Brier score, ECE, log loss, reliability plots)
# 6) Stability diagnostics (probability collapse and single-class prediction)
# 7) Controlled-imbalance CIFAR-10 Bird-versus-Frog stress test
# 8) Export of manuscript-ready tables, figures, and paired statistics
#
# The computational settings match the final manuscript and archived
# result tables included in this repository.
# ============================================================

# -----------------------------
# 0. PACKAGE CHECK / INSTALL
# -----------------------------
import sys
import subprocess
import importlib.util

REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
}

missing = [pip_name for module_name, pip_name in REQUIRED.items()
           if importlib.util.find_spec(module_name) is None]
if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

# -----------------------------
# 1. IMPORTS
# -----------------------------
import os
import gc
import glob
import copy
import json
import math
import time
import random
import shutil
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

from tqdm.auto import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# 2. USER CONFIGURATION
# -----------------------------
# Repository / data paths.
# Set BEESPOTTER_ZIP in the environment if the archive is stored elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BEESPOTTER_ZIP = os.environ.get("BEESPOTTER_ZIP", str(PROJECT_ROOT / "image_data.zip"))

# Output/cache directories. These can also be overridden through environment variables.
OUTPUT_DIR = Path(os.environ.get("LOSS_STABILITY_OUTPUT_DIR", str(PROJECT_ROOT / "analysis_outputs")))
CACHE_DIR = Path(os.environ.get("LOSS_STABILITY_CACHE_DIR", str(PROJECT_ROOT / "analysis_cache")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Repeated-run design
# Ten repeated stratified splits are used in the manuscript.
N_REPEATS = 10
SEEDS = [42, 73, 101, 137, 211, 307, 401, 509, 613, 719][:N_REPEATS]

# Split design: 30% untouched test; remaining 70% -> 80/20 train/validation
# Final proportions = 56% train, 14% validation, 30% test.
TEST_SIZE = 0.30
VAL_WITHIN_TRAIN = 0.20

# Image preprocessing
IMG_SIZE = 200
MEDIAN_KERNEL = 3

# Model/training (same architecture and optimizer for BOTH losses)
HIDDEN_UNITS = 64
BATCH_SIZE = 256
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-5
SOFT_F_BETA = 2.0

# Threshold selection is done ONLY on validation data.
# The manuscript's focus is minority-class Apis F1, so threshold is chosen
# by validation Apis F1; ties are broken by validation Macro-F1, then by
# closeness to 0.5.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
SENSITIVITY_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]

# Calibration / collapse diagnostics
ECE_BINS = 10
PROB_COLLAPSE_STD = 1e-4

# Additional publicly available image benchmark.
RUN_SECONDARY_DATASET = True
SECONDARY_DATASET_NAME = "CIFAR10_Bird_vs_Frog_ControlledImbalance"
SECONDARY_POSITIVE_CLASS = "Bird"
SECONDARY_NEGATIVE_CLASS = "Frog"
# Match BeeSpotter class counts exactly for a controlled imbalance experiment.
SECONDARY_N_POSITIVE = 827
SECONDARY_N_NEGATIVE = 3142
SECONDARY_SUBSET_SEED = 2026

# Save a zipped bundle at the end
AUTO_DOWNLOAD_RESULTS = False  # set True in Colab if you want automatic download

# -----------------------------
# 3. REPRODUCIBILITY
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print("Repeated seeds:", SEEDS)

# -----------------------------
# 4. PREPROCESSING
# -----------------------------
def median_sobel_feature_from_gray(gray_uint8: np.ndarray,
                                   img_size: int = IMG_SIZE) -> np.ndarray:
    """Replicates the intended controlled edge pipeline.

    Steps:
      1) resize grayscale image to 200x200
      2) 3x3 median filter
      3) Sobel x/y gradients
      4) gradient magnitude
      5) per-image [0,1] normalization
      6) flatten to 40,000 features
    """
    gray = cv2.resize(gray_uint8, (img_size, img_size), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(gray, MEDIAN_KERNEL)
    med = med.astype(np.float32) / 255.0

    gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    max_val = float(mag.max())
    if max_val > 0:
        mag /= max_val

    return mag.reshape(-1).astype(np.float32)


def median_sobel_feature_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return median_sobel_feature_from_gray(gray)


def median_sobel_feature_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return median_sobel_feature_from_gray(gray)

# -----------------------------
# 5. BEESPOTTER LOADING + LABEL AUDIT
# -----------------------------
def resolve_beespotter_zip() -> str:
    """Find a VALID BeeSpotter zip rather than trusting the filename.

    Colab may keep an old/corrupt /content/image_data.zip while the newly uploaded
    valid file is named image_data(1).zip. We therefore inspect every candidate
    and select the first real ZIP that contains train.csv and train_images/.
    """
    candidates = []

    # Prefer the explicitly configured path, then all similarly named uploads.
    if os.path.exists(BEESPOTTER_ZIP):
        candidates.append(BEESPOTTER_ZIP)
    candidates.extend(sorted(glob.glob(str(PROJECT_ROOT / "image_data*.zip"))))
    candidates.extend(sorted(glob.glob("/content/image_data*.zip")))

    # Local fallback for sandbox/notebook execution.
    candidates.extend(sorted(glob.glob("/mnt/data/image_data*.zip")))

    # De-duplicate while preserving order.
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        raise FileNotFoundError(
            "BeeSpotter zip not found. Upload image_data.zip (or image_data(1).zip) "
            "to /content and rerun."
        )

    print("BeeSpotter ZIP candidates found:")
    for p in candidates:
        try:
            size_mb = os.path.getsize(p) / (1024 ** 2)
        except OSError:
            size_mb = float('nan')
        print(f"  - {p} ({size_mb:.2f} MB)")

    invalid = []
    for p in candidates:
        if not zipfile.is_zipfile(p):
            invalid.append((p, "not a valid ZIP archive"))
            continue
        try:
            with zipfile.ZipFile(p, "r") as zf:
                names = zf.namelist()
                has_csv = any(name.rstrip("/").endswith("train.csv") for name in names)
                has_images = any("train_images/" in name for name in names)
                bad_member = zf.testzip()
                if bad_member is not None:
                    invalid.append((p, f"corrupt member: {bad_member}"))
                    continue
                if not has_csv or not has_images:
                    invalid.append((p, "ZIP does not contain train.csv and train_images/"))
                    continue
            print("Using valid BeeSpotter zip:", p)
            return p
        except Exception as exc:
            invalid.append((p, repr(exc)))

    details = "\n".join(f"  - {p}: {reason}" for p, reason in invalid)
    raise zipfile.BadZipFile(
        "No valid BeeSpotter archive was found among the uploaded image_data*.zip files.\n"
        + details
        + "\nDelete the invalid file(s), re-upload the original BeeSpotter ZIP, and rerun."
    )


def build_beespotter_cache():
    dataset_dir = CACHE_DIR / "beespotter_extracted"
    csv_path = dataset_dir / "train.csv"
    image_dir = dataset_dir / "train_images"

    if not csv_path.exists() or not image_dir.exists():
        zip_path = resolve_beespotter_zip()
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting BeeSpotter dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_dir)

    df = pd.read_csv(csv_path)
    if not {"id", "genus"}.issubset(df.columns):
        raise ValueError("train.csv must contain 'id' and 'genus' columns.")

    raw_counts = df["genus"].value_counts().sort_index().to_dict()
    print("\nBeeSpotter raw label counts:", raw_counts)

    # VERIFIED FROM THE UPLOADED DATA:
    # raw genus=0 -> Apis (827; minority)
    # raw genus=1 -> Bombus (3142; majority)
    if raw_counts.get(0, 0) != 827 or raw_counts.get(1, 0) != 3142:
        raise ValueError(
            "Unexpected BeeSpotter label counts. Expected raw genus 0=827 and genus 1=3142. "
            f"Observed: {raw_counts}. Stop and verify dataset version before analysis."
        )

    # Correct study encoding required by the manuscript:
    # Apis = 1 = positive/minority; Bombus = 0 = negative/majority
    y = (df["genus"].to_numpy() == 0).astype(np.int64)
    mapped_counts = dict(zip(*np.unique(y, return_counts=True)))
    print("Study encoding counts (0=Bombus, 1=Apis):", mapped_counts)

    audit = pd.DataFrame({
        "raw_genus": [0, 1],
        "raw_count": [raw_counts[0], raw_counts[1]],
        "true_class": ["Apis", "Bombus"],
        "study_label": [1, 0],
        "study_role": ["positive/minority", "negative/majority"],
    })
    audit.to_csv(OUTPUT_DIR / "beespotter_label_audit.csv", index=False)

    x_path = CACHE_DIR / f"beespotter_X_{IMG_SIZE}.npy"
    y_path = CACHE_DIR / "beespotter_y.npy"

    if not x_path.exists() or not y_path.exists():
        print("Building BeeSpotter Sobel feature cache (first run only)...")
        n = len(df)
        d = IMG_SIZE * IMG_SIZE
        X_mm = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float16, shape=(n, d)
        )

        for i, row in tqdm(df.iterrows(), total=n, desc="BeeSpotter preprocessing"):
            img_path = image_dir / f"{int(row['id'])}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Image not found/readable: {img_path}")
            X_mm[i] = median_sobel_feature_from_bgr(img).astype(np.float16)

        del X_mm
        np.save(y_path, y)
    else:
        # Verify cached labels still match the study encoding.
        cached_y = np.load(y_path)
        if not np.array_equal(cached_y, y):
            print("Cached labels do not match the study mapping; rewriting y cache.")
            np.save(y_path, y)

    X = np.load(x_path).astype(np.float32)  # ~635 MB; manageable in standard Colab RAM
    y = np.load(y_path).astype(np.int64)

    meta = {
        "dataset": "BeeSpotter",
        "positive_class": "Apis",
        "negative_class": "Bombus",
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_total": int(len(y)),
        "imbalance_ratio_negative_to_positive": float((y == 0).sum() / (y == 1).sum()),
        "imbalance_type": "natural",
    }
    return X, y, meta

# -----------------------------
# 6. SECOND PUBLIC DATASET
# -----------------------------
def build_cifar10_bird_frog_cache():
    """Creates a controlled-imbalance public benchmark from CIFAR-10.

    Positive/minority: Bird (class 2), n=827
    Negative/majority: Frog (class 6), n=3142

    The class ratio exactly matches BeeSpotter, allowing us to ask whether the
    loss-function behavior reproduces on a distinct public image source while
    holding the imbalance level approximately constant.
    """
    x_path = CACHE_DIR / f"cifar10_bird_frog_X_{IMG_SIZE}.npy"
    y_path = CACHE_DIR / "cifar10_bird_frog_y.npy"
    selection_path = CACHE_DIR / "cifar10_bird_frog_selection.csv"

    if not x_path.exists() or not y_path.exists():
        print("\nDownloading/loading CIFAR-10 public benchmark...")
        cifar = CIFAR10(root=str(CACHE_DIR / "cifar10_raw"), train=True, download=True)
        data = np.asarray(cifar.data)       # RGB uint8, shape [50000,32,32,3]
        targets = np.asarray(cifar.targets)

        bird_idx = np.where(targets == 2)[0]
        frog_idx = np.where(targets == 6)[0]

        if len(bird_idx) < SECONDARY_N_POSITIVE or len(frog_idx) < SECONDARY_N_NEGATIVE:
            raise ValueError("CIFAR-10 does not contain enough Bird/Frog samples for requested subset.")

        rng = np.random.default_rng(SECONDARY_SUBSET_SEED)
        bird_sel = rng.choice(bird_idx, size=SECONDARY_N_POSITIVE, replace=False)
        frog_sel = rng.choice(frog_idx, size=SECONDARY_N_NEGATIVE, replace=False)

        selected = np.concatenate([bird_sel, frog_sel])
        y = np.concatenate([
            np.ones(len(bird_sel), dtype=np.int64),   # Bird = positive/minority = 1
            np.zeros(len(frog_sel), dtype=np.int64),  # Frog = negative/majority = 0
        ])

        # Shuffle once only to remove class ordering; repeated splits happen later.
        order = rng.permutation(len(selected))
        selected = selected[order]
        y = y[order]

        pd.DataFrame({
            "cifar_train_index": selected,
            "study_label": y,
            "class_name": np.where(y == 1, "Bird", "Frog"),
        }).to_csv(selection_path, index=False)

        print("Building CIFAR-10 Bird/Frog Sobel feature cache (first run only)...")
        n = len(selected)
        d = IMG_SIZE * IMG_SIZE
        X_mm = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float16, shape=(n, d)
        )

        for i, idx in enumerate(tqdm(selected, desc="CIFAR-10 preprocessing")):
            X_mm[i] = median_sobel_feature_from_rgb(data[idx]).astype(np.float16)

        del X_mm
        np.save(y_path, y)

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    if int((y == 1).sum()) != SECONDARY_N_POSITIVE or int((y == 0).sum()) != SECONDARY_N_NEGATIVE:
        raise ValueError("Secondary cache has unexpected class counts; delete cache and rerun.")

    meta = {
        "dataset": SECONDARY_DATASET_NAME,
        "positive_class": SECONDARY_POSITIVE_CLASS,
        "negative_class": SECONDARY_NEGATIVE_CLASS,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_total": int(len(y)),
        "imbalance_ratio_negative_to_positive": float((y == 0).sum() / (y == 1).sum()),
        "imbalance_type": "controlled by deterministic subsampling of public CIFAR-10 training images",
    }
    return X, y, meta

# -----------------------------
# 7. MODEL + LOSSES
# -----------------------------
class CompactMLP(nn.Module):
    """Identical architecture for both losses."""
    def __init__(self, input_dim: int, hidden_units: int = HIDDEN_UNITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class SoftFBetaLoss(nn.Module):
    """Differentiable Soft-F_beta loss for positive class y=1."""
    def __init__(self, beta: float = SOFT_F_BETA, eps: float = 1e-7):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = torch.sum(probs * targets)
        fp = torch.sum(probs * (1.0 - targets))
        fn = torch.sum((1.0 - probs) * targets)

        beta2 = self.beta ** 2
        soft_f = ((1.0 + beta2) * tp) / (
            (1.0 + beta2) * tp + beta2 * fn + fp + self.eps
        )
        return 1.0 - soft_f


def make_loss(loss_name: str, pos_weight: float):
    if loss_name == "Soft-F2":
        return SoftFBetaLoss(beta=SOFT_F_BETA).to(DEVICE)
    elif loss_name == "Weighted BCE":
        # In PyTorch pos_weight applies ONLY to target y=1.
        # Since y=1 is correctly mapped to the minority class, this is now correct.
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=pw).to(DEVICE)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

# -----------------------------
# 8. SPLITTING + DATALOADERS
# -----------------------------
def repeated_stratified_split(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=VAL_WITHIN_TRAIN,
        random_state=seed,
        stratify=y[trainval_idx],
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def make_loader(X, y, idx, shuffle: bool, seed: int):
    X_t = torch.from_numpy(X[idx])
    y_t = torch.from_numpy(y[idx].astype(np.float32))
    ds = TensorDataset(X_t, y_t)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        drop_last=False,
    )

# -----------------------------
# 9. TRAINING WITH EARLY STOPPING
# -----------------------------
def average_epoch_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = len(yb)
            total += float(loss.item()) * bs
            n += bs
    return total / max(n, 1)


def train_model(X, y, train_idx, val_idx, loss_name: str, seed: int):
    # Reset seed before EACH loss so the two losses start from identical weights.
    set_all_seeds(seed)

    train_loader = make_loader(X, y, train_idx, shuffle=True, seed=seed)
    val_loader = make_loader(X, y, val_idx, shuffle=False, seed=seed)

    n_pos = int(np.sum(y[train_idx] == 1))
    n_neg = int(np.sum(y[train_idx] == 0))
    pos_weight = n_neg / n_pos

    model = CompactMLP(input_dim=X.shape[1], hidden_units=HIDDEN_UNITS).to(DEVICE)
    criterion = make_loss(loss_name, pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = len(yb)
            train_loss_sum += float(loss.item()) * bs
            n_train += bs

        train_loss = train_loss_sum / max(n_train, 1)
        val_loss = average_epoch_loss(model, val_loader, criterion)
        history.append((epoch, train_loss, val_loss))

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_val_loss = val_loss

    model.load_state_dict(best_state)

    info = {
        "loss": loss_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_n_positive": n_pos,
        "train_n_negative": n_neg,
        "pos_weight": float(pos_weight),
        "epochs_run": int(len(history)),
    }
    return model, info, history

# -----------------------------
# 10. PREDICTION / THRESHOLD SELECTION
# -----------------------------
def predict_proba(model, X, y, idx):
    loader = make_loader(X, y, idx, shuffle=False, seed=0)
    probs = []
    targets = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            targets.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(targets).astype(np.int64)


def choose_validation_threshold(y_val, p_val):
    rows = []
    for t in THRESHOLD_GRID:
        pred = (p_val >= t).astype(np.int64)
        f1_pos = f1_score(y_val, pred, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_val, pred, average="macro", zero_division=0)
        rows.append((float(t), float(f1_pos), float(f1_macro)))

    # Maximize positive-class F1; tie-break by Macro-F1; then prefer threshold closest to 0.5.
    rows.sort(key=lambda r: (r[1], r[2], -abs(r[0] - 0.5)), reverse=True)
    return rows[0][0]

# -----------------------------
# 11. CALIBRATION + METRICS
# -----------------------------
def expected_calibration_error(y_true, probs, n_bins=ECE_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Include p=1 in the final bin.
    bin_ids = np.digitize(probs, bins[1:-1], right=True)
    ece = 0.0
    rows = []

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            rows.append({
                "bin": b,
                "bin_lower": bins[b],
                "bin_upper": bins[b + 1],
                "count": 0,
                "mean_confidence": np.nan,
                "fraction_positive": np.nan,
                "abs_gap": np.nan,
            })
            continue

        conf = float(np.mean(probs[mask]))
        frac_pos = float(np.mean(y_true[mask]))
        gap = abs(conf - frac_pos)
        weight = float(np.mean(mask))
        ece += weight * gap
        rows.append({
            "bin": b,
            "bin_lower": bins[b],
            "bin_upper": bins[b + 1],
            "count": int(mask.sum()),
            "mean_confidence": conf,
            "fraction_positive": frac_pos,
            "abs_gap": gap,
        })
    return float(ece), pd.DataFrame(rows)


def safe_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return np.nan


def safe_ap(y_true, probs):
    try:
        return float(average_precision_score(y_true, probs))
    except ValueError:
        return np.nan


def evaluate_predictions(y_true, probs, threshold, positive_name, negative_name):
    pred = (probs >= threshold).astype(np.int64)

    precision, recall, f1s, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    ece, _ = expected_calibration_error(y_true, probs, ECE_BINS)
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)

    out = {
        "threshold": float(threshold),
        f"Precision_{negative_name}": float(precision[0]),
        f"Recall_{negative_name}": float(recall[0]),
        f"F1_{negative_name}": float(f1s[0]),
        f"Precision_{positive_name}": float(precision[1]),
        f"Recall_{positive_name}": float(recall[1]),
        f"F1_{positive_name}": float(f1s[1]),
        f"F2_{positive_name}": float(fbeta_score(y_true, pred, beta=2, pos_label=1, zero_division=0)),
        "Macro_F1": float(np.mean(f1s)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "AUC": safe_auc(y_true, probs),
        "Average_Precision": safe_ap(y_true, probs),
        "Brier": float(brier_score_loss(y_true, probs)),
        "ECE": float(ece),
        "Log_Loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "Prob_Mean_All": float(np.mean(probs)),
        "Prob_SD_All": float(np.std(probs)),
        f"Prob_Mean_{negative_name}": float(np.mean(probs[y_true == 0])),
        f"Prob_SD_{negative_name}": float(np.std(probs[y_true == 0])),
        f"Prob_Mean_{positive_name}": float(np.mean(probs[y_true == 1])),
        f"Prob_SD_{positive_name}": float(np.std(probs[y_true == 1])),
        "Probability_Collapse": bool(np.std(probs) < PROB_COLLAPSE_STD),
        "Single_Class_Prediction": bool(np.unique(pred).size == 1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "N_Test": int(len(y_true)),
        "N_Test_Positive": int(np.sum(y_true == 1)),
        "N_Test_Negative": int(np.sum(y_true == 0)),
    }
    return out

# -----------------------------
# 12. REPEATED EXPERIMENT
# -----------------------------
def run_dataset_experiment(X, y, meta):
    dataset_name = meta["dataset"]
    positive_name = meta["positive_class"]
    negative_name = meta["negative_class"]

    print("\n" + "=" * 80)
    print("DATASET:", dataset_name)
    print(f"Positive/minority: {positive_name} (n={(y == 1).sum()})")
    print(f"Negative/majority: {negative_name} (n={(y == 0).sum()})")
    print("=" * 80)

    all_rows = []
    split_rows = []
    history_rows = []
    prediction_store = []

    for seed in SEEDS:
        train_idx, val_idx, test_idx = repeated_stratified_split(y, seed)

        split_rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "train_total": len(train_idx),
            "train_positive": int((y[train_idx] == 1).sum()),
            "train_negative": int((y[train_idx] == 0).sum()),
            "val_total": len(val_idx),
            "val_positive": int((y[val_idx] == 1).sum()),
            "val_negative": int((y[val_idx] == 0).sum()),
            "test_total": len(test_idx),
            "test_positive": int((y[test_idx] == 1).sum()),
            "test_negative": int((y[test_idx] == 0).sum()),
        })

        print(f"\n[{dataset_name}] seed={seed} | "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        for loss_name in ["Soft-F2", "Weighted BCE"]:
            t0 = time.time()
            model, train_info, history = train_model(
                X, y, train_idx, val_idx, loss_name, seed
            )

            # Validation-only threshold selection
            p_val, y_val = predict_proba(model, X, y, val_idx)
            selected_threshold = choose_validation_threshold(y_val, p_val)

            # Untouched test evaluation
            p_test, y_test = predict_proba(model, X, y, test_idx)
            metrics = evaluate_predictions(
                y_test, p_test, selected_threshold, positive_name, negative_name
            )

            row = {
                "dataset": dataset_name,
                "positive_class": positive_name,
                "negative_class": negative_name,
                "seed": seed,
                "loss": loss_name,
                **train_info,
                **metrics,
                "runtime_seconds": float(time.time() - t0),
            }
            all_rows.append(row)

            for epoch, tr_loss, va_loss in history:
                history_rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "loss": loss_name,
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                })

            prediction_store.append({
                "dataset": dataset_name,
                "seed": seed,
                "loss": loss_name,
                "threshold": selected_threshold,
                "y_test": y_test.copy(),
                "p_test": p_test.copy(),
            })

            print(
                f"  {loss_name:12s} | best_epoch={train_info['best_epoch']:2d} "
                f"| t_val={selected_threshold:.2f} "
                f"| Macro-F1={metrics['Macro_F1']:.4f} "
                f"| AUC={metrics['AUC']:.4f} "
                f"| Brier={metrics['Brier']:.4f} "
                f"| ECE={metrics['ECE']:.4f} "
                f"| collapse={metrics['Probability_Collapse']} "
                f"| one-class={metrics['Single_Class_Prediction']}"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    results_df = pd.DataFrame(all_rows)
    splits_df = pd.DataFrame(split_rows)
    histories_df = pd.DataFrame(history_rows)

    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    results_df.to_csv(OUTPUT_DIR / f"{safe_name}_all_repeated_runs.csv", index=False)
    splits_df.to_csv(OUTPUT_DIR / f"{safe_name}_split_audit.csv", index=False)
    histories_df.to_csv(OUTPUT_DIR / f"{safe_name}_training_histories.csv", index=False)

    return results_df, splits_df, histories_df, prediction_store

# -----------------------------
# 13. SUMMARY TABLES
# -----------------------------
def mean_sd_ci(series):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = sd / math.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        low = mean - tcrit * se
        high = mean + tcrit * se
    else:
        low = high = mean
    return mean, sd, low, high


def build_summary_tables(all_results, dataset_metas):
    # Dataset characteristics
    dataset_table = pd.DataFrame(dataset_metas)
    dataset_table.to_csv(OUTPUT_DIR / "Table_dataset_characteristics.csv", index=False)

    # Transparent model configuration.
    config_rows = []
    for meta in dataset_metas:
        for loss_name in ["Soft-F2", "Weighted BCE"]:
            config_rows.append({
                "dataset": meta["dataset"],
                "model": "Compact MLP",
                "input_features": IMG_SIZE * IMG_SIZE,
                "architecture": f"Linear({IMG_SIZE*IMG_SIZE},{HIDDEN_UNITS}) -> ReLU -> Linear({HIDDEN_UNITS},1)",
                "output": "single logit; sigmoid used only for probabilities",
                "optimizer": "Adam",
                "loss": loss_name,
                "loss_details": (
                    f"Soft-F_beta with beta={SOFT_F_BETA}; positive class={meta['positive_class']}"
                    if loss_name == "Soft-F2"
                    else "BCEWithLogitsLoss; pos_weight=n_negative/n_positive computed from training split only"
                ),
                "max_epochs": MAX_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "early_stopping": f"validation loss; patience={EARLY_STOPPING_PATIENCE}; min_delta={MIN_DELTA}",
                "test_fraction": TEST_SIZE,
                "validation_fraction_total": (1 - TEST_SIZE) * VAL_WITHIN_TRAIN,
                "threshold_selection": "validation set only; maximize positive-class F1; tie-break Macro-F1 then proximity to 0.5",
                "independent_repeats": len(SEEDS),
                "seeds": ",".join(map(str, SEEDS)),
            })
    pd.DataFrame(config_rows).to_csv(OUTPUT_DIR / "Table_model_configuration.csv", index=False)

    # Metrics to summarize. Dynamic class-specific F1 columns are added per dataset.
    summary_rows = []
    for dataset_name, ddf in all_results.groupby("dataset"):
        pos = ddf["positive_class"].iloc[0]
        neg = ddf["negative_class"].iloc[0]
        metrics = [
            f"Precision_{pos}", f"Recall_{pos}", f"F1_{pos}", f"F2_{pos}",
            f"Precision_{neg}", f"Recall_{neg}", f"F1_{neg}",
            "Macro_F1", "Accuracy", "Balanced_Accuracy", "MCC",
            "AUC", "Average_Precision", "Brier", "ECE", "Log_Loss",
            "Prob_SD_All", "threshold", "best_epoch", "pos_weight"
        ]
        for loss_name, g in ddf.groupby("loss"):
            for metric in metrics:
                mean, sd, lo, hi = mean_sd_ci(g[metric])
                summary_rows.append({
                    "dataset": dataset_name,
                    "loss": loss_name,
                    "metric": metric,
                    "n_runs": len(g),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "mean_plus_minus_sd": f"{mean:.4f} ± {sd:.4f}",
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "Table_repeated_run_summary_mean_sd_95CI.csv", index=False)

    # Collapse/stability frequencies
    stability = (
        all_results.groupby(["dataset", "loss"])
        .agg(
            n_runs=("seed", "count"),
            probability_collapse_count=("Probability_Collapse", "sum"),
            single_class_prediction_count=("Single_Class_Prediction", "sum"),
            probability_sd_mean=("Prob_SD_All", "mean"),
            probability_sd_sd=("Prob_SD_All", "std"),
            macro_f1_mean=("Macro_F1", "mean"),
            macro_f1_sd=("Macro_F1", "std"),
            auc_mean=("AUC", "mean"),
            auc_sd=("AUC", "std"),
            brier_mean=("Brier", "mean"),
            brier_sd=("Brier", "std"),
            ece_mean=("ECE", "mean"),
            ece_sd=("ECE", "std"),
        )
        .reset_index()
    )
    stability["probability_collapse_rate"] = stability["probability_collapse_count"] / stability["n_runs"]
    stability["single_class_prediction_rate"] = stability["single_class_prediction_count"] / stability["n_runs"]
    stability.to_csv(OUTPUT_DIR / "Table_stability_and_collapse_rates.csv", index=False)

    # Mean ± SD confusion-matrix counts across repeated test splits.
    confusion_rows = []
    for (dataset_name, loss_name), g in all_results.groupby(["dataset", "loss"]):
        pos = g["positive_class"].iloc[0]
        neg = g["negative_class"].iloc[0]
        for col, interpretation in [
            ("TN", f"true {neg} predicted {neg}"),
            ("FP", f"true {neg} predicted {pos}"),
            ("FN", f"true {pos} predicted {neg}"),
            ("TP", f"true {pos} predicted {pos}"),
        ]:
            mean, sd, lo, hi = mean_sd_ci(g[col])
            confusion_rows.append({
                "dataset": dataset_name,
                "loss": loss_name,
                "cell": col,
                "interpretation": interpretation,
                "mean_count": mean,
                "sd_count": sd,
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_plus_minus_sd": f"{mean:.2f} ± {sd:.2f}",
            })
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "Table_confusion_matrix_counts_mean_sd.csv", index=False
    )

    return summary_df, stability

# -----------------------------
# 14. PAIRED STATISTICAL COMPARISONS
# -----------------------------
def paired_bootstrap_ci(diff, seed=2026, n_boot=10000):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(diff)
    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        means[i] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_paired_tests(all_results):
    rows = []
    for dataset_name, ddf in all_results.groupby("dataset"):
        pos = ddf["positive_class"].iloc[0]
        neg = ddf["negative_class"].iloc[0]
        metrics = [
            f"F1_{pos}", f"F1_{neg}", "Macro_F1", "Balanced_Accuracy",
            "AUC", "Average_Precision", "Brier", "ECE", "Log_Loss", "Prob_SD_All"
        ]

        soft = ddf[ddf["loss"] == "Soft-F2"].set_index("seed")
        bce = ddf[ddf["loss"] == "Weighted BCE"].set_index("seed")
        common = sorted(set(soft.index).intersection(bce.index))

        for metric in metrics:
            a = soft.loc[common, metric].astype(float).to_numpy()
            b = bce.loc[common, metric].astype(float).to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]
            diff = b - a  # positive = Weighted BCE numerically higher

            if len(diff) == 0:
                continue

            # Wilcoxon can fail if all paired differences are exactly zero.
            try:
                wilcox = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
                w_stat, w_p = float(wilcox.statistic), float(wilcox.pvalue)
            except ValueError:
                w_stat, w_p = 0.0, 1.0

            try:
                ttest = stats.ttest_rel(b, a, nan_policy="omit")
                t_stat, t_p = float(ttest.statistic), float(ttest.pvalue)
            except Exception:
                t_stat, t_p = np.nan, np.nan

            ci_lo, ci_hi = paired_bootstrap_ci(diff)

            rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "n_pairs": len(diff),
                "mean_SoftF2": float(np.mean(a)),
                "mean_WeightedBCE": float(np.mean(b)),
                "mean_difference_BCE_minus_SoftF2": float(np.mean(diff)),
                "bootstrap95CI_diff_low": ci_lo,
                "bootstrap95CI_diff_high": ci_hi,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_p_two_sided": w_p,
                "paired_t_statistic": t_stat,
                "paired_t_p_two_sided": t_p,
            })

    tests_df = pd.DataFrame(rows)
    tests_df.to_csv(OUTPUT_DIR / "Table_paired_statistical_tests.csv", index=False)
    return tests_df

# -----------------------------
# 15. REPRESENTATIVE-SEED TABLES
# -----------------------------
def choose_representative_seed(results_df):
    """Choose a seed whose mean Macro-F1 across the two losses is closest to median.
    Used ONLY for illustrative threshold/probability tables and figures; main inference
    remains based on all repeated runs.
    """
    tmp = results_df.groupby("seed")["Macro_F1"].mean()
    med = tmp.median()
    return int((tmp - med).abs().idxmin())


def threshold_sensitivity_rows(y_true, probs, positive_name, negative_name, loss_name, seed):
    rows = []
    for t in SENSITIVITY_THRESHOLDS:
        m = evaluate_predictions(y_true, probs, t, positive_name, negative_name)
        rows.append({
            "seed": seed,
            "loss": loss_name,
            "threshold": t,
            f"Precision_{positive_name}": m[f"Precision_{positive_name}"],
            f"Recall_{positive_name}": m[f"Recall_{positive_name}"],
            f"F1_{positive_name}": m[f"F1_{positive_name}"],
            f"F1_{negative_name}": m[f"F1_{negative_name}"],
            "Macro_F1": m["Macro_F1"],
            "Accuracy": m["Accuracy"],
        })
    return rows


def probability_stats_rows(y_true, probs, positive_name, negative_name, loss_name, seed):
    rows = []
    for label, cname in [(0, negative_name), (1, positive_name)]:
        p = probs[y_true == label]
        rows.append({
            "seed": seed,
            "loss": loss_name,
            "class": cname,
            "study_label": label,
            "N": len(p),
            "mean_prob_positive_class": float(np.mean(p)),
            "median_prob_positive_class": float(np.median(p)),
            "std_prob_positive_class": float(np.std(p)),
            "min_prob_positive_class": float(np.min(p)),
            "max_prob_positive_class": float(np.max(p)),
        })
    return rows


def build_representative_tables(all_results, all_prediction_stores):
    threshold_rows = []
    prob_rows = []
    representative_records = []

    for dataset_name, ddf in all_results.groupby("dataset"):
        seed = choose_representative_seed(ddf)
        pos = ddf["positive_class"].iloc[0]
        neg = ddf["negative_class"].iloc[0]
        representative_records.append({"dataset": dataset_name, "representative_seed": seed})

        stores = [s for s in all_prediction_stores
                  if s["dataset"] == dataset_name and s["seed"] == seed]
        for s in stores:
            threshold_rows.extend(
                threshold_sensitivity_rows(
                    s["y_test"], s["p_test"], pos, neg, s["loss"], seed
                )
            )
            prob_rows.extend(
                probability_stats_rows(
                    s["y_test"], s["p_test"], pos, neg, s["loss"], seed
                )
            )

    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_DIR / "Table_threshold_sensitivity_representative_seed.csv", index=False
    )
    pd.DataFrame(prob_rows).to_csv(
        OUTPUT_DIR / "Table_probability_statistics_representative_seed.csv", index=False
    )
    pd.DataFrame(representative_records).to_csv(
        OUTPUT_DIR / "Representative_seeds.csv", index=False
    )

# -----------------------------
# 16. PLOTS ACROSS REPEATED RUNS
# -----------------------------
def plot_performance_boxplots(all_results):
    for dataset_name, ddf in all_results.groupby("dataset"):
        pos = ddf["positive_class"].iloc[0]
        neg = ddf["negative_class"].iloc[0]
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")

        for metric in [f"F1_{pos}", f"F1_{neg}", "Macro_F1", "AUC", "Brier", "ECE", "Prob_SD_All"]:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            groups = [
                ddf.loc[ddf["loss"] == "Soft-F2", metric].astype(float).to_numpy(),
                ddf.loc[ddf["loss"] == "Weighted BCE", metric].astype(float).to_numpy(),
            ]
            ax.boxplot(groups, labels=["Soft-F2", "Weighted BCE"], showmeans=True)
            ax.set_title(f"{dataset_name}: {metric} across repeated splits")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"Figure_{safe_name}_{metric}_boxplot.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_mean_roc_pr(all_results, prediction_stores):
    for dataset_name, ddf in all_results.groupby("dataset"):
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")
        prevalence = float(ddf["N_Test_Positive"].mean() / ddf["N_Test"].mean())

        # Mean ROC
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        grid = np.linspace(0, 1, 201)
        for loss_name in ["Soft-F2", "Weighted BCE"]:
            curves = []
            aucs = []
            relevant = [s for s in prediction_stores
                        if s["dataset"] == dataset_name and s["loss"] == loss_name]
            for s in relevant:
                fpr, tpr, _ = roc_curve(s["y_test"], s["p_test"])
                interp = np.interp(grid, fpr, tpr)
                interp[0] = 0.0
                curves.append(interp)
                aucs.append(roc_auc_score(s["y_test"], s["p_test"]))
            arr = np.vstack(curves)
            mean_curve = arr.mean(axis=0)
            sd_curve = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean_curve)
            line, = ax.plot(grid, mean_curve, linewidth=2,
                            label=f"{loss_name} (mean AUC={np.mean(aucs):.3f})")
            ax.fill_between(
                grid,
                np.clip(mean_curve - sd_curve, 0, 1),
                np.clip(mean_curve + sd_curve, 0, 1),
                alpha=0.15,
                color=line.get_color(),
            )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{dataset_name}: mean ROC across repeated splits")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"Figure_{safe_name}_mean_ROC.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Mean PR
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        recall_grid = np.linspace(0, 1, 201)
        for loss_name in ["Soft-F2", "Weighted BCE"]:
            curves = []
            aps = []
            relevant = [s for s in prediction_stores
                        if s["dataset"] == dataset_name and s["loss"] == loss_name]
            for s in relevant:
                precision, recall, _ = precision_recall_curve(s["y_test"], s["p_test"])
                order = np.argsort(recall)
                recall_sorted = recall[order]
                precision_sorted = precision[order]
                interp = np.interp(recall_grid, recall_sorted, precision_sorted)
                curves.append(interp)
                aps.append(average_precision_score(s["y_test"], s["p_test"]))
            arr = np.vstack(curves)
            mean_curve = arr.mean(axis=0)
            sd_curve = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean_curve)
            line, = ax.plot(recall_grid, mean_curve, linewidth=2,
                            label=f"{loss_name} (mean AP={np.mean(aps):.3f})")
            ax.fill_between(
                recall_grid,
                np.clip(mean_curve - sd_curve, 0, 1),
                np.clip(mean_curve + sd_curve, 0, 1),
                alpha=0.15,
                color=line.get_color(),
            )
        ax.axhline(prevalence, linestyle="--", linewidth=1, label=f"prevalence={prevalence:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{dataset_name}: mean PR across repeated splits")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"Figure_{safe_name}_mean_PR.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mean_calibration(all_results, prediction_stores):
    for dataset_name, ddf in all_results.groupby("dataset"):
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")
        fig, ax = plt.subplots(figsize=(6.2, 5.2))

        for loss_name in ["Soft-F2", "Weighted BCE"]:
            bin_confs = []
            bin_fracs = []
            relevant = [s for s in prediction_stores
                        if s["dataset"] == dataset_name and s["loss"] == loss_name]
            for s in relevant:
                _, tab = expected_calibration_error(s["y_test"], s["p_test"], ECE_BINS)
                bin_confs.append(tab["mean_confidence"].to_numpy(dtype=float))
                bin_fracs.append(tab["fraction_positive"].to_numpy(dtype=float))

            conf_arr = np.vstack(bin_confs)
            frac_arr = np.vstack(bin_fracs)
            mean_conf = np.nanmean(conf_arr, axis=0)
            mean_frac = np.nanmean(frac_arr, axis=0)
            valid = np.isfinite(mean_conf) & np.isfinite(mean_frac)
            ax.plot(mean_conf[valid], mean_frac[valid], marker="o", linewidth=2,
                    label=f"{loss_name} (mean ECE={ddf.loc[ddf['loss']==loss_name, 'ECE'].mean():.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability of positive class")
        ax.set_ylabel("Observed positive-class frequency")
        ax.set_title(f"{dataset_name}: reliability diagram across repeated splits")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"Figure_{safe_name}_calibration.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# -----------------------------
# 17. HUMAN-READABLE SUMMARY
# -----------------------------
def create_text_summary(all_results, dataset_metas):
    lines = []
    lines.append("LOSS-STABILITY ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Repeated stratified splits: {len(SEEDS)}")
    lines.append(f"Seeds: {SEEDS}")
    lines.append(f"Architecture: {IMG_SIZE*IMG_SIZE} -> {HIDDEN_UNITS} ReLU -> 1 logit")
    lines.append(f"Max epochs: {MAX_EPOCHS}; early stopping patience: {EARLY_STOPPING_PATIENCE}")
    lines.append("Threshold selection: validation only; test never used for threshold choice")
    lines.append("")

    for meta in dataset_metas:
        d = all_results[all_results["dataset"] == meta["dataset"]]
        lines.append(f"DATASET: {meta['dataset']}")
        lines.append(
            f"Positive/minority={meta['positive_class']} n={meta['n_positive']}; "
            f"Negative/majority={meta['negative_class']} n={meta['n_negative']}; "
            f"ratio={meta['imbalance_ratio_negative_to_positive']:.3f}:1"
        )
        for loss_name in ["Soft-F2", "Weighted BCE"]:
            g = d[d["loss"] == loss_name]
            lines.append(
                f"  {loss_name}: Macro-F1={g['Macro_F1'].mean():.4f}±{g['Macro_F1'].std(ddof=1):.4f}; "
                f"AUC={g['AUC'].mean():.4f}±{g['AUC'].std(ddof=1):.4f}; "
                f"Brier={g['Brier'].mean():.4f}±{g['Brier'].std(ddof=1):.4f}; "
                f"ECE={g['ECE'].mean():.4f}±{g['ECE'].std(ddof=1):.4f}; "
                f"prob-collapse={int(g['Probability_Collapse'].sum())}/{len(g)}; "
                f"single-class={int(g['Single_Class_Prediction'].sum())}/{len(g)}"
            )
        lines.append("")

    lines.append("INTERPRETATION RULE FOR THE REVISED PAPER:")
    lines.append(
        "Do NOT claim Weighted BCE is calibrated merely because it has non-degenerate probabilities. "
        "Use the Brier/ECE results above. If calibration is not clearly better, replace 'calibrated' with "
        "'non-degenerate probability outputs' or 'greater probability dispersion'."
    )
    lines.append(
        "Do NOT claim general instability of F-measure losses. Frame conclusions as a controlled case study "
        "of this Soft-F2 formulation under severe imbalance and limited edge-based feature expressiveness."
    )

    with open(OUTPUT_DIR / "RUN_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))

# -----------------------------
# 18. MAIN
# -----------------------------
def main():
    # Clean only prior result files; keep expensive feature caches.
    for p in OUTPUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    all_result_dfs = []
    all_prediction_stores = []
    dataset_metas = []

    # -------- BeeSpotter --------
    X_bee, y_bee, meta_bee = build_beespotter_cache()
    dataset_metas.append(meta_bee)
    res_bee, splits_bee, hist_bee, pred_bee = run_dataset_experiment(X_bee, y_bee, meta_bee)
    all_result_dfs.append(res_bee)
    all_prediction_stores.extend(pred_bee)

    # Free BeeSpotter feature matrix before loading secondary dataset.
    del X_bee, y_bee
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------- Secondary public benchmark --------
    if RUN_SECONDARY_DATASET:
        X_sec, y_sec, meta_sec = build_cifar10_bird_frog_cache()
        dataset_metas.append(meta_sec)
        res_sec, splits_sec, hist_sec, pred_sec = run_dataset_experiment(X_sec, y_sec, meta_sec)
        all_result_dfs.append(res_sec)
        all_prediction_stores.extend(pred_sec)
        del X_sec, y_sec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine results
    all_results = pd.concat(all_result_dfs, ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "ALL_DATASETS_repeated_runs.csv", index=False)

    # Summaries / tests / manuscript-ready diagnostics
    summary_df, stability_df = build_summary_tables(all_results, dataset_metas)
    paired_tests_df = build_paired_tests(all_results)
    build_representative_tables(all_results, all_prediction_stores)

    # Figures
    plot_performance_boxplots(all_results)
    plot_mean_roc_pr(all_results, all_prediction_stores)
    plot_mean_calibration(all_results, all_prediction_stores)

    # Save configuration as JSON too
    config = {
        "N_REPEATS": N_REPEATS,
        "SEEDS": SEEDS,
        "TEST_SIZE": TEST_SIZE,
        "VAL_WITHIN_TRAIN": VAL_WITHIN_TRAIN,
        "IMG_SIZE": IMG_SIZE,
        "MEDIAN_KERNEL": MEDIAN_KERNEL,
        "HIDDEN_UNITS": HIDDEN_UNITS,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_EPOCHS": MAX_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "MIN_DELTA": MIN_DELTA,
        "SOFT_F_BETA": SOFT_F_BETA,
        "THRESHOLD_GRID_MIN": float(THRESHOLD_GRID.min()),
        "THRESHOLD_GRID_MAX": float(THRESHOLD_GRID.max()),
        "THRESHOLD_GRID_STEP": 0.01,
        "ECE_BINS": ECE_BINS,
        "PROB_COLLAPSE_STD": PROB_COLLAPSE_STD,
        "SECONDARY_DATASET": SECONDARY_DATASET_NAME if RUN_SECONDARY_DATASET else None,
        "SECONDARY_SUBSET_SEED": SECONDARY_SUBSET_SEED if RUN_SECONDARY_DATASET else None,
        "DEVICE": str(DEVICE),
    }
    with open(OUTPUT_DIR / "analysis_configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    create_text_summary(all_results, dataset_metas)

    # Bundle outputs
    zip_base = str(PROJECT_ROOT / "analysis_outputs_bundle")
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=OUTPUT_DIR)
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("All tables/figures:", OUTPUT_DIR)
    print("ZIP bundle:", zip_path)
    print("=" * 80)

    if AUTO_DOWNLOAD_RESULTS:
        try:
            from google.colab import files
            files.download(zip_path)
        except Exception as e:
            print("Automatic download could not start:", e)

    return all_results, summary_df, stability_df, paired_tests_df


# Run everything
all_results, summary_table, stability_table, paired_tests = main()

# Display the most important outputs in Colab
print("\nMAIN REPEATED-RUN SUMMARY")
display(summary_table)
print("\nSTABILITY / COLLAPSE SUMMARY")
display(stability_table)
print("\nPAIRED STATISTICAL TESTS")
display(paired_tests)
