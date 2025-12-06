import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

import torch

def get_probs_and_targets(model, data_loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    return all_probs, all_targets

def evaluate_with_threshold_search(model, test_loader, device):
    probs, targets = get_probs_and_targets(model, test_loader, device)

    thresholds = np.linspace(0.1, 0.9, 9)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=[0, 1], zero_division=0
        )
        f1_bombus = f1[0]
        f1_apis = f1[1]
        f1_macro = f1.mean()
        results.append((t, f1_apis, f1_bombus, f1_macro))

    print("\nThreshold search (on TEST set):")
    print("t\tF1_Apis\tF1_Bombus\tF1_macro")
    for t, fa, fb, fm in results:
        print(f"{t:.1f}\t{fa:.4f}\t{fb:.4f}\t{fm:.4f}")

    best = max(results, key=lambda x: x[1])
    best_t, best_f1_apis, best_f1_bombus, best_f1_macro = best
    print("\nBest threshold for Apis F1:", best_t)
    print(
        f"Best Apis F1: {best_f1_apis:.4f}, "
        f"Bombus F1: {best_f1_bombus:.4f}, Macro F1: {best_f1_macro:.4f}"
    )

    final_preds = (probs >= best_t).astype(int)
    print("\nClassification report at best threshold:")
    print(classification_report(targets, final_preds,
                                target_names=["Bombus", "Apis"],
                                digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(targets, final_preds))

def threshold_sensitivity_table(model, data_loader, device, name="model"):
    probs, targets = get_probs_and_targets(model, data_loader, device)
    thresholds = np.linspace(0.1, 0.9, 9)
    rows = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=[0,1], zero_division=0
        )
        prec_apis = precision[1]
        rec_apis  = recall[1]
        f1_bombus = f1[0]
        f1_apis   = f1[1]
        f1_macro  = f1.mean()

        rows.append({
            "model": name,
            "threshold": t,
            "F1_Apis": f1_apis,
            "F1_Bombus": f1_bombus,
            "Precision_Apis": prec_apis,
            "Recall_Apis": rec_apis,
            "Macro_F1": f1_macro
        })

    return pd.DataFrame(rows)

def probability_stats_table(model, data_loader, device, name="model"):
    probs, targets = get_probs_and_targets(model, data_loader, device)

    stats_rows = []
    for cls, cls_name in zip([0,1], ["Bombus", "Apis"]):
        cls_probs = probs[targets == cls]
        stats_rows.append({
            "model": name,
            "class": cls_name,
            "count": len(cls_probs),
            "mean_prob": float(np.mean(cls_probs)),
            "median_prob": float(np.median(cls_probs)),
            "std_prob": float(np.std(cls_probs)),
            "min_prob": float(np.min(cls_probs)),
            "max_prob": float(np.max(cls_probs)),
        })

    return pd.DataFrame(stats_rows)

def plot_roc(models_dict, loader, device):
    plt.figure()
    for name, model in models_dict.items():
        probs, targets = get_probs_and_targets(model, loader, device)
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_pr(models_dict, loader, device):
    plt.figure()
    for name, model in models_dict.items():
        probs, targets = get_probs_and_targets(model, loader, device)
        precision, recall, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves (Apis as positive)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()
