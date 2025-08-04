import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.calibration import calibration_curve

def threshold_tuning(y_true, y_probs, plot=True):
    """
    Tune classification threshold to improve F1-score.
    
    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities for class 1.
        plot: Whether to plot PR, ROC, and calibration curves.

    Returns:
        best_threshold: Threshold giving highest F1-score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n Best Threshold based on F1-score: {best_threshold:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

    # Classification at best threshold
    custom_preds = (y_probs > best_threshold).astype(int)
    print("\n Classification Report (Custom Threshold):")
    print(classification_report(y_true, custom_preds))
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_true, custom_preds))

    if plot:
        _plot_pr_threshold(thresholds, precision, recall, best_threshold)
        _plot_pr_curve(precision, recall, best_idx)
        _plot_roc_curve(y_true, y_probs)
        _plot_calibration(y_true, y_probs)
        _plot_proba_histogram(y_probs)

    return best_threshold

def _plot_pr_threshold(thresholds, precision, recall, best_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _plot_pr_curve(precision, recall, best_idx):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label='Best Threshold Point')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _plot_calibration(y_true, y_probs):
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _plot_proba_histogram(y_probs):
    plt.figure(figsize=(8, 6))
    plt.hist(y_probs, bins=50, edgecolor='black')
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Probability of Class 1")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
