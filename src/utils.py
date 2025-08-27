import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False):
    """
    Plots a confusion matrix as a heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.show()
    return fig

def plot_roc_curve(y_true, y_scores):
    """
    Plots the ROC curve and returns the AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def plot_precision_recall(y_true, y_scores):
    """
    Plots precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.show()

def shap_summary_plot(model, X_sample):
    """
    Generates a SHAP summary plot for feature importance.
    Works for tree-based models (like XGBoost).
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
    return shap_values

def print_classification_report(y_true, y_pred):
    """
    Prints sklearn's classification report (precision, recall, f1).
    """
    print(classification_report(y_true, y_pred))
