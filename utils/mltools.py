"""Helper module for machine learning tasks"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_statistics(cm):
    """
    Calculate and return summary statistics such as accuracy, precision, recall, and F1-score.

    Parameters:
        cm (numpy.ndarray): The confusion matrix.

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    statistics = {}
    accuracy = np.trace(cm) / float(np.sum(cm))
    statistics["accuracy"] = accuracy

    if len(cm) == 2:
        precision = cm[1, 1] / sum(cm[:, 1])
        recall = cm[1, 1] / sum(cm[1, :])
        f1_score = 2 * precision * recall / (precision + recall)

        statistics["precision"] = precision
        statistics["recall"] = recall
        statistics["f1_score"] = f1_score

    return statistics


def plot_confusion_matrix(
    cm,
    categories="auto",
    cbar=False,
    sum_stats=True,
    cmap="Blues",
    title="Confusion Matrix",
    figsize=None,
    ax=None,
    display_accuracy=True,
    display_precision=True,
    display_recall=True,
    display_f1_score=True,
) -> None:
    """
    Plot a heatmap representation of the confusion matrix along with optional summary statistics.

    Parameters:
        cm (numpy.ndarray): The confusion matrix to be visualized. It should be a 2D array-like object.
        categories ({"auto", list}, optional): The categories to be displayed on the x and y-axis.
            If "auto", the categories will be inferred from `cm`. If a list is provided,
            it should match the length of `cm`. Default is "auto".
        count (bool, optional): If True, display the count of occurrences in each cell of the matrix. Default is True.
        percent (bool, optional): If True, display the percentage of occurrences in each cell of the matrix. Default is True.
        cbar (bool, optional): If True, display a colorbar alongside the heatmap. Default is False.
        sum_stats (bool, optional): If True, calculate and display summary statistics such as accuracy,
            precision, recall, and F1-score (applicable for binary confusion matrices). Default is True.
        cmap (str, optional): The color map to be used for the heatmap. Default is "Blues".
        title (str, optional): The title of the plot. Default is "Confusion Matrix".
        figsize (tuple, optional): The size of the figure. If not provided, the default size will be used.
        ax (matplotlib.axes.Axes, optional): The matplotlib Axes on which to plot the heatmap. If not provided, a new
            figure and axis will be created.
        display_accuracy (bool, optional): If True, display accuracy in the title. Default is True.
        display_precision (bool, optional): If True, display precision in the title. Default is True.
        display_recall (bool, optional): If True, display recall in the title. Default is True.
        display_f1_score (bool, optional): If True, display F1 score in the title. Default is True.

    Returns:
        None: This function only displays the heatmap plot and optional summary statistics.
    """
    if categories == "auto":
        categories = [str(i) for i in range(len(cm))]

    labels = [
        [
            f"Count: {cm[i, j]:0.0f}\n"
            f"Percent: {cm[i, j] / max(1, np.sum(cm[i, :])):.2%}"
            for j in range(cm.shape[1])
        ]
        for i in range(cm.shape[0])
    ]
    labels = np.array(labels)

    if sum_stats:
        statistics = calculate_statistics(cm)
        stats_text = "\n\n"
        if display_accuracy:
            stats_text += f"Accuracy={statistics['accuracy']:.3f}\n"
        if display_precision and "precision" in statistics:
            stats_text += f"Precision={statistics['precision']:.3f}\n"
        if display_recall and "recall" in statistics:
            stats_text += f"Recall={statistics['recall']:.3f}\n"
        if display_f1_score and "f1_score" in statistics:
            stats_text += f"F1 Score={statistics['f1_score']:.3f}"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        ax=ax,
        square=True,
        xticklabels=categories,
        yticklabels=categories,
    )
    if sum_stats is True:
        ax.set(
            title=title + stats_text,
            xlabel="Predicted label",
            ylabel="Actual label",
        )
    else:
        ax.set(title=title, xlabel="Predicted label", ylabel="Actual label")

    if ax is None:
        plt.show()


def binary_performances(
    y_true, y_prob, thresh=0.5, labels=["Positives", "Negatives"], categories=[0, 1]
):
    shape = y_prob.shape
    if len(shape) > 1:
        if shape[1] > 2:
            raise ValueError("A binary class problem is required")
        else:
            y_prob = y_prob[:, 1]

    plt.figure(figsize=[15, 4])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y_true, (y_prob > thresh).astype(int))

    ax = plt.subplot(131)
    plot_confusion_matrix(cm, categories=categories, sum_stats=False, ax=ax)

    # 2 -- Distributions of Predicted Probabilities of both classes
    plt.subplot(132)
    plt.hist(
        y_prob[y_true == 1],
        density=True,
        bins=25,
        alpha=0.5,
        color="green",
        label=labels[0],
    )
    plt.hist(
        y_prob[y_true == 0],
        density=True,
        bins=25,
        alpha=0.5,
        color="red",
        label=labels[1],
    )
    plt.axvline(thresh, color="blue", linestyle="--", label="Boundary")
    plt.xlim([0, 1])
    plt.title("Distributions of Predictions", size=13)
    plt.xlabel("Positive Probability (predicted)", size=10)
    plt.ylabel("Samples (normalized scale)", size=10)
    plt.legend(loc="upper right")

    # 3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(133)
    plt.plot(
        fp_rates,
        tp_rates,
        color="orange",
        lw=1,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp / (fp + tn), tp / (tp + fn), "bo", markersize=8, label="Decision Point")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", size=10)
    plt.ylabel("True Positive Rate", size=10)
    plt.title("ROC Curve", size=13)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=0.3)
    plt.show()

    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    results = {"Precision": precision, "Recall": recall, "F1 Score": F1, "AUC": roc_auc}

    prints = [f"{kpi}: {round(score, 3)}" for kpi, score in results.items()]
    prints = " | ".join(prints)
    print(prints)


class CustomCategoricalConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_convert=None):
        self.columns_to_convert = columns_to_convert

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_convert is not None:
            for col in self.columns_to_convert:
                X[col] = X[col].astype("category")
        return X


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, order):
        self.column_name = column_name
        self.order = order

    def fit(self, X, y=None):
        unique_values = X[self.column_name].unique()
        self.mapping = {
            val: i for i, val in enumerate(self.order) if val in unique_values
        }
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column_name] = X[self.column_name].map(self.mapping)
        X[self.column_name] = pd.Categorical(
            X[self.column_name], categories=range(len(self.order)), ordered=True
        )
        return X

    def set_output(self, transform=None):
        return self
