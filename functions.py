import seaborn as sns
import pandas as pd
import numpy as np
from tabulate import tabulate
import itertools
import re
import scipy.stats as ss
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import confusion_matrix, auc, roc_curve


def overview_data(df, display_frst_value=False):
    """
    Overview data from a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        display_frst_value (bool): Whether to display the first value of each column.

    Returns:
        None. Prints the analysis results.
    """
    print(f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")

    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "NA Values"]
    missing_values["%"] = np.round((missing_values["NA Values"] / df.shape[0]) * 100, 1)

    data_types = df.dtypes.reset_index()
    data_types.columns = ["Column", "Data Type"]
    data_types.drop("Column", axis=1, inplace=True)

    unique_values = df.nunique().reset_index()
    unique_values.columns = ["Column", "Unique"]
    unique_values.drop("Column", axis=1, inplace=True)

    if display_frst_value:
        first_value = df.apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
        )
        first_value = first_value.reset_index()
        first_value.columns = ["Column", "First Value"]
        first_value["First Value"] = first_value["First Value"].astype(str)
        first_value["First Value"] = np.where(
            first_value["First Value"].str.len() > 10,
            first_value["First Value"].str[:10] + "...",
            first_value["First Value"],
        )
        first_value.drop("Column", axis=1, inplace=True)
    else:
        first_value = pd.DataFrame(columns=None)

    result_table = pd.concat(
        [missing_values, data_types, unique_values, first_value], axis=1
    )

    print(tabulate(result_table, headers="keys", tablefmt="psql"))

    return None


def remove_missing_features(df, threshold=0.5, feat_not_to_remove=[]):
    """
    Remove features with missing values based on the given threshold.

    Parameters:
    - df: pandas DataFrame
        Input DataFrame.
    - threshold: float, optional (default=0.5)
        Threshold for missing value percentage. Features with missing values
        percentage greater than this threshold will be removed.
    - feat_not_to_remove: list, optional (default=[])

    Returns:
    - df_cleaned: pandas DataFrame
        DataFrame with missing features removed.
    - removed_features: list
        List of features that were removed.
    """
    missing_percentages = df.isnull().mean()
    features_to_remove = missing_percentages[
        missing_percentages > threshold
    ].index.tolist()
    features_to_remove = [
        feat for feat in features_to_remove if feat not in feat_not_to_remove
    ]
    df_cleaned = df.drop(columns=features_to_remove)
    return df_cleaned, features_to_remove


def convert_types(df, print_info=False):
    original_memory = df.memory_usage().sum()

    for c in df:
        if ("SK_ID" in c) or ("sk_id" in c):
            df[c] = df[c].fillna(0).astype(np.int32)

        elif (df[c].dtype == "object") and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype("category")

        elif list(df[c].unique()) == [1, 0] or list(df[c].unique()) == [0, 1]:
            df[c] = df[c].astype("category")

        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)

        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)

    new_memory = df.memory_usage().sum()

    if print_info:
        print(f"Original Memory Usage: {round(original_memory / 1e9, 2)} gb.")
        print(f"New Memory Usage: {round(new_memory / 1e9, 2)} gb.")

    return df


def distribution_plot(
    df,
    col: str = None,
    hue: str = None,
    labels: tuple = None,
    palette: tuple = ("Set2", 2),
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(data=df, x=col, fill=True, palette=palette, ax=axs[0])
    axs[0].set_title(f"Overall {format_feature_name(col)} Distribution")
    sns.kdeplot(
        data=df,
        x=col,
        hue=hue,
        common_norm=False,
        fill=True,
        palette=palette,
        ax=axs[1],
    )
    axs[1].set_title(
        f"{format_feature_name(col)}-{format_feature_name(hue)} Distribution"
    )
    if labels is not None:
        axs[1].legend(title="", labels=labels)
    else:
        axs[1].get_legend().set_title("")


def format_feature_name(name):
    words = name.split("_")
    formatted_name = " ".join(word.lower().capitalize() for word in words)
    return formatted_name


def normalized_barplots(
    df,
    cols,
    hue,
    grid_x,
    grid_y,
    palette: tuple = ("Set2", 2),
    figsize=(12, 8),
    legend_loc=(0.2, 1.5),
    **kwargs,
):
    num_plots = len(cols)
    fig, axs = plt.subplots(grid_y, grid_x, figsize=figsize, sharey=True)
    legend_created = False

    if grid_x > 1 and grid_y > 1:
        axs = axs.reshape(-1)

    for i, col in enumerate(cols):
        if grid_x > 1 or grid_y > 1:
            row = i // grid_x
            col = i % grid_x if grid_x > 1 else i // grid_y
        else:
            col = i
        df1 = df.groupby(cols[i])[hue].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename("percent").reset_index()
        sns.barplot(
            x=cols[i],
            y="percent",
            hue=hue,
            data=df1,
            ax=axs[i],
            palette=palette,
        )
        sns.despine()
        axs[i].set_title(
            f"The {hue} class distribution of the {format_feature_name(cols[i])}"
        )
        axs[i].set_xlabel(f"{format_feature_name(cols[i])}")
        axs[i].get_legend().set_visible(False)

        if not legend_created:
            axs[i].legend(bbox_to_anchor=legend_loc, loc="upper right", borderaxespad=0)
            legend_created = True

        for label in axs[i].containers:
            axs[i].bar_label(label, fmt="%.2f%%")

    for i in range(num_plots, grid_y * grid_x):
        if grid_x > 1 or grid_y > 1:
            row = i // grid_x
            col = i % grid_x if grid_x > 1 else i // grid_y
        else:
            col = i
        axs[i].axis("off")

    plt.tight_layout()


def corr_heatmap(df, columns=None, method="pearson", figsize=(10, 5), **kwargs):
    if columns is not None:
        df = df[columns]

    if method in ["pearson", "kendall", "spearman"]:
        corr = df.corr(method=method)
        vmin = -1
        vmax = 1
        center = 0
        cbar_kws = None
    elif method == "chi_squared":
        cat_var_combinations = list(itertools.combinations(columns, 2))
        result = {}
        for combination in cat_var_combinations:
            cross_tab = pd.crosstab(df[combination[0]], df[combination[1]])
            chi2, p, _, _ = ss.chi2_contingency(cross_tab)
            result[(combination[0], combination[1])] = p
        corr = pd.DataFrame(index=columns, columns=columns, dtype=float)
        np.fill_diagonal(corr.values, 1.0)
        for combination in cat_var_combinations:
            corr.at[combination[0], combination[1]] = result[
                (combination[0], combination[1])
            ]
            corr.at[combination[1], combination[0]] = result[
                (combination[0], combination[1])
            ]
        vmin = None
        vmax = None
        center = 0.05
        cbar_kws = {"ticks": [0.05]}
    else:
        raise ValueError(
            "Invalid method specified. Valid options are 'pearson', 'kendall', 'spearman', or 'chi_squared'."
        )
    plt.figure(figsize=figsize)
    plt.title(f"{format_feature_name(method)} Correlation Matrix")
    mask = np.triu(np.ones_like(corr))
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws=cbar_kws,
        **kwargs,
    )


def corr_cat_vs_cont(df, cat, cont):
    corr_result = pd.DataFrame(index=cont, columns=cat)
    pval_result = pd.DataFrame(index=cont, columns=cat)

    for c in cat:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

        for c2 in cont:
            df[c2] = df[c2].replace([np.inf, -np.inf], np.nan)

            valid_data_mask = ~df[c].isna() & ~df[c2].isna()

            valid_data = df[c][valid_data_mask], df[c2][valid_data_mask]

            if np.unique(valid_data[0]).size > 1 and np.unique(valid_data[1]).size > 1:
                corr, pval = pointbiserialr(*valid_data)
            else:
                corr, pval = np.nan, np.nan

            corr_result.loc[c2, c] = corr
            pval_result.loc[c2, c] = pval

    return corr_result, pval_result


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


def plot_roc_curve(
    fpr, tpr, fpr_2=None, tpr_2=None, label=None, label_2=None, figsize=(8, 6)
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve for one or two sets of data.

    Parameters:
    fpr (array-like): An array of false positive rates.
    tpr (array-like): An array of true positive rates.
    fpr_2 (array-like, optional): An array of false positive rates for a second dataset. Default is None.
    tpr_2 (array-like, optional): An array of true positive rates for a second dataset. Default is None.
    label (str, optional): Label for the main dataset's ROC curve. Default is None.
    label_2 (str, optional): Label for the second dataset's ROC curve. Default is None.
    figsize (tuple, optional): A tuple specifying the figure size. Default is (8, 6).

    Returns:
    None

    This function creates a ROC curve plot with optional support for a second dataset's curve.
    If a second dataset is provided, its ROC curve will be plotted with a dashed blue line.
    The diagonal line representing random guessing is also plotted as a black dashed line.
    The plot includes labels, title, and grid for better visualization.

    If label_2 is provided, a legend will be displayed in the lower right corner of the plot.

    Example:
    >>> fpr = [0.0, 0.1, 0.2, 0.3, 0.4]
    >>> tpr = [0.2, 0.4, 0.6, 0.8, 1.0]
    >>> plot_roc_curve(fpr, tpr, label="Dataset 1")
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    if fpr_2 is not None:
        plt.plot(fpr_2, tpr_2, "b:", label=label_2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (Fall-Out)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.grid(True)
    if label_2 is not None:
        plt.legend(loc="lower right")
    plt.show()


class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.

    Parameters
    ----------
    group_cols : list
        List of columns used for calculating the aggregated value
    target : str
        The name of the column to impute
    metric : str
        The metric to be used for replacement, can be one of ['mean', 'median']

    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    """

    def __init__(self, group_cols, target, metric="mean"):
        assert metric in [
            "mean",
            "median",
        ], "Unrecognized value for metric, should be mean/median"
        assert type(group_cols) == list, "group_cols should be a list of columns"
        assert type(target) == str, "target should be a string"

        self.group_cols = group_cols
        self.target = target
        self.metric = metric

    def fit(self, X, y=None):
        assert (
            pd.isnull(X[self.group_cols]).any(axis=None) == False
        ), "There are missing values in group_cols"

        impute_map = (
            X.groupby(self.group_cols)[self.target]
            .agg(self.metric)
            .reset_index(drop=False)
        )

        self.impute_map_ = impute_map

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "impute_map_")

        X = X.copy()

        for index, row in self.impute_map_.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind, self.target] = X.loc[ind, self.target].fillna(row[self.target])

        return X


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


class CustomWordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        self.output_format = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_and_lower(text):
            if isinstance(text, str):
                cleaned_text = re.sub(r"[\W_]+", " ", text)
                words = re.findall(r"\w+", cleaned_text)
                return " ".join(word.lower() for word in words)
            else:
                return text

        X = X.copy()
        X[self.column_name] = X[self.column_name].apply(extract_and_lower)
        return X

    def set_output(self, transform=None):
        return self


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
