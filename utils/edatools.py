"""Helper module for EDA"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def format_feature_name(name):
    words = name.split("_")
    formatted_name = " ".join(word.lower().capitalize() for word in words)
    return formatted_name


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


def categorical_correlations(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conduct Chi-square test of independence and Cramér's V test for
    each pair of categorical features of the given data"""

    correlation_matrix = pd.DataFrame(index=columns, columns=columns)
    p_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                # Set diagonal values
                correlation_matrix.loc[col1, col2] = 1.0
                p_matrix.loc[col1, col2] = 0.0
            else:
                # Conduct the chi-square test
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                if np.min(expected) <= 5:
                    print(contingency_table)
                    print(expected)
                    print(
                        f"""One of the {col1} and {col2} combinations has
                          expected value less than 5."""
                    )

                # Calculate Cramér's V
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, k = contingency_table.shape
                phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                rcorr = r - ((r - 1) ** 2) / (n - 1)
                kcorr = k - ((k - 1) ** 2) / (n - 1)
                correlation_matrix.loc[col1, col2] = np.sqrt(
                    phi2corr / min((kcorr - 1), (rcorr - 1))
                )
                p_matrix.loc[col1, col2] = p

    correlation_matrix = correlation_matrix.astype("float")
    p_matrix = p_matrix.astype("float")
    return correlation_matrix, p_matrix


def categorical_numerical_correlation(
    df: pd.DataFrame, categorical_columns: list[str], numerical_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conduct Kruskal-Wallis test for each pair of categorical features of the given data"""

    effect_size_matrix = pd.DataFrame(
        index=categorical_columns, columns=numerical_columns
    )
    p_matrix = pd.DataFrame(index=categorical_columns, columns=numerical_columns)

    if df.empty:
        raise ValueError("Input DataFrame 'df' is empty.")
    for col in categorical_columns + numerical_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            grouped_data = []
            for category in df[cat_col].unique():
                values = df[df[cat_col] == category][num_col]
                grouped_data.append(values)

            max_length = max(len(sublist) for sublist in grouped_data)
            min_length = min(len(sublist) for sublist in grouped_data)
            # Checking if the sizes of subgroupds (num_cols) differ too much from each other.
            if max_length / min_length > 100 or min_length < 50:
                print("The size of the largest and smaller group differs too much.")
                print(f"{cat_col} vs {num_col}")
                print(df[cat_col].unique())
                print([len(sublist) for sublist in grouped_data])

            # Perform the Kruskal-Wallis test
            try:
                statistic, p = kruskal(*grouped_data)
                p_matrix.loc[cat_col, num_col] = p
            except ValueError as e:
                print(f"An error occurred: {e}")
                print(f"Categorical: {cat_col}")
                print(f"Numerical: {num_col}")
                p = np.nan
                p_matrix.loc[cat_col, num_col] = p

            if p < 0.05:
                # Calculate Epsilon-squared (Effect size)
                # Effect size range between 0 to 1.
                n_total = sum(len(sublist) for sublist in grouped_data)
                k = len(grouped_data)
                epsilon_squared = (statistic - (k - 1)) / (n_total - k)
                effect_size_matrix.loc[cat_col, num_col] = epsilon_squared
            else:
                effect_size_matrix.loc[cat_col, num_col] = np.nan

    effect_size_matrix = effect_size_matrix.astype("float")
    p_matrix = p_matrix.astype("float")
    return effect_size_matrix, p_matrix


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
            chi2, p, _, _ = chi2_contingency(cross_tab)
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
