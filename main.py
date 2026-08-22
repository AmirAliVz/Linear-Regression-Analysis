"""
main.py

Predicts housing prices using linear regression on a dataset of property features.

Covers:
- Descriptive statistics and univariate / bivariate visualizations
- Box-Cox transformation for skewed variables before correlation analysis
- Feature selection via forward stepwise, backward elimination, and RFE
- Model training, evaluation, and assumption verification on training and test sets

Author: Amir Vaziri
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import boxcox, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.eval_measures import mse

# ── Module-level constants ─────────────────────────────────────────────────────
FIGURES_DIR    = "Figures"
DATA_PATH      = "data/Housing_Information.csv"
RANDOM_SEED    = 42
THRESHOLD_IN   = 0.05   # p-value threshold for forward stepwise selection
THRESHOLD_OUT  = 0.05   # p-value threshold for backward elimination


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all boolean-typed columns in a DataFrame to integer (0 / 1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain boolean columns after one-hot encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean columns cast to int.
    """
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate a DataFrame into feature matrix and target vector,
    one-hot-encode categorical columns, and convert booleans to int.

    Parameters
    ----------
    df : pd.DataFrame
        Input split (train or test).
    target : str
        Name of the dependent variable column.

    Returns
    -------
    X : pd.DataFrame
        Encoded and typed feature matrix.
    y : pd.Series
        Target variable.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    X = encode_bool_columns(X)
    return X, y


# ── Visualisation functions ────────────────────────────────────────────────────

def plot_histogram_boxplot(
    data, filename, kde=True, display=True,
    figsize=(10, 6), fontsize=12, fontcolor="black",
):
    """
    Plot a combined box plot and histogram for a given data Series.

    Parameters
    ----------
    data : pd.Series
        Input column of data.
    filename : str
        Column label used as the plot title and saved filename.
    kde : bool, optional
        Overlay a kernel density estimate on the histogram. Default True.
    display : bool, optional
        Render and save the figure. Default True.
    figsize : tuple, optional
        Figure dimensions in inches. Default (10, 6).
    fontsize : int, optional
        Font size for box-plot annotations. Default 12.
    fontcolor : str, optional
        Font colour for box-plot annotations. Default 'black'.

    Returns
    -------
    lower_whisker : float
        Lower IQR bound (Q1 - 1.5 * IQR, floored at the data minimum).
    upper_whisker : float
        Upper IQR bound (Q3 + 1.5 * IQR, capped at the data maximum).
    """
    q1            = np.percentile(data, 25)
    median        = np.median(data)
    q3            = np.percentile(data, 75)
    iqr           = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        sharex=True,
    )

    cmap = plt.get_cmap("inferno")

    # ── Box plot ──────────────────────────────────────────────────────────────
    sns.boxplot(x=data, ax=ax_box, color="skyblue")

    box_stats = {
        "Min":    lower_whisker,
        "Q1":     q1,
        "Median": median,
        "Q3":     q3,
        "Max":    upper_whisker,
    }
    for label, val in box_stats.items():
        ax_box.text(
            val, 0.02, f"{label}\n{val:.2f}",
            ha="center", va="bottom",
            fontsize=fontsize, color=fontcolor, rotation=45, weight="bold",
        )
    ax_box.set(xlabel="")

    # ── Histogram ─────────────────────────────────────────────────────────────
    hist_plot = sns.histplot(
        data, bins="auto", kde=kde,
        color="steelblue", edgecolor="black", ax=ax_hist,
    )

    # Summary statistics annotation overlaid on the histogram
    stats_text = "\n".join([
        f"Mean: {data.mean():.2f}",
        f"Mode: {data.mode().values[0]:.2f}",
        f"Std:  {data.std():.2f}",
        f"Skew: {data.skew():.2f}",
    ])
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    ax_hist.text(
        0.85, 0.6, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top", horizontalalignment="right",
        bbox=bbox_props, fontsize=fontsize + 2, color=fontcolor, weight="bold",
    )

    patches = hist_plot.patches
    counts  = [patch.get_height() for patch in patches]

    # Apply inferno colour gradient scaled to bar height
    for patch, count in zip(patches, counts):
        patch.set_facecolor(cmap(0.3 + 0.7 * count / max(counts)))

    # Annotate each bar with its count
    for patch, count in zip(patches, counts):
        if count > 0:
            ax_hist.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center", va="bottom", fontsize=fontsize,
            )

    ax_hist.set_title(f"Distribution of {filename}", fontweight="bold")
    ax_hist.set_xlabel(filename, fontweight="bold")
    ax_hist.set_ylabel("Count", fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.jpg"), dpi=300)
    plt.show()

    return lower_whisker, upper_whisker


def barchart(data, filename, fontsize=14, verbose=True):
    """
    Generate a frequency bar chart from a categorical Series and save the figure.

    Parameters
    ----------
    data : pd.Series
        Categorical column to plot.
    filename : str
        Output filename (without path or extension).
    fontsize : int, optional
        Font size for axis labels and bar annotations. Default 14.
    verbose : bool, optional
        Display the figure interactively. Default True.
    """
    counts = data.value_counts()

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("pastel")[: len(counts)]
    bars   = plt.bar(counts.index, counts, color=colors, edgecolor="black", linewidth=1.5)

    # Annotate each bar with its absolute count and percentage of total
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            1.03 * bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center", fontsize=fontsize, fontweight="bold", color="black",
        )
        pct = round(bar.get_height() / len(data) * 100, 1)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            0.5 * bar.get_height(),
            f"{float(pct)}%",
            ha="center", fontsize=fontsize, fontweight="bold", color="black",
        )

    plt.title(f"Distribution of {filename}", fontsize=fontsize + 4, fontweight="bold", color="#333333")
    plt.xlabel(filename, fontsize=fontsize + 2, fontweight="bold", color="#555555")
    plt.ylabel("Frequency Count", fontsize=fontsize + 2, fontweight="bold", color="#555555")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.box(False)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.jpg"), dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()


def catNum(df, numeric_col, categorical_col, verbose=True):
    """
    Create a paired violin plot and histogram + KDE overlay for a continuous
    variable broken down by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both columns.
    numeric_col : str
        Continuous variable to plot on the y-axis / x-axis.
    categorical_col : str
        Categorical grouping variable.
    verbose : bool, optional
        Display the figure interactively. Default True.
    """
    plt.figure(figsize=(14, 6))

    # Violin plot showing distribution shape and quartile positions per category
    plt.subplot(1, 2, 1)
    sns.violinplot(
        x=categorical_col, y=numeric_col, hue=categorical_col,
        data=df, inner="quartile",
        palette="Set2", dodge=False, legend=False,
    )
    plt.title(
        f"{numeric_col} Distribution by {categorical_col} (Violin Plot)",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel(categorical_col, fontsize=14, fontweight="bold")
    plt.ylabel(numeric_col, fontsize=14, fontweight="bold")

    # Overlaid histogram and KDE for each category, normalised to density
    plt.subplot(1, 2, 2)
    sns.histplot(
        data=df, x=numeric_col, hue=categorical_col,
        kde=True, element="step", stat="density",
        common_norm=False, palette="Set1",
    )
    plt.title(
        f"{numeric_col} Distribution by {categorical_col} (Histogram + KDE)",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel(numeric_col, fontsize=14, fontweight="bold")
    plt.ylabel("Density", fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{numeric_col}by{categorical_col}.jpg"), dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()


def scatter_pearson(x, y, c1, c2, verbose=True):
    """
    Create a scatter plot and compute the Pearson correlation between two
    continuous variables, applying Box-Cox transformation to either variable
    whose absolute skewness is ≥ 1 before computing the coefficient.

    Parameters
    ----------
    x : array-like or pd.Series
        Values for the x-axis.
    y : array-like or pd.Series
        Values for the y-axis.
    c1 : str
        Label for x (used in the plot title and filename).
    c2 : str
        Label for y.
    verbose : bool, optional
        Print the correlation coefficient and display the figure. Default True.

    Returns
    -------
    pearson_corr : float
        Pearson correlation coefficient between (possibly transformed) x and y.
    """
    c1 = c1.capitalize()
    c2 = c2.capitalize()

    # Apply Box-Cox to each variable independently if its skewness exceeds the threshold
    if abs(skew(x)) >= 1:
        x, _lambda_x = boxcox(x + 1)
    if abs(skew(y)) >= 1:
        y, _lambda_y = boxcox(y + 1)

    correlation_matrix = np.corrcoef(x, y)
    pearson_corr       = correlation_matrix[0, 1]

    if verbose:
        print(f"Pearson correlation  {c1} vs {c2}: {pearson_corr:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.5})

    # Display the correlation coefficient as an annotated text box on the plot
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    plt.text(
        0.9, 0.8, f"r: {pearson_corr:.2f}",
        transform=ax.transAxes,
        verticalalignment="top", horizontalalignment="right",
        bbox=bbox_props, fontsize=16, color="black", weight="bold",
    )

    plt.title(f"{c1} vs {c2}", fontsize=16, fontweight="bold")
    plt.xlabel(c1, weight="bold")
    plt.ylabel(c2, weight="bold")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{c1}vs{c2}.jpg"), dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return pearson_corr


# ── Feature selection ──────────────────────────────────────────────────────────

def forward_stepwise_selection(X, y, threshold_in=THRESHOLD_IN, verbose=True):
    """
    Forward stepwise feature selection for OLS linear regression.

    Starts with an empty model and iteratively adds the predictor with the
    lowest p-value below threshold_in, stopping when no remaining variable
    meets the threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Candidate predictor variables (pre-processed: dummies encoded, scaled).
    y : pd.Series
        Target variable.
    threshold_in : float, optional
        p-value threshold for adding a variable. Default 0.05.
    verbose : bool, optional
        Print each addition step. Default True.

    Returns
    -------
    list of str
        Names of the selected features in the order they were added.
    """
    included = []
    while True:
        changed  = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]

        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"Add  {best_feature:30}  p-value: {best_pval:.6f}")

        if not changed:
            break

    return included


def backward_stepwise_elimination(X, y, threshold_out=THRESHOLD_OUT, verbose=True):
    """
    Backward stepwise elimination for OLS linear regression.

    Starts with the full model and iteratively removes the predictor with the
    highest p-value above threshold_out, stopping when all remaining variables
    are significant.

    Parameters
    ----------
    X : pd.DataFrame
        Full set of candidate predictors (pre-processed).
    y : pd.Series
        Target variable.
    threshold_out : float, optional
        p-value threshold for removing a variable. Default 0.05.
    verbose : bool, optional
        Print each removal step. Default True.

    Returns
    -------
    list of str
        Names of the retained features.
    """
    features = list(X.columns)
    while True:
        model    = sm.OLS(y, sm.add_constant(X[features])).fit()
        pvalues  = model.pvalues.iloc[1:]   # exclude the intercept
        max_pval = pvalues.max()
        if max_pval > threshold_out:
            excluded_feature = pvalues.idxmax()
            features.remove(excluded_feature)
            if verbose:
                print(f"Remove  {excluded_feature:30}  p-value: {max_pval:.6f}")
        else:
            break

    return features


def rfe_statsmodels(X, y, n_features_to_select=5, verbose=True):
    """
    Recursive Feature Elimination using statsmodels OLS.

    Iteratively removes the least statistically significant predictor (highest
    p-value) until the specified number of features remains.

    Parameters
    ----------
    X : pd.DataFrame
        Candidate predictor variables (pre-processed).
    y : pd.Series
        Target variable.
    n_features_to_select : int, optional
        Number of features to retain. Default 5.
    verbose : bool, optional
        Print each removal step. Default True.

    Returns
    -------
    list of str
        Names of the retained features.
    """
    features = list(X.columns)
    while len(features) > n_features_to_select:
        model         = sm.OLS(y, sm.add_constant(X[features])).fit()
        pvalues       = model.pvalues.iloc[1:]
        worst_feature = pvalues.idxmax()
        if verbose:
            print(f"Remove  {worst_feature}  p-value: {pvalues[worst_feature]:.4f}")
        features.remove(worst_feature)

    return features


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def plot_residuals_analysis(model):
    """
    Plot residuals vs. fitted values and the autocorrelation function of residuals.

    Residuals vs. fitted values checks the linearity and homoscedasticity
    assumptions. The ACF plot checks the independence-of-errors assumption.

    Parameters
    ----------
    model : statsmodels RegressionResultsWrapper
        A fitted OLS model.
    """
    fitted_vals = model.fittedvalues
    residuals   = model.resid

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs. fitted values — should show no systematic pattern
    ax[0].scatter(fitted_vals, residuals, alpha=0.5)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].set_xlabel("Fitted values")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals vs Fitted Values", weight="bold")

    # Autocorrelation of residuals — values within the confidence band indicate independence
    sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05, zero=False, ax=ax[1])
    ax[1].set_title("Autocorrelation of Residuals", weight="bold")
    ax[1].set_xlabel("Lag")
    ax[1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_analysis.jpg"), dpi=300)
    plt.show()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(verbose=True):
    """
    Orchestrate the full linear regression analysis pipeline.

    Steps
    -----
    1. Load and prepare the housing dataset.
    2. Descriptive statistics and univariate / bivariate visualizations.
    3. Train / test split (80 / 20).
    4. Feature encoding and MinMax scaling.
    5. Forward stepwise feature selection on the scaled training set.
    6. OLS model training on selected features.
    7. MSE evaluation on training and test sets.
    8. Residual diagnostic plots.

    Parameters
    ----------
    verbose : bool, optional
        Print progress and display figures. Default True.
    """
    plt.rcParams["font.family"] = "Georgia"
    plt.rcParams["font.size"]   = 14

    # ── Load and prepare data ─────────────────────────────────────────────────
    df       = pd.read_csv(DATA_PATH)
    df_clean = df.copy()
    df_clean = df_clean.drop(["ID", "PreviousSalePrice"], axis=1)

    # Windows contains negative entries that are physically implausible — take absolute value
    df_clean["Windows"] = df_clean["Windows"].abs()

    # ── Part C: Descriptive statistics and visualizations ─────────────────────
    numeric = df_clean.select_dtypes(include=np.number)

    for col in numeric:
        if col not in ["NumBedrooms", "Floors", "IsLuxury"]:
            plot_histogram_boxplot(df_clean[col], col, display=verbose)
            if col != "Price":
                scatter_pearson(df_clean["Price"], df_clean[col], "Price", col, verbose=verbose)

    for col in ["Fireplace", "Garage", "IsLuxury", "HouseColor", "NumBedrooms", "Floors"]:
        barchart(df_clean[col], col, verbose=verbose)
        catNum(df_clean, "Price", col, verbose=verbose)

    # ── Part D: Model training and evaluation ─────────────────────────────────

    # 80 / 20 stratified split with a fixed seed for reproducibility
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=RANDOM_SEED)
    train_df.to_csv("data/train_dataset.csv", index=False)
    test_df.to_csv("data/test_dataset.csv",  index=False)

    # Encode categoricals and cast booleans for both splits
    X_train, y_train = prepare_features(train_df, "Price")
    X_test,  y_test  = prepare_features(test_df,  "Price")

    # Fit the scaler on training data only — transform is applied separately to
    # the test set to prevent leakage of test-set statistics into the model
    scaler         = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),           # transform only — scaler already fitted
        columns=X_test.columns, index=X_test.index,
    )

    # Feature selection runs on the scaled training set for consistency with
    # the final model, which also trains on the scaled data
    selected_features = forward_stepwise_selection(X_train_scaled, y_train, verbose=verbose)
    # Alternatives (all three converge on the same feature set):
    # selected_features = backward_stepwise_elimination(X_train_scaled, y_train, verbose=verbose)
    # selected_features = rfe_statsmodels(X_train_scaled, y_train, n_features_to_select=11, verbose=verbose)

    # Train OLS on the selected and scaled features
    X_train_final = sm.add_constant(X_train_scaled[selected_features])
    model = sm.OLS(y_train, X_train_final).fit()

    if verbose:
        print(model.summary())

    # MSE on the training set
    y_train_pred    = model.predict(X_train_final)
    mse_train       = mse(y_train, y_train_pred)
    print(f"MSE — training set: {mse_train:,.0f}")

    # MSE on the held-out test set
    X_test_final = sm.add_constant(X_test_scaled[selected_features])
    y_test_pred  = model.predict(X_test_final)
    mse_test     = mse(y_test, y_test_pred)
    print(f"MSE — test set:     {mse_test:,.0f}")

    # Residual diagnostic plots (linearity, homoscedasticity, autocorrelation)
    plot_residuals_analysis(model)


if __name__ == "__main__":
    main(verbose=True)