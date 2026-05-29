"""
Task 1: Linear Regression for Housing Price Prediction

This code is for Task 1: Linear Regression. It applies linear regression to predict housing prices based on our dataset of house features.
In addition to building and evaluating the regression model, this code also visualizes all relevant univariate and bivariate analyses for the variables involved.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, mode
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse
from sklearn.preprocessing import MinMaxScaler


def plot_histogram_boxplot(data, filename, kde=True, display=True, figsize=(10, 6), fontsize=12, fontcolor='black'):

    """
    Plots a combined figure of a box plot and a histogram for a specified column of data in a Series.

    Parameters:
        data (pd.Series): The input column of data. Must be provided.
        filename (str): The name of the column to visualize. Must be provided.
        kde (bool, optional): Whether to use a kernel density estimator or not.
        display (bool): Whether or not to display the figure.
        ylog (bool, optional): If True, sets the histogram's y-axis to a logarithmic scale. Defaults to False.
        figsize (tuple, optional): The size of the figure shown. Defaults to (10, 6).
        fontsize (int, optional): Font size for annotations within the box plot. Defaults to 8.
        fontcolor (str, optional): Font color for annotations within the box plot. Defaults to 'black'.



    Returns:
        lower_bound (float): The lower bound used in the box plot for outlier detection.
        upper_bound (float): The upper bound used in the box plot for outlier detection.

    Note:
        The plot is saved as a .jpg image in the 'Figures' folder, named after the column with the specified suffix.
    """

    # filename = filename.capitalize()
    # Calculate boxplot stats
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    # Setup the figure
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": (0.25, 0.75)}, sharex=True
    )

    cmap = plt.get_cmap('inferno')

    # --- Boxplot
    sns.boxplot(x=data, ax=ax_box, color='skyblue')

    # Annotate stats on boxplot
    box_stats = {
        'Min': lower_whisker,
        'Q1': q1,
        'Median': median,
        'Q3': q3,
        'Max': upper_whisker
    }

    # ------------------------
    for label, val in box_stats.items():
        ax_box.text(val, 0.02, f'{label}\n{val:.2f}',
                    ha='center', va='bottom', fontsize=fontsize,
                    color=fontcolor, rotation=45, weight='bold')

    ax_box.set(xlabel='')

    # --- Histogram
    hist_plot  = sns.histplot(data, bins='auto', kde=kde, color='steelblue', edgecolor='black', ax=ax_hist)

    # Annotate stats on boxplot
    hist_stats = {
        'Skew': skew(data),
        'STD': np.std(data),
        'Mode': mode(data, keepdims=True).mode[0],
        'Mean': np.mean(data)
    }

    # Example stats text
    stats_text = '\n'.join([
        f'Mean: {data.mean():.2f}',
        f'Mode: {data.mode().values[0]:.2f}',
        f'Std: {data.std():.2f}',
        f'Skew: {data.skew():.2f}'
    ])

    # Draw rectangle (manual position and size in Axes coordinates: 0–1 range)
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    ax_hist.text(
        0.85, 0.6, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=bbox_props, ha='center',
        va='bottom', fontsize=fontsize+2, color=fontcolor, weight='bold'
    )

    # Accessing the patches (bars) and bins from the plot
    patches = hist_plot.patches  # These are the bars of the histogram

    # To get the counts, you can use `patches` to calculate the heights (counts)
    n = [patch.get_height() for patch in patches]

    # Gradient fill on histogram bars
    for patch, count in zip(patches, n):
        color = cmap(0.3 + 0.7 * count / max(n))
        patch.set_facecolor(color)

    # Annotate count values on bars
    for patch, count in zip(patches, n):
        if count > 0:
            ax_hist.text(patch.get_x() + patch.get_width() / 2,
                         count,
                         f'{int(count)}',
                         ha='center', va='bottom', fontsize=8)

    # Titles and labels
    ax_hist.set_title(f'Distribution of {filename}', fontweight='bold')
    ax_hist.set_xlabel(filename, fontweight='bold')
    ax_hist.set_ylabel('Count', fontweight='bold')

    plt.tight_layout()

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if display:
        plt.show()
    else:
        plt.close()

    return lower_whisker, upper_whisker

def barchart(data, filename, fontsize=14, verbose=True):
    """
    Generates and saves a bar chart based on the frequency of categorical data.

    Parameters:
    -----------
    data : pandas.Series or list-like
        The data column containing categorical values to be plotted in the bar chart.

    filename : str
        The name of the output image file to be saved in the 'Figures' directory (without path).

    fontsize : int, optional (default=14)
        Font size for axis labels and tick marks.

    verbose : bool, optional (default=True)
        If True, displays the plot after saving. If False, suppresses plot display.
    """

    # filename = filename.capitalize()
    # Get frequency counts
    counts = data.value_counts()
    # counts.index = counts.index.str.capitalize()

    # Set figure size
    plt.figure(figsize=(8, 6))

    # Define custom colors using Seaborn's palette
    colors = sns.color_palette("pastel")[:len(counts)]

    # Create bar chart with custom aesthetics
    bars = plt.bar(counts.index, counts, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add labels on top of bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, 1.03*bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', fontsize=fontsize, fontweight='bold', color='black')

        prc = round(bar.get_height()/len(data)*100, 1)
        plt.text(bar.get_x() + bar.get_width()/2, 0.5*bar.get_height(), f'{float(prc)}' + '%',
                 ha='center', fontsize=fontsize, fontweight='bold', color='black')

    # Style the chart
    plt.title('Distribution of ' + filename, fontsize=fontsize+4, fontweight='bold', color='#333333')
    plt.xlabel(filename, fontsize=fontsize+2, fontweight='bold', color='#555555')
    plt.ylabel('Frequency Count', fontsize=fontsize+2, fontweight='bold', color='#555555')

    # Adjust grid aesthetics
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Remove top and right borders for a cleaner look
    plt.box(False)

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    # Save & show the plot
    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

def catNum(df, numeric_col, categorical_col, verbose=True):
    """
    Creates a bivariate visualization for one continuous numeric column and one categorical column.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    numeric_col (str): The name of the continuous numeric column.
    categorical_col (str): The name of the categorical column.

    This function creates a boxplot to show the distribution of the numeric variable across categories.
    """
    # Set figure size and style
    plt.figure(figsize=(14, 6))

    # === Violin Plot ===
    plt.subplot(1, 2, 1)
    sns.violinplot(x=categorical_col, y=numeric_col, hue=categorical_col,
                   data=df, inner='quartile',
                   palette='Set2', dodge=False, legend=False)
    plt.title(numeric_col + ' Distribution by ' + categorical_col + ' (Violin Plot)', fontsize=14, fontweight='bold')
    plt.xlabel(categorical_col, fontsize=14, fontweight='bold')
    plt.ylabel(numeric_col, fontsize=14, fontweight='bold')

    # === Histogram + KDE Overlay ===
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=numeric_col, hue=categorical_col, kde=True,
                 element='step', stat='density', common_norm=False,palette='Set1')
    plt.title(numeric_col + ' Distribution by ' + categorical_col + ' (Histogram + KDE)', fontsize=14, fontweight='bold')
    plt.xlabel(numeric_col, fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')

    # Display both plots
    plt.tight_layout()
    plt.savefig('Figures/' + numeric_col + 'by' + categorical_col + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

def scatter_pearson(x, y, c1, c2, verbose=True):
    """
        Creates a scatter plot of two continuous variables and computes the Pearson correlation coefficient.

        Parameters:
        -----------
        x : array-like or pandas.Series
            Values for the x-axis (independent variable).

        y : array-like or pandas.Series
            Values for the y-axis (dependent variable).

        c1 : str
            Label for the x-axis (e.g., variable name).

        c2 : str
            Label for the y-axis (e.g., variable name).

        verbose : bool, optional (default=True)
            If True, displays the scatter plot with the correlation.

        Returns:
        --------
        pearson_corr : float
            The computed Pearson correlation coefficient between x and y.
        """

    c1 = c1.capitalize()
    c2 = c2.capitalize()

    """Applies Box-Cox transformation if skewness is high."""
    if abs(skew(x)) >= 1:
        x, best_lambda = boxcox(x + 1)
    if abs(skew(y)) >= 1:
        y, best_lambda = boxcox(y + 1)

    # Compute correlation
    correlation_matrix = np.corrcoef(x, y)

    # Extract correlation coefficient
    pearson_corr = correlation_matrix[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5})

    # Draw rectangle (manual position and size in Axes coordinates: 0–1 range)
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)

    plt.text(
        0.9, 0.8, f'r: {pearson_corr:.2f}',
        transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=bbox_props, ha='center',
        va='bottom', fontsize=14 + 2, color='black', weight='bold'
    )

    plt.title(c1+'vs'+c2, fontsize=16, fontweight='bold')
    plt.xlabel(c1, weight='bold')
    plt.ylabel(c2, weight='bold')

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + c1+'vs'+c2 + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return pearson_corr

def forward_stepwise_selection(X, y, threshold_in=0.05, verbose=True):
    """
    Perform forward stepwise selection for linear regression.
    Args:
        X (pd.DataFrame): Candidate predictor variables (already preprocessed, e.g., dummies for categoricals).
        y (pd.Series): Target variable.
        threshold_in (float): p-value threshold for adding a variable.
        verbose (bool): Whether to print progress.
    Returns:
        List of selected features.
    """
    included = []
    while True:
        changed = False
        # Find remaining features not yet included
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        # Find the best candidate
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f'Add {best_feature:30} with p-value {best_pval:.6f}')
        if not changed:
            break
    return included

def backward_stepwise_elimination(X, y, threshold_out=0.05, verbose=True):
    """
    Perform backward stepwise elimination for linear regression.
    Removes features with p-value above threshold_out.
    """
    features = list(X.columns)
    while True:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        max_pval = pvalues.max()
        if max_pval > threshold_out:
            excluded_feature = pvalues.idxmax()
            features.remove(excluded_feature)
            if verbose:
                print(f"Remove {excluded_feature:30} with p-value {max_pval:.6f}")
        else:
            break
    return features

def rfe_statsmodels(X, y, n_features_to_select=5, verbose=True):
    """
    Recursive Feature Elimination using statsmodels OLS.
    Removes the least significant feature (highest p-value) at each iteration.
    Stops when n_features_to_select features remain.
    """
    features = list(X.columns)
    while len(features) > n_features_to_select:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        # Exclude intercept from p-values
        pvalues = model.pvalues.iloc[1:]
        worst_feature = pvalues.idxmax()
        if verbose:
            print(f"Removing {worst_feature} (p-value: {pvalues[worst_feature]:.4f})")
        features.remove(worst_feature)
    return features

def plot_residuals_analysis(model):
    """
    Plots residuals vs fitted values and autocorrelation of residuals using provided fig and ax.
    If fig and ax are None, creates new ones.
    """
    # Get fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted plot on the first axis
    ax[0].scatter(fitted_vals, residuals, alpha=0.5)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].tick_params(axis='x', rotation=30)
    ax[0].set_xlabel('Fitted values')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Residuals vs Fitted Values', weight='bold')

    # Autocorrelation plot of residuals on the second axis
    sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05, zero=False, ax=ax[1])
    ax[1].set_title('Autocorrelation of Residuals', weight='bold')
    ax[1].set_xlabel('Lag')           # X-axis label
    ax[1].set_ylabel('Autocorrelation') # Y-axis label

    plt.tight_layout()
    plt.savefig('Figures/' + 'residuals_analysis' + '.jpg', dpi=300)
    plt.show()


verbose = 1
# Change global font to 'Arial', size 14
plt.rcParams['font.family'] = 'Georgia'
plt.rcParams['font.size'] = 14

# Read Dataset file into a DataFrame
df = pd.read_csv("data/D600 Task 1 Dataset 1 Housing Information.csv")

# Preparing the dataset for our analysis
df_clean = df.copy()
df_clean = df_clean.drop(['ID', 'PreviousSalePrice'], axis=1)
df_clean['Windows'] = df_clean['Windows'].abs()

# ==============================================================================
# part C. Descriptive Statistics and Visualizations
# ==============================================================================

# Visualization of the Numerical Variables Distributions
numeric = df_clean.select_dtypes(include=np.number)
for col in numeric:
    if col not in ['NumBedrooms', 'Floors', 'IsLuxury']:
        # Univariate Visualizations (Histogram & BoxPlot)
        plot_histogram_boxplot(df_clean[col], col, display=verbose)
        if col!='Price':
            # Bivariate Visualizations (Scatter Plot with Pearson Coefficient)
            scatter_pearson(df_clean['Price'], df_clean[col], 'Price', col, verbose=verbose)

# Visualization of the Categorical Variables
for col in ['Fireplace', 'Garage', 'IsLuxury', 'HouseColor', 'NumBedrooms', 'Floors']:
    barchart(df_clean[col], col, verbose=verbose)
    catNum(df_clean, 'Price', col, verbose=verbose)

# ==============================================================================
# part D. Data Analysis and Report
# ==============================================================================

# Split the dataset: 80% for training, 20% for testing
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

# Save the splits to CSV files
train_df.to_csv('data/train_dataset.csv', index=False)
test_df.to_csv('data/test_dataset.csv', index=False)

print("Training and test datasets created and saved as 'train_dataset.csv' and 'test_dataset.csv'.")

# Linear Regression with Step Forward Optimization
X_train = train_df.drop(columns=['Price'])
y_train = train_df['Price']
X_train = pd.get_dummies(X_train, drop_first=True)  # Encode categoricals if needed
# This will find all boolean columns and convert them to int
bool_cols = X_train.select_dtypes(include='bool').columns
X_train[bool_cols] = X_train[bool_cols].astype(int)

# # Create quadratic term for SchoolRating on train set
# X_train['SchoolRating_sq'] = X_train['SchoolRating'] ** 2
# X_train['AgeOfHome_sq'] = X_train['AgeOfHome'] ** 2
# X_train['EmploymentRate_sq'] = X_train['EmploymentRate'] ** 2
# X_train['TransportAccess_sq'] = X_train['TransportAccess'] ** 2

# Normalization -------------------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

# Feature Selection --------------------------------
selected_features = forward_stepwise_selection(X_train, y_train)
# selected_features = backward_stepwise_elimination(X_train, y_train)
# selected_features = rfe_statsmodels(X_train_scaled, y_train, n_features_to_select=15)

# Training ------------------------------------------
Xtrain_selected = sm.add_constant(X_train_scaled[selected_features])
model = sm.OLS(y_train, Xtrain_selected).fit()
print(model.summary())


# part D3. MSE on training set ----------------------
# Predict the target values using the optimized model on the training data
ytrain_pred = model.predict(Xtrain_selected)
mse_value_train = mse(y_train, ytrain_pred)
print(f"Mean Squared Error on training set: {mse_value_train}")

# part D4. MSE on training set ----------------------
X_test = test_df.drop(columns=['Price'])
y_test = test_df['Price']
X_test = pd.get_dummies(X_test, drop_first=True)  # Encode categoricals if needed
# This will find all boolean columns and convert them to int
bool_cols = X_test.select_dtypes(include='bool').columns
X_test[bool_cols] = X_test[bool_cols].astype(int)

# # Create quadratic term for SchoolRating on train set
# X_test['SchoolRating_sq'] = X_test['SchoolRating'] ** 2
# X_test['AgeOfHome_sq'] = X_test['AgeOfHome'] ** 2
# X_test['EmploymentRate_sq'] = X_test['EmploymentRate'] ** 2
# X_test['TransportAccess_sq'] = X_test['TransportAccess'] ** 2

# Normalization -------------------------------------
X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

Xtest_selected = sm.add_constant(X_test_scaled[selected_features])

# Predict the target values using the optimized model on the test data
ytest_pred = model.predict(Xtest_selected)
mse_value_test = mse(y_test, ytest_pred)
print(f"Mean Squared Error on test set: {mse_value_test}")

# Plotting the Residuals Analysis
plot_residuals_analysis(model)

