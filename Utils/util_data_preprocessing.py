import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

def outlier_detection_z_score(x, y):
    """
    Perform outlier detection using z-scores and remove outliers.

    Parameters:
    - x (pd.DataFrame): Input features.
    - y (np.ndarray): Target variable.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: DataFrame with outliers removed and corresponding target variable.
    """

    x_zeros_temp = pd.DataFrame(columns=x.columns)
    x_ones_temp = pd.DataFrame(columns=x.columns)

    for i, row in x.iterrows():
        if y[i] == 0:
            x_zeros_temp = pd.concat([x_zeros_temp, pd.DataFrame(row).T])
        else:
            x_ones_temp = pd.concat([x_ones_temp, pd.DataFrame(row).T])

    # Calculate z-scores for each column
    z_scores_ones = pd.DataFrame(zscore(x_ones_temp.values))
    z_scores_zeros = pd.DataFrame(zscore(x_zeros_temp.values))
    add_value = 0.25
    threshold = 1
    y_temp = []

    while len(y_temp) < 10:
        # Specify the z-score threshold for outlier removal
        threshold = threshold + add_value

        y_temp = []
        add = True

        for i in range(z_scores_ones.shape[0]):
            for j in range(z_scores_ones.shape[1]):
                if abs(z_scores_ones.iloc[i, j]) > threshold:
                    add = False
            if add:
                y_temp.append(1)
            add = True

        for i in range(z_scores_zeros.shape[0]):
            for j in range(z_scores_zeros.shape[1]):
                if abs(z_scores_zeros.iloc[i, j]) > threshold:
                    add = False
            if add:
                y_temp.append(0)
            add = True

    # Identify and remove rows with outliers based on the threshold
    df_no_outliers_ones = pd.DataFrame(x_ones_temp[(abs(z_scores_ones) < threshold).all(axis=1).values])
    df_no_outliers_zeros = pd.DataFrame(x_zeros_temp[(abs(z_scores_zeros) < threshold).all(axis=1).values])

    return pd.concat([df_no_outliers_ones, df_no_outliers_zeros]), np.array(y_temp)


def discretize_with_kbins(x, y):
    """
    Discretize continuous features using KBinsDiscretizer.

    Parameters:
    - x (pd.DataFrame): Input features.
    - y (np.ndarray): Target variable.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: Discretized features and corresponding target variable.
    """

    # number of bins
    n_bins = round((np.sqrt(x.shape[0])) * 0.6 * x.shape[1] * x.shape[1] * x.shape[1])

    # Initialize KBinsDiscretizer
    kbins_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    # Discretize and replace values in each feature column independently
    for column in x.columns:
        # Reshape the column to a 2D array
        feature_values = x[column].values.reshape(-1, 1)

        # Fit and transform the feature column
        x[column] = kbins_discretizer.fit_transform(feature_values)

    # Initialize MinMaxScaler
    min_max_scaler = MinMaxScaler()

    # Scale the entire DataFrame to ensure values are between 0 and 1
    df_scaled = pd.DataFrame(min_max_scaler.fit_transform(x), columns=x.columns)

    return df_scaled, y