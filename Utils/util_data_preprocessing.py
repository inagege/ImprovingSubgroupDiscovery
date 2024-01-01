import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
import imblearn as il
from collections import Counter
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture


def split_x_ones_zeros_(x, y):
    x_zeros_temp = pd.DataFrame(columns=x.columns)
    x_ones_temp = pd.DataFrame(columns=x.columns)
    x = pd.DataFrame(x.values)

    for i, row in x.iterrows():
        if y[i] == 0:
            x_zeros_temp = pd.concat([x_zeros_temp, pd.DataFrame(row).T])
        else:
            x_ones_temp = pd.concat([x_ones_temp, pd.DataFrame(row).T])

    return x_zeros_temp, x_ones_temp


def scaling_zscore(x, y):
    """
    Perform scaling using z-score.

    Parameters:
    - x (pd.DataFrame): Input features.
    - y (np.ndarray): Target variable.

    Returns:
    - .
    """

    x_zeros, x_ones = split_x_ones_zeros_(x, y)
    y_temp = []

    # Calculate z-scores for each column
    z_scores_ones = pd.DataFrame(zscore(x_ones.values))
    z_scores_zeros = pd.DataFrame(zscore(x_zeros.values))

    for i in range(z_scores_ones.shape[0]):
        y_temp.append(1)

    for i in range(z_scores_zeros.shape[0]):
        y_temp.append(0)

    return pd.concat([z_scores_ones, z_scores_zeros]), np.array(y_temp)


def outlier_detection_z_score(x, y):
    """
    Perform outlier detection using z-scores and remove outliers.

    Parameters:
    - x (pd.DataFrame): Input features.
    - y (np.ndarray): Target variable.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: DataFrame with outliers removed and corresponding target variable.
    """

    x_zeros_temp, x_ones_temp = split_x_ones_zeros_(x, y)

    # Calculate z-scores for each column
    z_scores_ones = pd.DataFrame(zscore(x_ones_temp.values))
    z_scores_ones.fillna(0, inplace=True)
    z_scores_zeros = pd.DataFrame(zscore(x_zeros_temp.values))
    z_scores_zeros.fillna(0, inplace=True)
    add_value = 0.0
    threshold = 2.5
    y_temp = []

    while len(y_temp) < len(x) / 2:
        # Specify the z-score threshold for outlier removal
        threshold = threshold + add_value

        y_temp = []
        add = True

        for i in range(z_scores_ones.shape[0]):
            for j in range(z_scores_ones.shape[1]):
                if abs(z_scores_ones.iloc[i, j]) >= threshold:
                    add = False
            if add:
                y_temp.append(1)
            add = True

        for i in range(z_scores_zeros.shape[0]):
            for j in range(z_scores_zeros.shape[1]):
                if abs(z_scores_zeros.iloc[i, j]) >= threshold:
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


def add_data_with_smote(x, y):
    counter = Counter(y)
    labeled_zero = counter[0]
    labeled_one = counter[1]

    if labeled_one < 6:
        return x, y

    if labeled_zero < 6:
        return x, y


    labeled_zero = math.ceil(1500 * labeled_zero / len(y))
    labeled_one = math.ceil(1500 * labeled_one / len(y))

    sampling_strategy = {0: labeled_zero + counter[0], 1: labeled_one + counter[1]}
    oversample = il.over_sampling.SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5)
    x, y = oversample.fit_resample(x, y)

    return x, y


def add_data_with_smote_dtc(x, y):
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 3})
    # Train the model on the training data
    rf_classifier.fit(x, y)

    if x.shape[0] < 1500:
        x, y = add_data_with_smote(x, y)

    y = rf_classifier.predict(x)

    return x, y


def add_data_with_dtc(x, y):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 3})
    # Train the model on the training data
    rf_classifier.fit(x, y)

    x_new = np.random.rand(1500, x.shape[1])
    x_new = pd.DataFrame(x_new)
    x = pd.concat([x, x_new], axis=0)
    x = pd.DataFrame(x.values)

    y = rf_classifier.predict(x)

    return x, y


def add_data_gmm(x, y):

    counter = Counter(y)
    labeled_zero = counter[0]
    labeled_one = counter[1]

    if labeled_zero < 2 or labeled_one < 2:
        return x, y

    # Separate data based on labels
    x_class_0 = x[y == 0]
    x_class_1 = x[y == 1]

    n_components_list_zero = []
    n_components_list_one = []

    for i in range(10):
        if round(0.1 * i * labeled_zero) > 1:
            n_components_list_zero.append(round(0.1*i*labeled_zero))
    for i in range(10):
        if round(0.1 * i * labeled_one) > 1:
            n_components_list_one.append(round(0.1*i*labeled_one))

    # Fit GMMs and calculate BIC for each
    bic_values = []
    for n_components in n_components_list_zero:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(x_class_0)
        bic_values.append(gmm.bic(x_class_0))

    # Choose the best n_components based on BIC
    best_n_components_zero = n_components_list_zero[np.argmin(bic_values)]

    bic_values = []
    for n_components in n_components_list_one:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(x_class_1)
        bic_values.append(gmm.bic(x_class_1))

    # Choose the best n_components based on BIC
    best_n_components_one = n_components_list_one[np.argmin(bic_values)]


    # Fit Gaussian Mixture Models for each class
    gmm_class_0 = GaussianMixture(n_components=best_n_components_zero)
    gmm_class_1 = GaussianMixture(n_components=best_n_components_one)

    gmm_class_0.fit(x_class_0)
    gmm_class_1.fit(x_class_1)

    labeled_zero = math.ceil(1500 * labeled_zero / len(y))
    labeled_one = math.ceil(1500 * labeled_one / len(y))

    synthetic_samples_class_0 = pd.DataFrame(gmm_class_0.sample(labeled_zero)[0])
    synthetic_samples_class_1 = pd.DataFrame(gmm_class_1.sample(labeled_one)[0])

    x = pd.concat([x, synthetic_samples_class_1, synthetic_samples_class_0], axis=0)
    x = x.apply(lambda col: np.clip(col, 0, 1))
    x = pd.DataFrame(x.values, columns=x.columns)
    y = np.concatenate([y, np.ones(labeled_one)])
    y = np.concatenate([y, np.zeros(labeled_zero)])


    return x, y


def duplicate_point(x, y, zero_or_one):
    x_dup = x.loc[y == zero_or_one].copy()

    # Append the duplicated rows to the original DataFrame
    x = pd.concat([x, x_dup], ignore_index=True)
    y = np.concatenate([y, y[y == zero_or_one]])

    return x, y


