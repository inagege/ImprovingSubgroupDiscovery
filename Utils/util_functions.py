import statistics

from Utils import prim_dens
from ema_workbench.analysis import prim as prim_emaworkbench
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import gzip
import sys
from Utils.data_generators import *
from Utils.util_data_preprocessing import *
import random


def get_data(data_name):
    """
    Load a dataset based on the provided data_name.

    Parameters:
    - data_name (str): Name of the dataset to load ('Bryant', 'Rozenberg', 'Susy', or 'Higgs').

    Returns:
    - pd.DataFrame: Loaded DataFrame containing the dataset.

    Raises:
    - ValueError: If an unsupported data_name is provided.
    """

    data = []
    if (data_name) == 'Bryant':
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/Bryant et al 2010.csv')

    if (data_name) == 'Rozenberg':
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/Rozenberg et al 2014.csv')

    if (data_name) == 'Susy':
        # Path to the Susy dataset .zip file
        gz_file_path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/SUSY.csv.gz'

        # Open the Gzip-compressed CSV file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            # Read the dataset into a DataFrame
            data = pd.read_csv(gz_file)
        columns = ['label', 'lepton  1 pT', 'lepton  1 eta', 'lepton  1 phi', 'lepton  2 pT', 'lepton  2 eta',
                   'lepton  2 phi', 'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R',
                   'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        data = pd.DataFrame(data.values, index=data.index, columns=columns)

    if (data_name) == 'Higgs':
        gz_file_path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/HIGGS.csv.gz'

        # Open the Gzip-compressed CSV file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            # Read the dataset into a DataFrame
            data = pd.read_csv(gz_file)

        columns = ['label', 'lepton  pT', 'lepton  eta', 'lepton  phi', 'missing energy magnitude',
                   'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta',
                   'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt',
                   'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        data = pd.DataFrame(data.values, index=data.index, columns=columns)

    return data

def Higgs():
    return None

def Susy():
    return None


def get_list_all_precisions_recalls_boxes(x, y, package, quality_function):
    """
    Get precision, recall, and boxes for a given package.

    Parameters:
    - x (pd.DataFrame): Input features.
    - y (pd.Series): Target variable.
    - package (str): Name of the package to use ('prim' or 'ema_workbench').

    Returns:
    - Tuple[List[float], List[float], Any]: Tuple containing precision, recall, and boxes.
    """

    if package == 'prim':
        prim_alg = prim_dens.PRIMdens(x.values, y, alpha=0.1, quality_measurement=quality_function)
        prim_alg.fit()
        return prim_alg.get_precisions(), prim_alg.get_recalls(), prim_alg.get_boxes()
    if package == "ema_workbench":
        recall = []
        precision = []

        prim_alg = prim_emaworkbench.Prim(x, y, peel_alpha=0.1, threshold=0.7)
        boxes = prim_alg.find_box()

        for index, row in boxes.peeling_trajectory.iterrows():
            recall.append(row['coverage'])
            precision.append(row['density'])
        del prim_alg
        return precision, recall, boxes.box_lims


def define_y_x_all_data(data, stratify_feature, drop_feature, package):
    """
    Define input features (x) and target variable (y) for the given dataset and package.

    Parameters:
    - data_name (str): Name of the dataset.
    - stratify_feature (str): Name of the feature used for stratification, normally label.
    - drop_feature (str): Name of the feature to be dropped e.g. other labels for data.
    - package (str): Name of the package to use ('prim' or 'ema_workbench').

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: Tuple containing input features (x) and target variable (y).
    """

    if package == 'prim':
        scaler = MinMaxScaler()
        data1 = data
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, index=data1.index, columns=data1.columns)

    y = data[stratify_feature]
    x = pd.DataFrame(data.drop(columns=drop_feature))

    return pd.DataFrame(x.values), y


def flat_prec_rec(prec, rec):
    """
    Flatten precision and recall lists.

    Parameters:
    - prec (List[List[float]]): List of precision values.
    - rec (List[List[float]]): List of recall values.

    Returns:
    - Tuple[List[float], List[float]]: Flattened precision and recall lists.
    """

    prec = [item for sublist in prec for item in sublist]
    rec = [item for sublist in rec for item in sublist]
    return prec, rec


def define_train_test_split(data_name, stratify_feature, drop_feature, test_size, package):
    """
    Define train and test splits for a given dataset and package.

    Parameters:
    - data_name (str): Name of the dataset.
    - stratify_feature (str): Name of the feature used for stratification.
    - drop_feature (str): Name of the feature to be dropped.
    - test_size (float): Proportion of the dataset to include in the test split.
    - package (str): Name of the package to use ('prim' or 'ema_workbench').

    Returns:
    - Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Tuple containing x, y, x_test, and y_test.
    """

    data = get_data(data_name)

    if package == 'prim':
        scaler = MinMaxScaler()
        data1 = data
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, index=data1.index, columns=data1.columns)

    # sampeling a subset of the whole data set
    sample_train, sample_test = train_test_split(data, test_size=test_size, stratify=data[stratify_feature])

    sample_train = pd.DataFrame(data=sample_train.values, columns=sample_train.columns)
    sample_test = pd.DataFrame(data=sample_test.values, columns=sample_test.columns)

    y = sample_train[stratify_feature]
    x = pd.DataFrame(sample_train.drop(columns=drop_feature))

    y_test = sample_test[stratify_feature]
    x_test = pd.DataFrame(sample_test.drop(columns=drop_feature))

    return pd.DataFrame(x.values), y, x_test, y_test


def calculate_precision_recall_test_data_allboxes(lims, x_test, y_test):
    """
    Calculate precision and recall for multiple boxes on test data.

    Parameters:
    - lims (List[pd.DataFrame]): List of box limits (dataframes).
    - x_test (pd.DataFrame): Test input features.
    - y_test (pd.Series): Test target variable.

    Returns:
    - Tuple[List[float], List[float]]: Precision and recall lists for each box.
    """

    precision_test = []
    recall_test = []

    # iterate over limit entries which is list of dataframes
    for j in range(len(lims)):
        # Initialize TP, FP, TN, FN counters
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        box = lims[j]
        box = pd.DataFrame(box)
        x_test_temp = pd.DataFrame(x_test.values)
        # Iterate over each row of temp_data
        for row_index, row in x_test_temp.iterrows():
            is_within_limits = True
            # Check if entry lies within the specified limits
            for a, (column, value) in enumerate(row.iteritems()):
                is_within_limits = (box.iloc[0, a] <= value <= box.iloc[1, a]) & is_within_limits

            tp, fp, tn, fn = calculate_tp_fp_tn_fn(y_test, row_index, is_within_limits, tp, fp, tn, fn)

        precision = recall = 0  # Default values

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)

        precision_test.append(precision)
        recall_test.append(recall)

    return precision_test, recall_test


def calculate_precision_test_data_onebox(lims, x_test, y_test):
    """
    Calculate precision for one box on test data.

    Parameters:
    - lims (pd.DataFrame): Box limits.
    - x_test (pd.DataFrame): Test input features.
    - y_test (pd.Series): Test target variable.

    Returns:
    - float: Precision for the given box.
    """
    is_within_limits = True

    # Iterate over each row of temp_data
    # Initialize TP, FP, TN, FN counters
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Assuming x_test is a 2D NumPy array and y_test is a 1D NumPy array
    for row_index in range(x_test.shape[0]):
        for col_index in range(x_test.shape[1]):
            element = x_test[row_index, col_index]

            # Check if the element is within the limits for the current column
            is_within_limits = (lims.iloc[0, col_index] <= element <= lims.iloc[
                1, col_index]) and is_within_limits  # Calculate column index

        tp, fp, tn, fn = calculate_tp_fp_tn_fn(y_test, row_index, is_within_limits, tp, fp, tn, fn)
        is_within_limits = True

    if (tp == 0):
        return 0
    else:
        return tp / (tp + fp)


def generate_data(function_string, dimension_max, numb_of_points, train_or_test, package, data=None):
    """
    Generate synthetic data based on a given function.

    Parameters:
    - function_string (str): String representation of the function fich is used to determine label.
    - dimension_max (int): Maximum dimensionality of the generated data.
    - numb_of_points (int): Number of data points to generate.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: Generated input features (x) and target variable (y).
    """

    if 'calculate' in function_string.__name__:
        y = []

        if train_or_test == 'test':
            np.random.seed(42)

        x = np.random.rand(numb_of_points, dimension_max)
        x = pd.DataFrame(x)
        for index, row in x.iterrows():
            y.append(function_string(row))
    else:
        x, y = define_y_x_all_data(data, 'label', 'label', package)
        x, x_test, y, y_test = train_test_split(x, y, train_size=numb_of_points, stratify=y)
        y = y.to_numpy()

    return x, y


def scale_y(y, function_string):
    if 'calculate' in function_string.__name__:
        min_value = np.min(y)
        max_value = np.max(y)
        y = (y - min_value) / (max_value - min_value)
        threshold = 0.5

        if function_string.__name__ == 'calculate_y_sobol_levitan1999':
            threshold = 0.01

        if function_string.__name__ == 'calculate_y_oakley_ohagan2004':
            threshold = 0.6

        if function_string.__name__ == 'calculate_y_moon2010':
            threshold = 0.6

        y = np.where(y > threshold, 1, 0)

    return y


def add_precision_recall_of_each_box_to_list_each_box(prec_in, rec_in, prec_out, rec_out):
    """
    Add precision and recall values of each box to separate lists.

    Parameters:
    - prec_in (List[float]): List of precision values.
    - rec_in (List[float]): List of recall values.
    - prec_out (List[List[float]]): List to store precision values for each box.
    - rec_out (List[List[float]]): List to store recall values for each box.

    Returns:
    - Tuple[List[List[float]], List[List[float]]]: Updated lists of precision and recall values.
    """

    if len(prec_in) > len(prec_out):
        while len(prec_in) > len(prec_out):
            prec_out.append([])
            rec_out.append([])
    for index, element in enumerate(prec_in):
        prec_out[index].append(element)
    for index, element in enumerate(rec_in):
        rec_out[index].append(element)

    return prec_out, rec_out


def get_precision_and_recall_train_test(number_of_repeats, function_string, package, preprocessing_string,
                                        dimension_max, quality_function='precision'):
    prec_train = []
    rec_train = []

    prec_test = []
    rec_test = []

    for i in range(number_of_repeats):
        sys.stdout.write('\r' + 'experiment' + ' ' + str(i + 1) + '/' + str(number_of_repeats))

        x, y = generate_data(function_string, dimension_max, 200, 'train', package)
        folds = KFold(n_splits=5, shuffle=True)

        y = scale_y(y, function_string)

        for fold, (train_id, test_id) in enumerate(folds.split(x)):
            x_train, x_test = x.iloc[train_id], x.iloc[test_id]
            y_train, y_test = y[train_id], y[test_id]

            if preprocessing_string is not None:
                for item in preprocessing_string:
                    x_train, y_train = item(x_train, y_train)

            precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x_train, y_train, package,
                                                                               quality_function)

            prec_train, rec_train = add_precision_recall_of_each_box_to_list_each_box(precisions, recalls, prec_train,
                                                                                      rec_train)

            prec_test_temp, rec_test_temp = calculate_precision_recall_test_data_allboxes(boxes, x_test, y_test)

            prec_test, rec_test = add_precision_recall_of_each_box_to_list_each_box(prec_test_temp, rec_test_temp,
                                                                                    prec_test, rec_test)

    return prec_test, prec_train, rec_test, rec_train


def calculate_tp_fp_tn_fn(y_test, row_index, is_within_limits, tp, fp, tn, fn):
    if is_within_limits & (y_test[row_index] == 1):
        tp = tp + 1
    if is_within_limits & (y_test[row_index] == 0):
        fp = fp + 1
    if (is_within_limits is False) & (y_test[row_index] == 0):
        tn = tn + 1
    if (is_within_limits is False) & (y_test[row_index] == 1):
        fn = fn + 1

    return tp, fp, tn, fn
