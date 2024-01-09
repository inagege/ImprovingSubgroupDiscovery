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
        data['Greater than 90%'] = (data['Greater than 90%'] | data['Less than 10%']).astype(int)
        data.drop('Less than 10%', axis=1, inplace=True)
        data.rename(columns={'Greater than 90%': 'label'}, inplace=True)

    if (data_name) == 'Rozenberg':
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/Rozenberg et al 2014.csv')
        data.drop(['SSP2', 'SSP3', 'SSP4'], axis=1, inplace=True)
        data.rename(columns={'SSP1': 'label'}, inplace=True)

    if (data_name) == 'Susy':
        # Path to the Susy dataset .zip file
        gz_file_path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/SUSY.csv.gz'

        # Open the Gzip-compressed CSV file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            # Read the dataset into a DataFrame
            data = pd.read_csv(gz_file, nrows=98049)
        columns = ['label', 'lepton  1 pT', 'lepton  1 eta', 'lepton  1 phi', 'lepton  2 pT', 'lepton  2 eta',
                   'lepton  2 phi', 'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R',
                   'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        data = pd.DataFrame(data.values, index=data.index, columns=columns)
        data = data.iloc[:, [0] + list(range(data.shape[1] - 10, data.shape[1]))]

    if (data_name) == 'Higgs':
        gz_file_path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/HIGGS.csv.gz'

        # Open the Gzip-compressed CSV file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            # Read the dataset into a DataFrame
            data = pd.read_csv(gz_file, nrows=98049)

        columns = ['label', 'lepton  pT', 'lepton  eta', 'lepton  phi', 'missing energy magnitude',
                   'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta',
                   'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt',
                   'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        data = pd.DataFrame(data.values, index=data.index, columns=columns)
        data = data[data.columns.drop(list(data.filter(regex='m_', axis=1)))]
        data = pd.DataFrame(data.values, columns=data.columns)

    if data_name == 'Occupancy':
        data = pd.concat([pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/occupancy/datatest.txt',
            delimiter=","), pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/occupancy/datatest2.txt',
            delimiter=","), pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/occupancy/datatraining.txt',
            delimiter=",")])
        data['date'] = pd.to_datetime(data['date']).dt.hour
        data.rename(columns={'Occupancy': 'label'}, inplace=True)
        data = pd.DataFrame(data.values, columns=data.columns)

    if data_name == "Htru":
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/HTRU_2.csv',
            delimiter=",", header=None)
        data.rename(columns={8: 'label'}, inplace=True)

    if data_name == "Shuttle":
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/shuttle.csv',
            header=None, delimiter=' ')
        data.rename(columns={9: 'label'}, inplace=True)
        data = data.drop(data.columns[0], axis=1)
        data['label'] = data['label'].apply(lambda x: 1 if x == 1 else 0)

    return data


def Higgs():
    return None


def Susy():
    return None


def Bryant():
    return None


def Rozenberg():
    return None


def Occupancy():
    return None


def Htru():
    return None


def Shuttle():
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
        prim_alg = prim_dens.PRIMdens(x.values, y, quality_measurement=quality_function)
        prim_alg.fit()
        return prim_alg.get_precisions(), prim_alg.get_recalls(), prim_alg.get_boxes()
    if package == "ema_workbench":
        recall = []
        precision = []

        prim_alg = prim_emaworkbench.Prim(x, y, peel_alpha=0.05, threshold=0.7)
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
    - x_test (np.ndarray): Test input features.
    - y_test (np.ndarray): Test target variable.

    Returns:
    - Tuple[List[float], List[float]]: Precision and recall lists for each box.
    """

    precision_test = []
    recall_test = []

    # Convert x_test and y_test to numpy arrays if they are DataFrames or Series
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # iterate over each box in lims
    for box in lims:
        box = box  # Convert the DataFrame box to numpy array
        lower_limits = box[0, :]
        upper_limits = box[1, :]

        # Check if each row in x_test is within the box limits
        is_within_limits = np.all((x_test >= lower_limits) & (x_test <= upper_limits), axis=1)

        # Calculate TP, FP, TN, FN
        tp = np.sum((is_within_limits) & (y_test == 1))
        fp = np.sum((is_within_limits) & (y_test == 0))
        tn = np.sum((~is_within_limits) & (y_test == 0))
        fn = np.sum((~is_within_limits) & (y_test == 1))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_test.append(precision)
        recall_test.append(recall)

    return precision_test, recall_test


def calculate_precision_test_data_onebox(lims, x_test, y_test):
    """
    Calculate precision for one box on test data.

    Parameters:
    - lims (pd.DataFrame): Box limits.
    - x_test (np.ndarray): Test input features.
    - y_test (np.ndarray): Test target variable.

    Returns:
    - float: Precision for the given box.
    """

    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Check if each element is within limits
    lower_limits = lims[0, :]
    upper_limits = lims[1, :]
    is_within_limits = np.all((x_test >= lower_limits) & (x_test <= upper_limits), axis=1)

    # Calculate TP, FP, TN, FN
    tp = np.sum(is_within_limits & (y_test == 1))
    fp = np.sum(is_within_limits & (y_test == 0))
    # tn and fn are not used for precision calculation
    # tn = np.sum((~is_within_limits) & (y_test == 0))
    # fn = np.sum((~is_within_limits) & (y_test == 1))

    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision


def calculate_sum_tp_fp_test_data_allboxes(lims, x_test, y_test):
    """
    Calculate the sum of TP and FP for multiple boxes on test data.

    Parameters:
    - lims (List[pd.DataFrame]): List of box limits (dataframes).
    - x_test (np.ndarray): Test input features.
    - y_test (np.ndarray): Test target variable.

    Returns:
    - Tuple[int, int]: Sum of TP and FP for all boxes.
    """

    total_tp = []
    total_fp = []

    # Convert x_test and y_test to numpy arrays if they are DataFrames or Series
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Iterate over each box in lims
    for box in lims:
        box = box  # Convert the DataFrame box to a numpy array
        lower_limits = box[0, :]
        upper_limits = box[1, :]

        # Check if each row in x_test is within the box limits
        is_within_limits = np.all((x_test >= lower_limits) & (x_test <= upper_limits), axis=1)

        # Calculate TP and FP
        tp = np.sum((is_within_limits) & (y_test == 1))
        fp = np.sum((is_within_limits) & (y_test == 0))

        # Aggregate TP and FP
        total_tp.append(tp)
        total_fp.append(fp)

    return np.array(total_tp) + np.array(total_fp)


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
        threshold = np.percentile(y, 70)

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

def add_n_in_box_to_all_values(n_in_box, all_n_in_box):
    if len(n_in_box) > len(all_n_in_box):
        while len(n_in_box) > len(all_n_in_box):
            all_n_in_box.append([])
    for index, element in enumerate(n_in_box):
        all_n_in_box[index].append(element)

    return all_n_in_box


def get_precision_and_recall_train_test(number_of_repeats, function_string, package, preprocessing_string,
                                        dimension_max, quality_function='precision'):
    prec_train = []
    rec_train = []

    prec_test = []
    rec_test = []

    all_n_in_boxes = []
    data = None

    if 'calculate' not in function_string.__name__:
        data = get_data(function_string.__name__)

    for i in range(number_of_repeats):
        sys.stdout.write('\r' + 'experiment' + ' ' + str(i + 1) + '/' + str(number_of_repeats))

        x, y = generate_data(function_string, dimension_max, 100, 'train', package, data)
        folds = KFold(n_splits=5, shuffle=True)

        if 'calculate' in function_string.__name__:
            y = scale_y(y, function_string)

        for fold, (train_id, test_id) in enumerate(folds.split(x)):
            x_train, x_test = x.iloc[train_id], x.iloc[test_id]
            y_train, y_test = y[train_id], y[test_id]

            y_train_temp, x_train_temp = y_train.copy(), x_train.copy(deep=True)

            if preprocessing_string is not None:
                for item in preprocessing_string:
                    x_train, y_train = item(x_train, y_train)

            precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x_train, y_train, package,
                                                                                     quality_function)

            n_in_boxes = calculate_sum_tp_fp_test_data_allboxes(boxes, x_train_temp, y_train_temp)
            all_n_in_boxes = add_n_in_box_to_all_values(n_in_boxes, all_n_in_boxes)

            prec_train, rec_train = add_precision_recall_of_each_box_to_list_each_box(precisions, recalls, prec_train,
                                                                                      rec_train)

            prec_test_temp, rec_test_temp = calculate_precision_recall_test_data_allboxes(boxes, x_test, y_test)

            prec_test, rec_test = add_precision_recall_of_each_box_to_list_each_box(prec_test_temp, rec_test_temp,
                                                                                    prec_test, rec_test)

    return prec_test, prec_train, rec_test, rec_train, all_n_in_boxes


def calculate_tp_fp_tn_fn(y_test, row_index, is_within_limits, tp, fp, tn, fn):
    y_value = y_test[row_index]
    if is_within_limits:
        if y_value == 1:
            tp += 1
        else:
            fp += 1
    else:
        if y_value == 0:
            tn += 1
        else:
            fn += 1

    return tp, fp, tn, fn


def get_data_information(synthetic_or_real):
    data_info = None
    if synthetic_or_real == 'r':
        rozenberg = [Rozenberg, 7, [50, 100, 200]]
        bryant = [Bryant, 14, [50, 100, 200, 400, 800]]
        higgs = [Higgs, 28, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        susy = [Susy, 14, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        occupancy = [Occupancy, 14, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        htru = [Htru, 14, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        shuttle = [Shuttle, 14, [50, 100, 200, 400, 800, 1600, 3200, 6400]]

        columns = ['function', 'dim', 'pts']
        index = ['rozenberg', 'bryant', 'higgs', 'susy', 'occupancy', 'htru', 'shuttle']

        data_info = pd.DataFrame([rozenberg, bryant, higgs, susy, occupancy, htru, shuttle], index=index,
                                 columns=columns)

    if synthetic_or_real == 's':
        moon = [calculate_y_moon2010, 20, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        morris = [calculate_y_morris, 30, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        oakley = [calculate_y_oakley_ohagan2004, 15, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        sobol = [calculate_y_sobol_levitan1999, 20, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        welch = [calculate_y_welch, 20, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        loeppky = [calculate_y_loeppky, 7, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        borehole = [calculate_y_borehole, 8, [50, 100, 200, 400, 800, 1600, 3200, 6400]]

        columns = ['function', 'dim', 'pts']
        index = ['moon', 'morris', 'oakley', 'sobol', 'welch', 'loeppky', 'borehole']
        data_info = pd.DataFrame([moon, morris, oakley, sobol, welch, loeppky, borehole], index=index,
                                 columns=columns)

    return data_info


def get_edited(synthetic_or_real):
    data_info = None
    if synthetic_or_real == 'r':
        rozenberg = [Rozenberg, 7, [50, 100, 200]]
        bryant = [Bryant, 14, [50, 100, 200, 400, 800]]
        higgs = [Higgs, 28, [50, 100, 200, 400, 800]]
        susy = [Susy, 14, [50, 100, 200, 400, 800]]
        occupancy = [Occupancy, 14, [50, 100, 200, 400, 800]]
        htru = [Htru, 14, [50, 100, 200, 400, 800]]
        shuttle = [Shuttle, 14, [50, 100, 200, 400, 800]]

        columns = ['function', 'dim', 'pts']
        index = ['rozenberg', 'bryant', 'higgs', 'susy', 'occupancy', 'htru', 'shuttle']

        data_info = pd.DataFrame([rozenberg, bryant, higgs, susy, occupancy, htru, shuttle], index=index,
                                 columns=columns)

    if synthetic_or_real == 's':
        moon = [calculate_y_moon2010, 20, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        morris = [calculate_y_morris, 30, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        oakley = [calculate_y_oakley_ohagan2004, 15, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        sobol = [calculate_y_sobol_levitan1999, 20, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        welch = [calculate_y_welch, 20, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        loeppky = [calculate_y_loeppky, 7, [50, 100, 200, 400, 800, 1600, 3200, 6400]]
        borehole = [calculate_y_borehole, 8, [50, 100, 200, 400, 800, 1600, 3200, 6400]]

        columns = ['function', 'dim', 'pts']
        index = ['moon', 'morris', 'oakley', 'sobol', 'welch', 'loeppky', 'borehole']
        data_info = pd.DataFrame([moon, morris, oakley, sobol, welch, loeppky, borehole], index=index,
                                 columns=columns)

    return data_info


def get_n_in_box_dataset(function_string, number_of_points, synthetic_or_real):
    if synthetic_or_real == 's':
        path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/PRIM_Performance_Analyzing/Results_Baseline/Scenario_Discovery'
        index = ['calculate_y_moon2010', 'calculate_y_morris', 'calculate_y_oakley_ohagan2004', 'calculate_y_sobol_levitan1999',
                 'calculate_y_welch', 'calculate_y_loeppky', 'calculate_y_borehole']
    else:
        path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/PRIM_Performance_Analyzing/Results_Baseline/Subgroup_Discovery'
        index = ['Rozenberg', 'Bryant', 'Higgs', 'Susy', 'Occupancy', 'Htru', 'Shuttle']

    all_n_points = pd.read_csv(path + '/res_n_in.csv', header=0, index_col=0).T
    all_n_points = pd.DataFrame(all_n_points.values, index=index, columns=[50, 100, 200, 400, 800, 1600, 3200, 6400])

    return all_n_points.loc[function_string.__name__, number_of_points]


def determine_box(function_string, number_of_points, synthetic_or_real, lims, x_train, y_train):
    list_n_in_box_train = calculate_sum_tp_fp_test_data_allboxes(lims, x_train, y_train)
    points_in_basleine = get_n_in_box_dataset(function_string, number_of_points, synthetic_or_real)
    return min(range(len(list_n_in_box_train)), key=lambda x: abs(list_n_in_box_train[x] - points_in_basleine))




