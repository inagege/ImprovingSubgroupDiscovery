import matplotlib.pyplot as plt
from Utils import prim_dens
from ema_workbench.analysis import prim as prim_emaworkbench
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import random
import gzip

def get_data(data_name):
    data = []
    if (data_name) == 'Bryant':
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/Bryant et al 2010.csv')

    if (data_name) == 'Rozenberg':
        data = pd.read_csv(
            '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/Rozenberg et al 2014.csv')

    if (data_name) == 'susy':
        # Path to the Susy dataset .zip file
        gz_file_path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/SUSY.csv.gz'

        # Open the Gzip-compressed CSV file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            # Read the dataset into a DataFrame
            data = pd.read_csv(gz_file)
        columns = ['label', 'lepton  1 pT', 'lepton  1 eta', 'lepton  1 phi', 'lepton  2 pT', 'lepton  2 eta',
                   'lepton  2 phi', 'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R',
                   'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        data = pd.DataFrame(data, index=data.index, columns=columns)

    if (data_name) == 'higgs':
        gz_file_path = '/Users/inagege/Documents/00_Uni/Bachelorarbeit/ImprovingSubgroupDiscovery/Data/HIGGS.csv.gz'

        # Open the Gzip-compressed CSV file
        with gzip.open(gz_file_path, 'rb') as gz_file:
            # Read the dataset into a DataFrame
            data = pd.read_csv(gz_file)

        columns = ['label', 'lepton  pT', 'lepton  eta', 'lepton  phi', 'missing energy magnitude',
                   'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta',
                   'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt',
                   'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        data = pd.DataFrame(data, index=data.index, columns=columns)

    return data


def visualize_precision_and_recall(precision_baseline, recall_baseline, precision2, recall2, first_label, second_label):
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(recall_baseline, precision_baseline, c='blue', marker='o', label=first_label)

    plt.scatter(recall2, precision2, c='red', marker='o', label=second_label)

    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # Add labels and a legend
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall')
    plt.legend()

    # Display the plot
    plt.grid(True)
    return plt


def get_list_all_precisions_recalls_boxes(x, y, package):
    if package == 'prim':
        prim_alg = prim_dens.PRIMdens(x.values, y, alpha=0.1)
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


def define_y_x_all_data(data_name, stratify_feature, drop_feature, package):

    data = get_data(data_name)

    if package == 'prim':
        scaler = MinMaxScaler()
        data1 = data
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, index=data1.index, columns=data1.columns)

    y = data[stratify_feature]
    x = pd.DataFrame(data.drop(columns=drop_feature))

    return x, y

def flat_prec_rec(prec, rec):
    prec = [item for sublist in prec for item in sublist]
    rec = [item for sublist in rec for item in sublist]
    return prec, rec

def define_train_test_split(data_name, stratify_feature, drop_feature, test_size, package):

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

    return x, y, x_test, y_test

def calculate_precision_recall_test_data_allboxes(lims, x_test, y_test):
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
        # Iterate over each row of temp_data
        for row_index, row in x_test.iterrows():
            is_within_limits = True
            # Check if entry lies within the specified limits
            for a, (column, value) in enumerate(row.iteritems()):
                is_within_limits = (box.iloc[0, a] <= value <= box.iloc[1, a]) & is_within_limits

            if is_within_limits & (y_test[row_index] == 1):
                tp = tp + 1
            if is_within_limits & (y_test[row_index] == 0):
                fp = fp + 1
            if (is_within_limits == False) & (y_test[row_index] == 0):
                tn = tn + 1
            if (is_within_limits == False) & (y_test[row_index] == 1):
                fn = fn + 1

        precision = recall = 0  # Default values

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)

        precision_test.append(precision)
        recall_test.append(recall)

    return precision_test, recall_test


def calculate_precision_test_data_onebox(lims, x_test, y_test):
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

        if is_within_limits & (y_test[row_index] == 1):
            tp = tp + 1
        if is_within_limits & (y_test[row_index] == 0):
            fp = fp + 1
        if (is_within_limits == False) & (y_test[row_index] == 0):
            tn = tn + 1
        if (is_within_limits == False) & (y_test[row_index] == 1):
            fn = fn + 1
        is_within_limits = True

    if (tp == 0):
        return 0
    else:
        return tp / (tp + fp)


def generate_data(function_string, dimension_max, numb_of_points):
    y_test = []

    x_test = np.random.rand(numb_of_points, dimension_max)
    x_test = pd.DataFrame(x_test)
    for index, row in x_test.iterrows():
        y_test.append(function_string(row))

    min_value = np.min(y_test)
    max_value = np.max(y_test)
    y_test = (y_test - min_value) / (max_value - min_value)
    y_test = np.where(y_test > 0.5, 1, 0)

    return x_test, y_test

def create_plot_generated_data(function_string, dimension_max, package):
    pts = [50, 100, 200, 400, 800, 1600, 3200, 6400]  # number of points to experiment with
    atrs = [5, 10, 15]  # number of dimensions to experiment with
    res_train = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_train[:] = np.nan
    res_test = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_test[:] = np.nan
    k = 1

    x_test, y_test = generate_data(function_string, dimension_max, 500)

    for n in range(len(pts)):
        for m in range(len(atrs)):
            prec_train = []
            prec_test = []
            for i in range(3):  # for each dataset size (n rows, m columns) do five experiments and average the results
                sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(len(pts) * len(atrs)))
                x, y = generate_data(function_string, dimension_max, pts[n])

                # Select random columns
                selected_columns = random.sample(list(x.columns), atrs[m])
                x = x[selected_columns]
                x_test_temp = x_test[selected_columns]

                precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x, y, package)
                if len(precisions) <= 0:
                    prec_train.append(0)
                else:
                    prec_train.append(precisions[len(precisions) - 1])

                if len(boxes) <= 0:
                    prec_test.append(0)
                else:
                    box = boxes[len(boxes) - 1]
                    box = pd.DataFrame(box)
                    prec_test.append(calculate_precision_test_data_onebox(box, x_test_temp.values, y_test))

            res_train[n, m] = np.mean(prec_train)
            res_test[n, m] = np.mean(prec_test)
            k = k + 1

    plt.imshow(res_train - res_test, cmap='hot')
    plt.yticks(np.arange(len(pts)), pts)
    plt.xticks(np.arange(len(atrs)), atrs)
    plt.colorbar()
    plt.show()


def add_precision_recall_of_each_box_to_list_each_box(prec_in, rec_in, prec_out, rec_out):
    if len(prec_in) > len(prec_out):
        while len(prec_in) > len(prec_out):
            prec_out.append([])
            rec_out.append([])
    for index, element in enumerate(prec_in):
        prec_out[index].append(element)
    for index, element in enumerate(rec_in):
        rec_out[index].append(element)

    return prec_out, rec_out