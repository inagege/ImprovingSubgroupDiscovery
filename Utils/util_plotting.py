import numpy as np
import pandas as pd

from Utils.util_functions import *
import sys
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import matplotlib.colors as mcolors


def create_heatmap_best_box_generated_data_precision(function_string, dimension_max, package, preprocessing_string,
                                                     number_of_repeats):
    """
    Create a plot comparing precision results for different dataset sizes and dimensions.

    Parameters:
    - function_string (str): String representation of the function.
    - dimension_max (int): Maximum dimensionality of the generated data.
    - package (str): Name of the package to use ('prim' or 'ema_workbench').
    - preprocessing_string (List[Callable]): List of preprocessing functions.
    - number_of_repeats (int): How many times the experiment will be run with different generated data

    Returns:
    - None: Displays a heatmap plot and the precisions of the boxes.
    """

    pts = [50, 100, 200, 400, 800, 1000, 1500]  # number of points to experiment with
    atrs = []  # number of dimensions to experiment with

    for i in range(int(dimension_max / 5)):
        atrs.append(5 * (i + 1))

    res_train = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_train[:] = np.nan
    res_test = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_test[:] = np.nan
    k = 1

    x_test, y_test = generate_data(function_string, dimension_max, 2000, 'test', package)

    for n in range(len(pts)):
        for m in range(len(atrs)):
            prec_train = []
            prec_test = []
            for i in range(number_of_repeats):
                sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(len(pts) * len(atrs)
                                                                                * number_of_repeats))
                x, y = generate_data(function_string, dimension_max, pts[n], 'train', package)

                y_temp = scale_y(np.concatenate((y, y_test), axis=0), function_string)
                y = y_temp[:len(y)]
                y_test_temp = y_temp[len(y):]

                # Select random columns
                selected_columns = random.sample(list(x.columns), atrs[m])
                x = x[selected_columns]
                if preprocessing_string is not None:
                    for item in preprocessing_string:
                        x, y = item(x, y)

                x_test_temp = x_test[selected_columns]

                # hier rein für PRIM
                precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x, y, package, 'precision')
                if len(precisions) <= 0:
                    prec_train.append(0)
                else:
                    prec_train.append(precisions[len(precisions) - 1])

                if len(boxes) <= 0:
                    prec_test.append(0)
                else:
                    box = boxes[len(boxes) - 1]
                    box = pd.DataFrame(box)
                    prec_test.append(calculate_precision_test_data_onebox(box, x_test_temp.values, y_test_temp))
                k = k + 1

            res_train[n, m] = np.mean(prec_train)
            res_test[n, m] = np.mean(prec_test)

    plt.imshow(res_train - res_test, cmap='hot', vmin=0, vmax=1)
    plt.yticks(np.arange(len(pts)), pts)
    plt.xticks(np.arange(len(atrs)), atrs)
    plt.colorbar()
    plt.show()
    print(res_test)
    print(res_train)


def visualize_precision_and_recall(precision_baseline, recall_baseline, precision2, recall2, first_label,
                                   second_label):
    """
    Create a scatter plot comparing precision vs. recall for two sets of data.

    Parameters:
    - precision_baseline (List[float]): Precision values for the baseline data.
    - recall_baseline (List[float]): Recall values for the baseline data.
    - precision2 (List[float]): Precision values for the second set of data.
    - recall2 (List[float]): Recall values for the second set of data.
    - first_label (str): Label for the baseline data.
    - second_label (str): Label for the second set of data.

    Returns:
    - plt.Figure: The matplotlib Figure object.

    Notes:
    - The resulting plot can be displayed using plt.show().
    """

    precision2 = [statistics.mean(l) for l in precision2]
    precision_baseline = [statistics.mean(l) for l in precision_baseline]
    recall2 = [statistics.mean(l) for l in recall2]
    recall_baseline = [statistics.mean(l) for l in recall_baseline]

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


def visualize_precision_recall_all_boxes(number_of_repeats, package, dimension_max, preprocessing_string,
                                         data_generator, path):
    prec_test, prec_train, rec_test, rec_train, all_n_in_boxes = get_precision_and_recall_train_test(number_of_repeats,
                                                                                     data_generator,
                                                                                     package, preprocessing_string,
                                                                                     dimension_max)

    prec_test_mean = [statistics.mean(l) for l in prec_test]
    prec_train_mean = [statistics.mean(l) for l in prec_train]
    rec_test_mean = [statistics.mean(l) for l in rec_test]
    rec_train_mean = [statistics.mean(l) for l in rec_train]

    all_n_in_boxes_mean = [statistics.mean(l) for l in all_n_in_boxes]

    prec_test_std = [statistics.stdev(l) if len(l) > 1 else 0 for l in prec_test]
    prec_train_std = [statistics.stdev(l) if len(l) > 1 else 0 for l in prec_train]
    rec_test_std = [statistics.stdev(l) if len(l) > 1 else 0 for l in rec_test]
    rec_train_std = [statistics.stdev(l) if len(l) > 1 else 0 for l in rec_train]

    while len(prec_test_mean) != len(prec_train_mean):
        if len(prec_train_mean) > len(prec_test_mean):
            prec_test_mean.append(0)
        else:
            prec_train_mean.append(0)

    while len(rec_test_mean) != len(rec_train_mean):
        if len(rec_train_mean) > len(rec_test_mean):
            rec_test_mean.append(0)
        else:
            rec_train_mean.append(0)

    pd.DataFrame(np.array(prec_test_mean).T).to_csv(path + '/100_prec_test.csv')
    pd.DataFrame(np.array(rec_test_mean).T).to_csv(path + '/100_rec_test.csv')
    pd.DataFrame(np.array(prec_train_mean).T).to_csv(path + '/100_prec_train.csv')
    pd.DataFrame(np.array(rec_train_mean).T).to_csv(path + '/100_rec_train.csv')

    pd.DataFrame(np.array(prec_test_std).T).to_csv(path + '/100_prec_test_std.csv')
    pd.DataFrame(np.array(rec_test_std).T).to_csv(path + '/100_rec_test_std.csv')
    pd.DataFrame(np.array(prec_train_std).T).to_csv(path + '/100_prec_train_std.csv')
    pd.DataFrame(np.array(rec_train_std).T).to_csv(path + '/100_rec_train_std.csv')

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs = axs.flatten()

    axs[0].scatter(range(len(prec_train_mean)), prec_train_mean, c='red', marker='o', label='Train')
    axs[0].scatter(range(len(prec_test_mean)), prec_test_mean, c='blue', marker='o', label='Test')
    axs[0].fill_between(range(len(prec_train_mean)),
                        [m - s for m, s in zip(prec_train_mean, prec_train_std)],
                        [m + s for m, s in zip(prec_train_mean, prec_train_std)],
                        color='red', alpha=0.2)
    axs[0].fill_between(range(len(prec_test_mean)),
                        [m - s for m, s in zip(prec_test_mean, prec_test_std)],
                        [m + s for m, s in zip(prec_test_mean, prec_test_std)],
                        color='blue', alpha=0.2)

    # Recall plot with shaded standard deviation
    axs[1].scatter(range(len(rec_train_mean)), rec_train_mean, c='red', marker='o', label='Train')
    axs[1].scatter(range(len(rec_test_mean)), rec_test_mean, c='blue', marker='o', label='Test')
    axs[1].fill_between(range(len(rec_train_mean)),
                        [m - s for m, s in zip(rec_train_mean, rec_train_std)],
                        [m + s for m, s in zip(rec_train_mean, rec_train_std)],
                        color='red', alpha=0.2)
    axs[1].fill_between(range(len(rec_test_mean)),
                        [m - s for m, s in zip(rec_test_mean, rec_test_std)],
                        [m + s for m, s in zip(rec_test_mean, rec_test_std)],
                        color='blue', alpha=0.2)

    axs[0].set_xlim(0, len(prec_train_mean) + 0.1)
    axs[0].set_ylim(0, 1.1)
    axs[0].set_xticks(np.arange(len(all_n_in_boxes)))
    axs[0].set_xticklabels(all_n_in_boxes_mean, rotation=45)
    axs[1].set_xlim(0, len(rec_test_mean) + 0.1)
    axs[1].set_ylim(0, 1.1)
    axs[1].set_xticks(np.arange(len(all_n_in_boxes)))
    axs[1].set_xticklabels(all_n_in_boxes_mean, rotation=45)

    # Add labels and a legend
    axs[0].set_xlabel('Count Points in Train Box', fontsize=15)
    axs[0].set_ylabel('Precision', fontsize=15)
    axs[0].legend(fontsize=15)

    # Add labels and a legend
    axs[1].set_xlabel('Count Points in Train Box', fontsize=15)
    axs[1].set_ylabel('Recall', fontsize=15)
    axs[1].legend(fontsize=15)

    axs[1].grid(True)
    axs[0].grid(True)
    plt.savefig(
        path + '/100_results_zoomed.svg',
        format='svg')


def create_heatmap_best_box_generated_data_precision_kfold(function_string, dimension_max, package,
                                                           preprocessing_string, number_of_repeats):
    """
    Create a plot comparing precision results for different dataset sizes and dimensions using kfold cross validation.

    Parameters:
    - function_string (str): String representation of the function.
    - dimension_max (int): Maximum dimensionality of the generated data.
    - package (str): Name of the package to use ('prim' or 'ema_workbench').
    - preprocessing_string (List[Callable]): List of preprocessing functions.
    - number_of_repeats (int): How many times the experiment will be run with different generated data

    Returns:
    - None: Displays a heatmap plot and the precisions of the boxes.
    """

    pts = [200, 400, 800, 1600, 3200, 6400]  # number of points to experiment with
    atrs = []
    atrs.append(dimension_max)

    results = np.empty((len(pts), len(atrs)))  # matrix with the results
    results[:] = np.nan

    k = 1

    for n in range(len(pts)):
        for m in range(len(atrs)):
            prec_train = []
            prec_test = []
            rec_train = []
            rec_test = []
            for i in range(number_of_repeats):
                sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(len(pts) * len(atrs)
                                                                                * number_of_repeats * 5))
                x, y = generate_data(function_string, dimension_max, pts[n], 'test', package)
                folds = KFold(n_splits=5, shuffle=True)

                y = scale_y(y, function_string)

                selected_columns = random.sample(list(x.columns), atrs[m])
                x = x[selected_columns]

                for fold, (train_id, test_id) in enumerate(folds.split(x)):
                    x_train, x_test = x.iloc[train_id], x.iloc[test_id]
                    y_train, y_test = y[train_id], y[test_id]

                    if preprocessing_string is not None:
                        for item in preprocessing_string:
                            x_train, y_train = item(x_train, y_train)

                    precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x_train, y_train, package,
                                                                                       quality_function='precision')

                    prec_train, rec_train = add_precision_recall_of_each_box_to_list_each_box(precisions, recalls,
                                                                                              prec_train, rec_train)

                    prec_test_temp, rec_test_temp = calculate_precision_recall_test_data_allboxes(boxes, x_test, y_test)

                    prec_test, rec_test = add_precision_recall_of_each_box_to_list_each_box(prec_test_temp,
                                                                                            rec_test_temp, prec_test,
                                                                                            rec_test)
                    k = k + 1

            prec_test = [statistics.mean(l) for l in prec_test]
            prec_train = [statistics.mean(l) for l in prec_train]

            while len(prec_test) != len(prec_train):
                if len(prec_train) > len(prec_test):
                    prec_test.append(0)
                else:
                    prec_train.append(0)

            results[n, m] = np.mean(np.array(prec_train) - np.array(prec_test))

    plt.imshow(results, cmap='hot', vmin=0, vmax=1)
    plt.yticks(np.arange(len(pts)), pts)
    plt.xticks(np.arange(len(atrs)), atrs)
    plt.colorbar()
    plt.show()


def create_heatmap_best_box_generated_data_precision_kfold_last_box(function_string, dimension_max, package,
                                                                    preprocessing_string, number_of_repeats,
                                                                    synthetic_or_real, pts=None):
    """
    Create a plot comparing precision results for different dataset sizes and dimensions using kfold cross validation.

    Parameters:
    - function_string (str): String representation of the function.
    - dimension_max (int): Maximum dimensionality of the generated data.
    - package (str): Name of the package to use ('prim' or 'ema_workbench').
    - preprocessing_string (List[Callable]): List of preprocessing functions.
    - number_of_repeats (int): How many times the experiment will be run with different generated data

    Returns:
    - None: Displays a heatmap plot and the precisions of the boxes.
    """

    if pts is None:
        pts = [50, 100, 200, 400, 800, 1600, 3200, 6400]

    atrs = []  # number of dimensions to experiment with

    atrs.append(dimension_max)

    res_train = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_train[:] = np.nan
    res_test = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_test[:] = np.nan
    res_in_box = np.empty((len(pts), len(atrs)))
    res_in_box[:] = np.nan

    k = 1

    data = None

    if 'calculate' not in function_string.__name__:
        data = get_data(function_string.__name__)

    for n in range(len(pts)):
        for m in range(len(atrs)):
            prec_train = []
            prec_test = []
            #n_in_box = []
            for i in range(number_of_repeats):
                sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(len(pts) * len(atrs)
                                                                                * number_of_repeats * 5))
                x, y = generate_data(function_string, dimension_max, pts[n], 'train', package, data=data)
                folds = KFold(n_splits=5, shuffle=True)

                y = scale_y(y, function_string)

                for fold, (train_id, test_id) in enumerate(folds.split(x)):
                    x_train, x_test = x.iloc[train_id], x.iloc[test_id]
                    y_train, y_test = y[train_id], y[test_id]

                    x_train_temp, y_train_temp = x_train, y_train

                    if preprocessing_string is not None:
                        for item in preprocessing_string:
                            x_train, y_train = item(x_train, y_train)

                    # , n_box
                    precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x_train, y_train,
                                                                                                 package,
                                                                                                 quality_function=
                                                                                                 'precision')

                    # n_in_box.append(n_box)

                    if len(precisions) <= 0:
                        prec_train.append(0)
                    else:
                        ind_box = determine_box(function_string, pts[n], synthetic_or_real, boxes, x_train_temp,
                                                y_train_temp)
                        prec_train.append(precisions[ind_box])

                    if len(boxes) <= 0:
                        prec_test.append(0)
                    else:
                        ind_box = determine_box(function_string, pts[n], synthetic_or_real, boxes, x_train_temp,
                                                y_train_temp)
                        box = boxes[ind_box]
                        prec_test.append(calculate_precision_test_data_onebox(box, x_test.values, y_test))
                    k = k + 1

            res_train[n, m] = np.mean(prec_train)
            res_test[n, m] = np.mean(prec_test)
            # res_in_box[n, m] = round(np.mean(n_in_box))

    return flat_prec_rec(res_train, res_test)  # , [item for sublist in res_in_box for item in sublist]


def heatmap_all_results(number_of_repeats, package, preprocessing_string, synthetic_or_real, path):
    res_train = []
    res_test = []
    data_names_dims = []
    # res_n_in = []

    data_info = get_data_information(synthetic_or_real)

    for row in data_info.iterrows():
        temp_tr, temp_te = create_heatmap_best_box_generated_data_precision_kfold_last_box(
            row[1]['function'],
            row[1]['dim'], package,
            preprocessing_string,
            number_of_repeats,
            synthetic_or_real,
            row[1]['pts'])
        while len(temp_tr) < 8:
            temp_tr.append(0)
            temp_te.append(0)
            # temp_n_in.append(0)

        res_train.append(temp_tr)
        res_test.append(temp_te)
        # res_n_in.append(temp_n_in)

        data_names_dims.append('d=' + str(row[1]['dim']) + ', ' + row[1]['function'].__name__)
        pd.DataFrame(np.array(res_test).T).to_csv(path + '/res_test.csv')
        pd.DataFrame(np.array(res_train).T).to_csv(path + '/res_train.csv')
        # pd.DataFrame(np.array(res_n_in).T).to_csv(path + '/res_n_in.csv')

    res_test = np.array(res_test).T
    res_train = np.array(res_train).T

    colors1 = plt.cm.CMRmap_r(np.linspace(0., 0.7, 128))
    colors2 = plt.cm.cubehelix(np.linspace(0.3, 1, 128))

    # combine them and build a new colormap
    colors = np.vstack((colors2, colors1))
    my_gradient = mcolors.LinearSegmentedColormap.from_list('my_gradient', colors)

    plt.figure(figsize=(8, 9.5))

    plt.imshow(res_train - res_test, cmap=my_gradient, vmin=-1, vmax=1)

    for i in range(8):
        for j in range(7):
            if res_train[i, j] > 0:
                text = f'{res_train[i, j]:.2f}\n{res_test[i, j]:.2f}'
                plt.text(j, i, text, ha='center', va='center', fontsize=20,
                         color='white' if res_train[i, j] - res_test[i, j] > 0.65 else 'black')

    plt.yticks(np.arange(8), np.array([50, 100, 200, 400, 800, 1600, 3200, 6400]), fontsize=15)
    plt.xticks(np.arange(7), data_names_dims, rotation=45, ha="right", fontsize=15)
    plt.xlabel('data set and dimensionality', weight='bold', fontsize=20)
    plt.ylabel('number of points', weight='bold', fontsize=20)
    colorbar = plt.colorbar(fraction=0.046)
    colorbar.ax.yaxis.set_tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(path + '/all_results.svg', format='svg')
