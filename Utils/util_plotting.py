import numpy as np

from Utils.util_functions import *
import sys
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns


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
                                         data_generator):

    prec_test, prec_train, rec_test, rec_train = get_precision_and_recall_train_test(number_of_repeats,
                                                                                     data_generator,
                                                                                     package, preprocessing_string,
                                                                                     dimension_max)

    prec_test = [statistics.mean(l) for l in prec_test]
    prec_train = [statistics.mean(l) for l in prec_train]
    rec_test = [statistics.mean(l) for l in rec_test]
    rec_train = [statistics.mean(l) for l in rec_train]

    while len(prec_test) != len(prec_train):
        if len(prec_train) > len(prec_test):
            prec_test.append(0)
        else:
            prec_train.append(0)

    while len(rec_test) != len(rec_train):
        if len(rec_train) > len(rec_test):
            rec_test.append(0)
        else:
            rec_train.append(0)

    #plt.figure(figsize=(8, 6))
    #fig_prec, ax_prec = plt.subplots()
    #ax_prec.boxplot(prec_train, boxprops={'color': 'red', 'facecolor': None}, medianprops={'color': 'red', 'linewidth': 3}, capprops={'color': 'red'},
                #flierprops={'color': 'red'}, whiskerprops={'color': 'red'}, patch_artist=True, showfliers=False)
    #ax_prec.boxplot(prec_test, boxprops={'color': 'yellow', 'facecolor': None}, medianprops={'color': 'yellow', 'linewidth': 3}, capprops={'color': 'yellow'},
                #flierprops={'color': 'yellow'}, whiskerprops={'color': 'yellow'}, patch_artist=True, showfliers=False)

    #fig_rec, ax_rec = plt.subplots()
    #ax_rec.boxplot(rec_train, boxprops={'color': 'red', 'facecolor': None},
                    #medianprops={'color': 'red', 'linewidth': 3}, capprops={'color': 'red'},
                    #flierprops={'color': 'red'}, whiskerprops={'color': 'red'}, patch_artist=True, showfliers=False)
    #ax_rec.boxplot(rec_test, boxprops={'color': 'yellow', 'facecolor': None},
                    #medianprops={'color': 'yellow', 'linewidth': 3}, capprops={'color': 'yellow'},
                    #flierprops={'color': 'yellow'}, whiskerprops={'color': 'yellow'}, patch_artist=True,
                    #showfliers=False)

    diff_prec = np.array(prec_train) - np.array(prec_test)

    plt.figure(figsize=(8, 6))
    fig_prec, ax_prec = plt.subplots()
    ax_prec.scatter(range(len(prec_train)), prec_train, c='red', marker='o', label='Precision Train')

    ax_prec.scatter(range(len(prec_test)), prec_test, c='yellow', marker='o', label='Precision Test')

    ax_prec.scatter(range(len(diff_prec)), diff_prec, c='blue', marker='o', label='Gap')

    diff_rec = np.array(rec_train) - np.array(rec_test)
    fig_rec, ax_rec = plt.subplots()
    ax_rec.scatter(range(len(rec_train)), rec_train, c='red', marker='o', label='Precision Train')

    ax_rec.scatter(range(len(rec_test)), rec_test, c='yellow', marker='o', label='Precision Test')

    ax_rec.scatter(range(len(diff_rec)), diff_rec, c='blue', marker='o', label='Gap')

    ax_prec.set_xlim(0, len(diff_rec) + 0.1)
    ax_prec.set_ylim(0, 1.1)

    # Add labels and a legend
    ax_prec.set_xlabel('Number of Box')
    ax_prec.set_ylabel('Precision')
    ax_prec.set_title('Gap Precision all Boxes')
    ax_prec.legend()

    ax_rec.set_xlim(0, len(diff_prec) + 0.1)
    ax_rec.set_ylim(0, 1.1)

    # Add labels and a legend
    ax_rec.set_xlabel('Number of Box')
    ax_rec.set_ylabel('Recall')
    ax_rec.set_title('Gap Recall all Boxes')
    ax_rec.legend()

    ax_rec.grid(True)
    ax_prec.grid(True)
    plt.show()


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

    pts = [50, 100, 200, 400, 800]#, 1600, 3200, 6400]  # number of points to experiment with
    atrs = []  # number of dimensions to experiment with

    atrs.append(dimension_max)

    res_train = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_train[:] = np.nan
    res_test = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_test[:] = np.nan

    k = 1

    data = None

    if 'calculate' not in function_string.__name__:
        data = get_data(function_string.__name__)

    for n in range(len(pts)):
        for m in range(len(atrs)):
            prec_train = []
            prec_test = []
            for i in range(number_of_repeats):
                sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(len(pts) * len(atrs)
                                                                                * number_of_repeats * 5))
                x, y = generate_data(function_string, dimension_max, pts[n], 'train', package, data=data)
                folds = KFold(n_splits=5, shuffle=True)

                y = scale_y(y, function_string)

                for fold, (train_id, test_id) in enumerate(folds.split(x)):
                    x_train, x_test = x.iloc[train_id], x.iloc[test_id]
                    y_train, y_test = y[train_id], y[test_id]

                    if preprocessing_string is not None:
                        for item in preprocessing_string:
                            x_train, y_train = item(x_train, y_train)

                    precisions, recalls, boxes = get_list_all_precisions_recalls_boxes(x_train, y_train, package,
                                                                                       quality_function='precision')

                    if len(precisions) <= 0:
                        prec_train.append(0)
                    else:
                        prec_train.append(precisions[len(precisions) - 1])

                    if len(boxes) <= 0:
                        prec_test.append(0)
                    else:
                        box = boxes[len(boxes) - 1]
                        box = pd.DataFrame(box)
                        prec_test.append(calculate_precision_test_data_onebox(box, x_test.values, y_test))
                    k = k + 1

            res_train[n, m] = np.mean(prec_train)
            res_test[n, m] = np.mean(prec_test)

    plt.imshow(abs(res_train - res_test), cmap='CMRmap', vmin=0, vmax=1)

    for i in range(len(pts)):
        for j in range(len(atrs)):
            text = f'{res_train[i, j]:.2f}\n{res_test[i, j]:.2f}'
            plt.text(j, i, text, ha='center', va='center',
                     color='white' if res_train[i, j] - res_test[i, j] < 0.35 else 'black', fontweight='bold')

    plt.yticks(np.arange(len(pts)), pts)
    plt.xticks(np.arange(len(atrs)), atrs)
    plt.colorbar()
    plt.show()
    print(res_test)
    print(res_train)

