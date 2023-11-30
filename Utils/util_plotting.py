from Utils.util_functions import *
import sys
import random
from matplotlib import pyplot as plt


def create_heatmap_generated_data_precision(function_string, dimension_max, package, preprocessing_string,
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

    pts = [50, 100, 200, 400, 800]  # number of points to experiment with
    atrs = []  # number of dimensions to experiment with

    for i in range(int(dimension_max / 5)):
        atrs.append(5 * (i + 1))

    res_train = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_train[:] = np.nan
    res_test = np.empty((len(pts), len(atrs)))  # matrix with the results
    res_test[:] = np.nan
    k = 1

    x_test, y_test = generate_data(function_string, dimension_max, 2000)

    for n in range(len(pts)):
        for m in range(len(atrs)):
            prec_train = []
            prec_test = []
            for i in range(number_of_repeats):
                sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(len(pts) * len(atrs)
                                                                                * number_of_repeats))
                x, y = generate_data(function_string, dimension_max, pts[n])

                # Select random columns
                selected_columns = random.sample(list(x.columns), atrs[m])
                x = x[selected_columns]
                if preprocessing_string is not None:
                    for item in preprocessing_string:
                        x, y = item(x, y)

                x_test_temp = x_test[selected_columns]

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
                    prec_test.append(calculate_precision_test_data_onebox(box, x_test_temp.values, y_test))
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
