import numpy as np
import random


class PRIMdens:
    def __init__(self, X, y, alpha=0.05, quality_measurement='precision'):
        self.alpha = alpha
        self.boxes_ = []  # To store intermediate boxes
        self.precisions_ = []  # To store precisions of intermediate boxes
        self.recalls_ = []  # To store recall of intermediate boxes
        self.qualities_ = []
        self.X_ = X
        self.y_ = y
        self.in_box_ = []
        self.n_in_box_ = 0
        self.n_points = self.X_.shape[0]
        self.num_ones = np.count_nonzero(self.y_)
        if callable(getattr(self, 'calculate_' + quality_measurement + '_')):
            self.quality_measurement = getattr(self, 'calculate_' + quality_measurement + '_')
        else:
            raise ValueError('Quality measurement is not callable')

    def fit(self):
        # Initial box is the unit box
        box = np.vstack((np.zeros(self.X_.shape[1]), np.ones(self.X_.shape[1])))
        n_in_best_box = self.n_points  # Initially, all points are inside the box
        max_quality = -np.inf

        while max_quality < 1:
            new_precision = -np.inf
            new_recall = -np.inf
            temp_quality = 0
            best_cut = None
            best_box = None
            ind_in_best_box = None  # Initially, all points are inside the box

            for dim in range(self.X_.shape[1]):
                for direction in [0, 1]:  # 0 for lower bound, 1 for upper bound
                    trial_box = box.copy()
                    current_dim_length = trial_box[1, dim] - trial_box[0, dim]

                    # Adjust the boundary in the dimension such that the resulting box's volume is (1-alpha) times the
                    # previous box
                    adjustment = current_dim_length * self.alpha
                    if direction == 0:
                        trial_box[0, dim] += adjustment
                        self.in_box_ = self.X_[:, dim] >= trial_box[0, dim]
                    else:
                        trial_box[1, dim] -= adjustment
                        self.in_box_ = self.X_[:, dim] <= trial_box[1, dim]

                    # Count points in the trial box
                    self.n_in_box_ = np.count_nonzero(self.in_box_)

                    quality = self.calculate_quality_()

                    if quality > (1 + self.alpha / 2) * max_quality and quality > (1 + self.alpha / 2) * temp_quality\
                            and self.n_in_box_ < n_in_best_box:
                        new_precision = self.calculate_precision_()
                        new_recall = self.calculate_recall_()
                        temp_quality = quality
                        best_cut = (dim, direction)
                        best_box = trial_box
                        n_in_best_box = self.n_in_box_  # Update the count for best box
                        ind_in_best_box = self.in_box_

            # If we can't find a cut that improves precision or the box contains less than 1/alpha points, break
            if best_cut is None or (n_in_best_box / self.n_points) < self.alpha:
                break

            if temp_quality > max_quality:
                max_quality = temp_quality

            # Update the box and record it
            box = best_box
            self.boxes_.append(box)
            self.precisions_.append(new_precision)
            self.recalls_.append(new_recall)
            self.qualities_.append(temp_quality)
            self.X_ = self.X_[ind_in_best_box]
            self.y_ = self.y_[ind_in_best_box]

        return self

    def get_boxes(self):
        return self.boxes_

    def get_precisions(self):
        return self.precisions_

    def get_recalls(self):
        return self.recalls_

    def calculate_precision_(self):
        return np.count_nonzero(self.y_[self.in_box_]) / self.n_in_box_ if self.n_in_box_ > 0 else 0

    def calculate_quality_(self):
        return self.quality_measurement()

    def calculate_recall_(self):
        return np.count_nonzero(self.y_[self.in_box_]) / np.count_nonzero(self.y_) if np.count_nonzero(self.y_) > 0 \
            else 0

    def calculate_f1score_(self):
        beta = 0.15
        return (((1 + beta * beta) * self.calculate_precision_() * self.calculate_recall_())
                / (beta * beta * self.calculate_precision_() + self.calculate_recall_())) \
            if (beta * beta * self.calculate_precision_() + self.calculate_recall_() > 0) else 0

    def calculate_accuracy_(self):
        return ((np.count_nonzero(self.y_[self.in_box_]) + np.sum(np.logical_and(self.y_ == 0, self.in_box_ == 0)))
                / self.n_points)