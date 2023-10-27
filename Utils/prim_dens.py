import numpy as np

class PRIMdens:
    def __init__(self, X, y, alpha=0.05):
        self.alpha = alpha
        self.boxes_ = []  # To store intermediate boxes
        self.precisions_ = []  # To store precisions of intermediate boxes
        self.X_ = X
        self.y_ = y

    def fit(self):
        # Initial box is the unit box
        box = np.vstack((np.zeros(self.X_.shape[1]), np.ones(self.X_.shape[1])))
        n_points = self.X_.shape[0]
        
        for iteration in range(120):
            max_precision = -np.inf
            best_cut = None
            best_box = None
            n_in_best_box = n_points  # Initially, all points are inside the box
            ind_in_best_box = None # Initially, all points are inside the box

            for dim in range(self.X_.shape[1]):
                for direction in [0, 1]:  # 0 for lower bound, 1 for upper bound
                    trial_box = box.copy()
                    current_dim_length = trial_box[1, dim] - trial_box[0, dim]
                    
                    # Adjust the boundary in the dimension such that the resulting box's volume is (1-alpha) times the previous box
                    adjustment = current_dim_length * self.alpha
                    if direction == 0:
                        trial_box[0, dim] += adjustment
                        in_box = self.X_[:, dim] >= trial_box[0, dim]
                    else:
                        trial_box[1, dim] -= adjustment
                        in_box = self.X_[:, dim] <= trial_box[1, dim]
                    
                    # Count points in the trial box
                    n_in_box = np.count_nonzero(in_box)

                    # Calculate volume of the trial box
                    precision = np.count_nonzero(self.y_[in_box]) / n_in_box if n_in_box > 0 else 0

                    if precision > max_precision:
                        max_precision = precision
                        best_cut = (dim, direction)
                        best_box = trial_box
                        n_in_best_box = n_in_box  # Update the count for best box
                        ind_in_best_box = in_box

            # If we can't find a cut that improves precision or the box contains less than 1/alpha points, break
            if best_cut is None or n_in_best_box < 1/self.alpha:
                break

            # Update the box and record it
            box = best_box
            self.boxes_.append(box)
            self.precisions_.append(max_precision)
            self.X_ = self.X_[ind_in_best_box]
            self.y_ = self.y_[ind_in_best_box]

        return self

    def get_boxes(self):
        return self.boxes_

    def get_precisions(self):
        return self.precisions_
