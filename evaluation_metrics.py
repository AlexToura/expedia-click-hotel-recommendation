


class HitsAtK:
    def __init__(self, k):
        self.k = k
        self.hits = 0.0
        self.total_samples = 0

    def update_metric(self, y_pred, y_true):
        if y_true in y_pred[:self.k]:
            self.hits += 1.0
        self.total_samples += 1

    def calculate_metric(self):
        if self.total_samples > 0:
            return self.hits / self.total_samples
        else:
            return 0  # Prevent division by zero

