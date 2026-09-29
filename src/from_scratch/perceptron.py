import numpy as np


class Perceptron:
    def __init__(self, learning_rate: float = 1.0, n_epochs: int = 100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None
        self.n_epochs_run = 0
        self.converged = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.converged = False
        self.n_epochs_run = 0

        for epoch in range(1, self.n_epochs + 1):
            updated = False
            for xi, target in zip(X, y):
                net_input = np.dot(xi, self.w) + self.b
                prediction = 1 if net_input >= 0.0 else -1
                
                if prediction != target:
                    update = self.learning_rate * target
                    self.w += update * xi
                    self.b += update
                    updated = True

            self.n_epochs_run = epoch
            if not updated:
                self.converged = True
                break

        return self

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return sign(net_input(X)) as {-1, +1}."""
        return np.where(self._net_input(X) >= 0.0, 1, -1)


class OneVsRestPerceptron:
    def __init__(self, learning_rate: float = 1.0, n_epochs: int = 100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.classifiers = {}
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestPerceptron":
        self.classes = np.unique(y)
        self.classifiers = {}

        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            clf = Perceptron(learning_rate=self.learning_rate, n_epochs=self.n_epochs)
            clf.fit(X, y_binary)
            self.classifiers[c] = clf

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([self.classifiers[c]._net_input(X) for c in self.classes])
        return self.classes[np.argmax(scores, axis=1)]


if __name__ == "__main__":
    X = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ])
    y = np.array([1, -1, -1, -1])

    binary_p = Perceptron(learning_rate=0.1, n_epochs=20)
    binary_p.fit(X, y)
    preds = binary_p.predict(X)
    print("Binary Perceptron Converged:", binary_p.converged)
    print("Epochs run:", binary_p.n_epochs_run)
    print("Predictions:", preds)
    assert np.array_equal(preds, y), "Binary perceptron failed AND gate test"

    X_multi = np.array([
        [1, 2], [2, 3],
        [-1, -2], [-2, -3],
        [10, -10], [11, -11]
    ])
    y_multi = np.array([0, 0, 1, 1, 2, 2])

    ovr_p = OneVsRestPerceptron(learning_rate=0.1, n_epochs=50)
    ovr_p.fit(X_multi, y_multi)
    preds_multi = ovr_p.predict(X_multi)
    print("OneVsRest Predictions:", preds_multi)
    assert np.array_equal(preds_multi, y_multi), "OneVsRest perceptron failed multi-class test"