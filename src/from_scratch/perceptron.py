"""W03. Rosenblatt perceptron.

Interface (Appendix A):
    perceptron_update(w, b, x, y, lr) -> (w, b)
        One update on one example; y is -1 or +1. If y * (w @ x + b) <= 0 then
        w <- w + lr * y * x and b <- b + lr * y, otherwise w, b are returned unchanged.
    class Perceptron(lr=1.0, max_epochs=100, seed=0)
        fit(X, y) with y in {-1, +1}; returns self. Attributes w_, b_.
        predict(X) -> array of -1 / +1.

OneVsRestPerceptron below is a multi-class extension (not part of the course contract
interface): one binary Perceptron per class, prediction by argmax over the classifiers'
continuous net-input scores rather than their signs, to resolve ties and the
all-classifiers-say-no case.
"""
import numpy as np


def perceptron_update(w, b, x, y, lr):
    if y * (np.dot(w, x) + b) <= 0:
        w = w + lr * y * x
        b = b + lr * y
    return w, b


class Perceptron:
    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr, self.max_epochs, self.seed = lr, max_epochs, seed
        self.w_ = None
        self.b_ = None
        self.n_epochs_run_ = 0
        self.converged_ = False

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w_ = np.zeros(n_features)
        self.b_ = 0.0
        self.converged_ = False

        for epoch in range(1, self.max_epochs + 1):
            updated = False
            for xi, target in zip(X, y):
                w_new, b_new = perceptron_update(self.w_, self.b_, xi, target, self.lr)
                if not (np.array_equal(w_new, self.w_) and b_new == self.b_):
                    updated = True
                self.w_, self.b_ = w_new, b_new

            self.n_epochs_run_ = epoch
            if not updated:
                self.converged_ = True
                break

        return self

    def _net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self._net_input(X) >= 0.0, 1, -1)


class OneVsRestPerceptron:
    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr, self.max_epochs, self.seed = lr, max_epochs, seed
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classifiers = {}
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            clf = Perceptron(lr=self.lr, max_epochs=self.max_epochs, seed=self.seed)
            clf.fit(X, y_binary)
            self.classifiers[c] = clf
        return self

    def predict(self, X):
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

    binary_p = Perceptron(lr=0.1, max_epochs=20)
    binary_p.fit(X, y)
    preds = binary_p.predict(X)
    print("Binary Perceptron Converged:", binary_p.converged_)
    print("Epochs run:", binary_p.n_epochs_run_)
    print("Predictions:", preds)
    assert np.array_equal(preds, y), "Binary perceptron failed AND gate test"

    X_multi = np.array([
        [1, 2], [2, 3],
        [-1, -2], [-2, -3],
        [10, -10], [11, -11]
    ])
    y_multi = np.array([0, 0, 1, 1, 2, 2])

    ovr_p = OneVsRestPerceptron(lr=0.1, max_epochs=50)
    ovr_p.fit(X_multi, y_multi)
    preds_multi = ovr_p.predict(X_multi)
    print("OneVsRest Predictions:", preds_multi)
    assert np.array_equal(preds_multi, y_multi), "OneVsRest perceptron failed multi-class test"
