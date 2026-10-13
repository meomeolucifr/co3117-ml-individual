import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.class_count_ = np.array([
            np.sum(y == c) for c in self.classes_
        ])
        self.class_prior_ = self.class_count_ / len(y)

        self.theta_ = np.array([
            X[y == c].mean(axis=0) for c in self.classes_
        ])
        self.var_ = np.array([
            X[y == c].var(axis=0, ddof=0)
            for c in self.classes_
        ])

        epsilon = self.var_smoothing * np.var(X, axis=0, ddof=0).max()
        self.var_ += epsilon

        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)

        joint = []
        for k in range(len(self.classes_)):
            var = self.var_[k]
            mean = self.theta_[k]

            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var)
                + (X - mean) ** 2 / var,
                axis=1
            )
            joint.append(np.log(self.class_prior_[k]) + log_likelihood)

        joint = np.column_stack(joint)

        max_log = joint.max(axis=1, keepdims=True)
        log_norm = max_log + np.log(
            np.exp(joint - max_log).sum(axis=1, keepdims=True)
        )

        return joint - log_norm

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.class_count_ = np.array([
            np.sum(y == c) for c in self.classes_
        ])
        self.class_prior_ = self.class_count_ / n_samples

        feature_count = np.array([
            X[y == c].sum(axis=0) for c in self.classes_
        ])

        smoothed_count = feature_count + self.alpha
        smoothed_total = smoothed_count.sum(axis=1, keepdims=True)

        self.feature_log_prob_ = (
            np.log(smoothed_count) - np.log(smoothed_total)
        )

        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X has an incorrect number of features")
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")

        joint = (
            X @ self.feature_log_prob_.T
            + np.log(self.class_prior_)[None, :]
        )

        max_log = joint.max(axis=1, keepdims=True)
        log_norm = max_log + np.log(
            np.exp(joint - max_log).sum(axis=1, keepdims=True)
        )

        return joint - log_norm

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
