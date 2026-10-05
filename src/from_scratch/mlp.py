"""W04. Multilayer perceptron with an explicit forward and backward pass.

Interface (Appendix A). params is a dict {"W1", "b1", ..., "WL", "bL"}; W_l has shape
(n_in, n_out). The output layer is softmax; the loss is mean cross-entropy. The hidden
activation is your choice (state it in MODEL_LOG.md).
    init_params(sizes, seed) -> params       sizes = [d, h1, ..., K]
    forward(params, X) -> (P, cache)          P has shape (n, K), rows sum to 1
    loss(params, X, Y) -> float               Y is one-hot (n, K); mean cross-entropy
    backward(params, cache, Y) -> grads       same keys and shapes as params, gradient of loss
"""
import numpy as np


def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    params = {}

    for l in range(1, len(sizes)):
        n_in = sizes[l - 1]
        n_out = sizes[l]

        params[f"W{l}"] = rng.normal(
            0,
            0.1,
            size=(n_in, n_out)
        )
        params[f"b{l}"] = np.zeros(n_out)

    return params


def forward(params, X):
    A = X
    cache = {"A0": X}

    L = len(params) // 2

    for l in range(1, L + 1):
        W = params[f"W{l}"]
        b = params[f"b{l}"]

        Z = A @ W + b
        cache[f"Z{l}"] = Z

        if l < L:
            A = np.tanh(Z)
        else:
            Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
            exp_Z = np.exp(Z_shifted)
            A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        cache[f"A{l}"] = A

    return A, cache


def loss(params, X, Y):
    P, _ = forward(params, X)

    P = np.clip(P, 1e-12, 1.0)

    return -np.mean(np.sum(Y * np.log(P), axis=1))


def backward(params, cache, Y):
    grads = {}

    L = len(params) // 2
    n = Y.shape[0]

    dZ = (cache[f"A{L}"] - Y) / n

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l - 1}"]
        W = params[f"W{l}"]

        grads[f"W{l}"] = A_prev.T @ dZ
        grads[f"b{l}"] = np.sum(dZ, axis=0)

        if l > 1:
            dA_prev = dZ @ W.T

            A_prev_hidden = cache[f"A{l - 1}"]
            dZ = dA_prev * (1 - A_prev_hidden ** 2)

    return grads
