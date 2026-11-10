import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.framework import ReLU, Sigmoid, MSELoss  # reuse activations/loss; only Dense differs here

np.random.seed(42)


# ============================================================
# DENSE LAYER — grad computation only, no update.
# (common.framework.Dense couples grad+update; optimizers need
# the raw gradient so a pluggable update rule can be applied.)
# ============================================================
class Dense:
    def __init__(self, input_dim, output_dim, weights=None, bias=None):
        self.weights = weights.copy() if weights is not None else np.random.randn(input_dim, output_dim) * 0.5
        self.bias = bias.copy() if bias is not None else np.zeros((1, output_dim))
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input, grad_weights, grad_bias


class MLP:
    """Fixed architecture: 2 -> 8 (ReLU) -> 1 (Sigmoid), so every optimizer trains the same net."""

    def __init__(self, w1, b1, w2, b2):
        self.d1 = Dense(2, 8, w1, b1)
        self.a1 = ReLU()
        self.d2 = Dense(8, 1, w2, b2)
        self.a2 = Sigmoid()

    def forward(self, x):
        return self.a2.forward(self.d2.forward(self.a1.forward(self.d1.forward(x))))

    def backward(self, grad):
        grad = self.a2.backward(grad)
        grad, gw2, gb2 = self.d2.backward(grad)
        grad = self.a1.backward(grad)
        _, gw1, gb1 = self.d1.backward(grad)
        params = [self.d1.weights, self.d1.bias, self.d2.weights, self.d2.bias]
        grads = [gw1, gb1, gw2, gb2]
        return params, grads


# ============================================================
# OPTIMIZERS
# ============================================================
class SGDMomentum:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocities):
            v *= self.momentum
            v -= self.lr * g
            p += v


class RMSprop:
    def __init__(self, lr=0.01, decay=0.9, eps=1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = None

    def step(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
        for p, g, c in zip(params, grads, self.cache):
            c *= self.decay
            c += (1 - self.decay) * g ** 2
            p -= self.lr * g / (np.sqrt(c) + self.eps)


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# TRAINING
# ============================================================
def train(model, optimizer, X, y, loss_fn, epochs):
    history = []
    for _ in range(epochs):
        y_pred = model.forward(X)
        history.append(loss_fn.forward(y_pred, y))
        params, grads = model.backward(loss_fn.backward(y_pred, y))
        optimizer.step(params, grads)
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZERS FROM SCRATCH WITH NUMPY")
    print("=" * 60)

    # Same non-linearly-separable task as before (XOR), so ReLU hidden
    # layer is required and optimizer choice actually matters.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Shared initial weights: every optimizer starts from the same point.
    w1 = np.random.randn(2, 8) * 0.5
    b1 = np.zeros((1, 8))
    w2 = np.random.randn(8, 1) * 0.5
    b2 = np.zeros((1, 1))

    loss_fn = MSELoss()
    epochs = 2000
    optimizers = {
        "SGD+Momentum": SGDMomentum(lr=0.1, momentum=0.9),
        "RMSprop": RMSprop(lr=0.05),
        "Adam": Adam(lr=0.05),
    }

    histories = {}
    for name, opt in optimizers.items():
        model = MLP(w1, b1, w2, b2)  # fresh model, identical starting weights
        histories[name] = train(model, opt, X, y, loss_fn, epochs)
        final_loss = histories[name][-1]
        preds = (model.forward(X) >= 0.5).astype(int)
        print(f"{name:15s} final_loss={final_loss:.6f} preds={preds.ravel().tolist()} target={y.ravel().astype(int).tolist()}")
        assert final_loss < 0.05, f"{name} failed to converge on XOR"

    plt.figure(figsize=(7, 4))
    for name, hist in histories.items():
        plt.plot(hist, label=name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Optimizer Comparison on XOR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "optimizer_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nComparison plot saved to: {out_path}")

    print("\nAll optimizers converged on XOR.")
