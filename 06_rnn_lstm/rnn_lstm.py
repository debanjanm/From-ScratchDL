import os

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# VANILLA RNN CELL — tanh recurrence, full backprop through time.
# ============================================================
class RNNCell:
    has_params = True

    def __init__(self, input_dim, hidden_size, l2=0.0, recurrent_gain=0.5):
        # recurrent_gain<1 => contractive Whh (spectral radius ~gain), the regime where vanishing gradients bite
        self.Wxh = np.random.randn(input_dim, hidden_size) / np.sqrt(input_dim)
        self.Whh = np.random.randn(hidden_size, hidden_size) * recurrent_gain / np.sqrt(hidden_size)
        self.bh = np.zeros((1, hidden_size))
        self.l2 = l2
        self.grad_norms = []  # |dh| per timestep, chronological — shows vanishing gradient

    def forward(self, X, training=True):
        N, T, D = X.shape
        H = self.Whh.shape[0]
        h = np.zeros((N, H))
        self.cache = []
        for t in range(T):
            x_t = X[:, t, :]
            a = x_t @ self.Wxh + h @ self.Whh + self.bh
            h_new = np.tanh(a)
            self.cache.append((x_t, h, h_new))
            h = h_new
        return h  # final hidden state only — that's all the classifier reads

    def backward(self, grad_output, lr):
        T = len(self.cache)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dh = grad_output
        self.grad_norms = []
        for t in reversed(range(T)):
            x_t, h_prev, h_new = self.cache[t]
            self.grad_norms.append(np.linalg.norm(dh))
            dtanh = dh * (1 - h_new ** 2)
            dWxh += x_t.T @ dtanh
            dWhh += h_prev.T @ dtanh
            dbh += dtanh.sum(axis=0, keepdims=True)
            dh = dtanh @ self.Whh.T
        self.grad_norms.reverse()  # now chronological: t=0 .. T-1

        self.Wxh -= lr * (dWxh + self.l2 * self.Wxh)
        self.Whh -= lr * (dWhh + self.l2 * self.Whh)
        self.bh -= lr * dbh


# ============================================================
# LSTM CELL — gated recurrence with a separate cell-state highway,
# so gradients don't have to repeatedly pass through a nonlinearity
# (that highway is what fixes the vanilla RNN's vanishing gradient).
# Gate layout follows the standard combined-matrix formulation:
# a = x@Wx + h_prev@Wh + b, split into [i, f, o, g] chunks.
# ============================================================
class LSTMCell:
    has_params = True

    def __init__(self, input_dim, hidden_size, l2=0.0):
        H = hidden_size
        self.Wx = np.random.randn(input_dim, 4 * H) / np.sqrt(input_dim)
        self.Wh = np.random.randn(H, 4 * H) / np.sqrt(H)
        self.b = np.zeros((1, 4 * H))
        self.b[:, H:2 * H] = 3.0  # forget-gate bias: σ(3)≈0.95 keeps memory open; σ(0)=0.5 erases it in ~40 steps
        self.H = H
        self.l2 = l2
        self.grad_norms = []

    def forward(self, X, training=True):
        N, T, D = X.shape
        H = self.H
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        self.cache = []
        for t in range(T):
            x_t = X[:, t, :]
            a = x_t @ self.Wx + h @ self.Wh + self.b
            ai, af, ao, ag = np.split(a, 4, axis=1)
            i, f, o, g = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            self.cache.append((x_t, h, c, i, f, o, g, c_new))
            h, c = h_new, c_new
        return h

    def backward(self, grad_output, lr):
        T = len(self.cache)
        H = self.H
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dh, dc = grad_output, np.zeros((grad_output.shape[0], H))
        self.grad_norms = []
        for t in reversed(range(T)):
            x_t, h_prev, c_prev, i, f, o, g, c = self.cache[t]
            self.grad_norms.append(np.linalg.norm(dh))
            tanh_c = np.tanh(c)

            dc = dc + dh * o * (1 - tanh_c ** 2)
            do = dh * tanh_c
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dprev_c = dc * f

            dai = di * i * (1 - i)
            daf = df * f * (1 - f)
            dao = do * o * (1 - o)
            dag = dg * (1 - g ** 2)
            da = np.concatenate([dai, daf, dao, dag], axis=1)

            dWx += x_t.T @ da
            dWh += h_prev.T @ da
            db += da.sum(axis=0, keepdims=True)

            dh = da @ self.Wh.T
            dc = dprev_c
        self.grad_norms.reverse()

        self.Wx -= lr * (dWx + self.l2 * self.Wx)
        self.Wh -= lr * (dWh + self.l2 * self.Wh)
        self.b -= lr * db


# ============================================================
# Output head + loss (same shapes as prior topics)
# ============================================================
class Dense:
    has_params = True

    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))

    def forward(self, x, training=True):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, lr):
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias
        return grad_input


class Sigmoid:
    has_params = False

    def forward(self, x, training=True):
        self.output = sigmoid(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class BCELoss:
    def forward(self, y_pred, y_true):
        eps = 1e-15
        yp = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp))

    def backward(self, y_pred, y_true):
        eps = 1e-15
        yp = np.clip(y_pred, eps, 1 - eps)
        return (-(y_true / yp) + (1 - y_true) / (1 - yp)) / y_true.shape[0]


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr) if layer.has_params else layer.backward(grad)


# ============================================================
# TASK: long-delay signal memorization.
# A +1/-1 marker appears ONLY at t=0; every other step is noise.
# The label = sign of the marker. The network must carry that one
# bit of information across T timesteps of BPTT to classify it —
# the classic long-term-dependency stress test (Hochreiter 1997).
# ============================================================
def make_dataset(n_per_class, T, seed):
    rng = np.random.RandomState(seed)
    N = n_per_class * 2
    X = rng.randn(N, T, 1) * 0.1
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=float).reshape(-1, 1)
    marker = np.where(y == 1, 1.0, -1.0)
    X[:, 0, 0] = marker[:, 0]
    perm = rng.permutation(N)
    return X[perm], y[perm]


def accuracy(model, X, y):
    preds = (model.forward(X, training=False) >= 0.5).astype(int)
    return np.mean(preds == y)


def train_model(model, X, y, loss_fn, epochs, lr):
    for _ in range(epochs):
        y_pred = model.forward(X, training=True)
        grad = loss_fn.backward(y_pred, y)
        model.backward(grad, lr)


if __name__ == "__main__":
    print("=" * 60)
    print("RNN vs LSTM FROM SCRATCH WITH NUMPY")
    print("=" * 60)

    T = 40  # sequence length — long enough that vanilla RNN struggles
    X_train, y_train = make_dataset(n_per_class=40, T=T, seed=42)
    X_test, y_test = make_dataset(n_per_class=40, T=T, seed=7)
    print(f"Task: remember a marker seen at t=0, classify after {T} noisy timesteps")
    print(f"Train: {len(y_train)} sequences | Test: {len(y_test)} sequences")

    loss_fn = BCELoss()
    epochs, lr, hidden = 400, 0.1, 16

    np.random.seed(1)
    rnn = Model([RNNCell(1, hidden), Dense(hidden, 1), Sigmoid()])
    train_model(rnn, X_train, y_train, loss_fn, epochs, lr)
    rnn_train_acc = accuracy(rnn, X_train, y_train)
    rnn_test_acc = accuracy(rnn, X_test, y_test)

    np.random.seed(1)
    lstm = Model([LSTMCell(1, hidden), Dense(hidden, 1), Sigmoid()])
    train_model(lstm, X_train, y_train, loss_fn, epochs, lr)
    lstm_train_acc = accuracy(lstm, X_train, y_train)
    lstm_test_acc = accuracy(lstm, X_test, y_test)

    print(f"\n{'Model':<12}{'Train Acc':<12}{'Test Acc'}")
    print(f"{'RNN':<12}{rnn_train_acc:<12.2%}{rnn_test_acc:.2%}")
    print(f"{'LSTM':<12}{lstm_train_acc:<12.2%}{lstm_test_acc:.2%}")

    # Gradient flow measured at INITIALIZATION (post-training norms are meaningless: loss -> 0 shrinks all grads).
    np.random.seed(1)
    probes = {"RNN": RNNCell(1, hidden), "LSTM": LSTMCell(1, hidden)}
    ratios = {}
    for name, cell in probes.items():
        cell.forward(X_train)
        cell.backward(np.ones((len(y_train), hidden)), lr=0.0)  # lr=0: measure only
        ratios[name] = cell.grad_norms[0] / cell.grad_norms[-1]
    print(f"\n|dh_0| / |dh_{T-1}| at init (1.0 = no vanishing):")
    for name, r in ratios.items():
        print(f"  {name:<5}{r:.2e}")

    assert lstm_test_acc > 0.85, "LSTM failed to solve the long-delay memorization task"
    assert ratios["RNN"] < 1e-6 < 1e-3 < ratios["LSTM"], "expected RNN gradient to vanish and LSTM's to survive"

    plt.figure(figsize=(7, 4))
    plt.plot(probes["RNN"].grad_norms, label="RNN |dh_t|", linewidth=2)
    plt.plot(probes["LSTM"].grad_norms, label="LSTM |dh_t|", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Timestep t")
    plt.ylabel("Gradient norm (log scale)")
    plt.title(f"Gradient flow at init: RNN vs LSTM over {T} timesteps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "rnn_lstm_gradients.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nGradient-norm plot saved to: {out_path}")
