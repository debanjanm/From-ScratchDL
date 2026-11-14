import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.gradcheck import grad_check


# ============================================================
# LAYERS — each stores its own grads in preallocated arrays
# (filled in place on backward) so one optimizer can hold
# (param, grad) pairs for the whole model.
# ============================================================
class Linear:
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out) / np.sqrt(d_in)
        self.b = np.zeros(d_out)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        x2, d2 = self.x.reshape(-1, self.W.shape[0]), dout.reshape(-1, self.W.shape[1])
        self.dW[...] = x2.T @ d2
        self.db[...] = d2.sum(axis=0)
        return (d2 @ self.W.T).reshape(self.x.shape)

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class Embedding:
    def __init__(self, vocab, d):
        self.W = np.random.randn(vocab, d)
        self.dW = np.zeros_like(self.W)

    def forward(self, tokens):
        self.tokens = tokens
        return self.W[tokens]

    def backward(self, dout):
        self.dW[...] = 0
        np.add.at(self.dW, self.tokens, dout)

    def parameters(self):
        return [(self.W, self.dW)]


class LayerNorm:
    """Normalize each token's feature vector (last axis) to zero mean / unit variance."""

    def __init__(self, d, eps=1e-5):
        self.g, self.b, self.eps = np.ones(d), np.zeros(d), eps
        self.dg, self.db = np.zeros(d), np.zeros(d)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.std_inv
        return self.g * self.xhat + self.b

    def backward(self, dout):
        axes = tuple(range(dout.ndim - 1))
        self.dg[...] = (dout * self.xhat).sum(axis=axes)
        self.db[...] = dout.sum(axis=axes)
        dxhat = dout * self.g
        D = dout.shape[-1]
        return self.std_inv / D * (
            D * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).sum(axis=-1, keepdims=True)
        )

    def parameters(self):
        return [(self.g, self.dg), (self.b, self.db)]


def positional_encoding(L, d):
    pos = np.arange(L)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / np.power(10000.0, i / d)
    pe = np.zeros((L, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ============================================================
# MULTI-HEAD SELF-ATTENTION
#   scores = Q K^T / sqrt(d_k);  A = softmax(scores);  out = A V
# ============================================================
class MultiHeadAttention:
    def __init__(self, d, heads):
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = Linear(d, d), Linear(d, d), Linear(d, d), Linear(d, d)

    def split(self, x):  # (N,L,d) -> (N,h,L,dk)
        N, L, _ = x.shape
        return x.reshape(N, L, self.h, self.dk).transpose(0, 2, 1, 3)

    def merge(self, x):  # (N,h,L,dk) -> (N,L,d)
        N, _, L, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, L, self.h * self.dk)

    def forward(self, x):
        self.Q, self.K, self.V = self.split(self.q.forward(x)), self.split(self.k.forward(x)), self.split(self.v.forward(x))
        scores = self.Q @ self.K.swapaxes(-1, -2) / np.sqrt(self.dk)
        scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
        e = np.exp(scores)
        self.A = e / e.sum(axis=-1, keepdims=True)  # (N,h,L,L): row i = how much token i attends to each token
        return self.o.forward(self.merge(self.A @ self.V))

    def backward(self, dout):
        dctx = self.split(self.o.backward(dout))
        dA = dctx @ self.V.swapaxes(-1, -2)
        dV = self.A.swapaxes(-1, -2) @ dctx
        dscores = self.A * (dA - (dA * self.A).sum(axis=-1, keepdims=True)) / np.sqrt(self.dk)  # softmax backward
        dQ = dscores @ self.K
        dK = dscores.swapaxes(-1, -2) @ self.Q
        return self.q.backward(self.merge(dQ)) + self.k.backward(self.merge(dK)) + self.v.backward(self.merge(dV))

    def parameters(self):
        return sum((m.parameters() for m in (self.q, self.k, self.v, self.o)), [])


class EncoderBlock:
    """Pre-LN transformer encoder block: x + MHA(LN(x)), then h + FFN(LN(h))."""

    def __init__(self, d, heads, d_ff):
        self.ln1, self.mha = LayerNorm(d), MultiHeadAttention(d, heads)
        self.ln2, self.ff1, self.ff2 = LayerNorm(d), Linear(d, d_ff), Linear(d_ff, d)

    def forward(self, x):
        h = x + self.mha.forward(self.ln1.forward(x))
        a = self.ff1.forward(self.ln2.forward(h))
        self.relu_mask = a > 0
        return h + self.ff2.forward(a * self.relu_mask)

    def backward(self, dout):
        dff = self.ff2.backward(dout) * self.relu_mask
        dh = dout + self.ln2.backward(self.ff1.backward(dff))
        return dh + self.ln1.backward(self.mha.backward(dh))

    def parameters(self):
        return sum((m.parameters() for m in (self.ln1, self.mha, self.ln2, self.ff1, self.ff2)), [])


# ============================================================
# MODEL: embed (+ positional enc.) -> N encoder blocks -> LN -> mean-pool -> Linear -> sigmoid
# ============================================================
class TransformerClassifier:
    def __init__(self, vocab, max_len, d=32, heads=4, d_ff=64, n_blocks=2, use_pos=True):
        self.emb = Embedding(vocab, d)
        self.pe = positional_encoding(max_len, d) if use_pos else 0.0
        self.blocks = [EncoderBlock(d, heads, d_ff) for _ in range(n_blocks)]
        self.ln_f = LayerNorm(d)
        self.head = Linear(d, 1)

    def forward(self, tokens):
        x = self.emb.forward(tokens) + self.pe
        for blk in self.blocks:
            x = blk.forward(x)
        self.L = x.shape[1]
        pooled = self.ln_f.forward(x).mean(axis=1)
        self.p = 1 / (1 + np.exp(-self.head.forward(pooled)))
        return self.p

    def loss(self, y):
        p = np.clip(self.p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def backward(self, y):
        dlogit = (self.p - y) / len(y)  # sigmoid + BCE combined
        dpooled = self.head.backward(dlogit)
        dx = self.ln_f.backward(np.repeat(dpooled[:, None, :] / self.L, self.L, axis=1))
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.emb.backward(dx)

    def parameters(self):
        layers = [self.emb, *self.blocks, self.ln_f, self.head]
        return sum((m.parameters() for m in layers), [])


class Adam:
    def __init__(self, pairs, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.pairs, self.lr, self.b1, self.b2, self.eps, self.t = pairs, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p, _ in pairs]
        self.v = [np.zeros_like(p) for p, _ in pairs]

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.pairs):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# GRADIENT CHECK — compare analytic backward to finite differences
# on a tiny model, a few random entries of every parameter array.
# ============================================================
def run_grad_check(seed=0):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    model = TransformerClassifier(vocab=6, max_len=5, d=8, heads=2, d_ff=16, n_blocks=2)
    tokens = rng.randint(0, 6, (3, 5))
    y = np.array([[1.0], [0.0], [1.0]])
    model.forward(tokens)
    model.backward(y)

    def loss():
        model.forward(tokens)
        return model.loss(y)

    return grad_check(loss, model.parameters())


# ============================================================
# TASK: label = 1 iff the FIRST token appears again later.
# Needs content lookup (attention) AND knowing which token is
# first (positional encoding) — mean-pooling alone can't tell.
# ============================================================
def make_dataset(n, L, V, seed):
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, V, (n, L))
    y = rng.rand(n) < 0.5
    tokens[~y, 1:] = ((tokens[:, :1] + rng.randint(1, V, (n, L - 1))) % V)[~y]  # never equals first token
    idx = np.where(y)[0]
    tokens[idx, rng.randint(1, L, len(idx))] = tokens[idx, 0]  # plant a copy of the first token
    return tokens, y.astype(float).reshape(-1, 1)


def accuracy(model, tokens, y):
    return np.mean((model.forward(tokens) >= 0.5) == y)


def train(model, X, y, epochs, batch=64, lr=3e-3, seed=0):
    opt = Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            model.forward(X[b])
            model.backward(y[b])
            opt.step()


if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFORMER ENCODER FROM SCRATCH WITH NUMPY")
    print("=" * 60)

    err = run_grad_check()
    print(f"Gradient check (analytic vs numeric, worst relative error): {err:.2e}")
    assert err < 1e-4, "backward pass disagrees with finite differences"

    L, V = 8, 10
    X_train, y_train = make_dataset(2000, L, V, seed=42)
    X_test, y_test = make_dataset(500, L, V, seed=7)
    print(f"\nTask: does the first token reappear later? (L={L}, vocab={V}, train={len(y_train)}, test={len(y_test)})")

    results = {}
    for name, use_pos in [("with positional enc.", True), ("without positional enc.", False)]:
        np.random.seed(1)
        model = TransformerClassifier(V, L, use_pos=use_pos)
        train(model, X_train, y_train, epochs=30)
        results[name] = (accuracy(model, X_train, y_train), accuracy(model, X_test, y_test), model)

    print(f"\n{'Model':<26}{'Train Acc':<12}{'Test Acc'}")
    for name, (tr, te, _) in results.items():
        print(f"{name:<26}{tr:<12.2%}{te:.2%}")

    with_pe, no_pe = results["with positional enc."], results["without positional enc."]
    assert with_pe[1] > 0.9, "transformer failed to learn the task"
    assert with_pe[1] > no_pe[1] + 0.15, "positional encoding did not help — order information expected to matter"

    # Attention maps (block 0) for a positive example: row 0 = where the first token looks.
    model = with_pe[2]
    sample = X_test[np.where(y_test[:, 0] == 1)[0][0]][None, :]
    model.forward(sample)
    A = model.blocks[0].mha.A[0]
    fig, axes = plt.subplots(1, A.shape[0], figsize=(3.2 * A.shape[0], 3.4))
    for h, ax in enumerate(axes):
        ax.imshow(A[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"head {h}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
    fig.suptitle(f"Block-0 attention, tokens={sample[0].tolist()}")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_maps.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nAttention maps saved to: {out_path}")
