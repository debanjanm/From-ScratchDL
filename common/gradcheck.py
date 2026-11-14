import numpy as np


def grad_check(loss_fn, pairs, samples_per_param=3, eps=1e-5, seed=0):
    """Compare analytic grads to central finite differences; return worst error.

    loss_fn: () -> float. Must do a fresh forward pass on every call.
    pairs:   [(param, grad)] with grads already filled by one forward+backward.
    Error is |num - ana| / (|num| + |ana| + 1e-6); the floor stops params whose
    true gradient is 0 (e.g. attention key bias) from reporting float noise.
    """
    rng = np.random.RandomState(seed)
    worst = 0.0
    for p, g in pairs:
        for _ in range(samples_per_param):
            idx = tuple(rng.randint(0, s) for s in p.shape)
            orig = p[idx]
            p[idx] = orig + eps
            lp = loss_fn()
            p[idx] = orig - eps
            lm = loss_fn()
            p[idx] = orig
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx]) + 1e-6))
    return worst


if __name__ == "__main__":
    # Self-check: f(W) = sum((x@W)^2); df/dW = 2 x^T (x@W)
    rng = np.random.RandomState(1)
    x, W = rng.randn(4, 3), rng.randn(3, 2)
    dW = 2 * x.T @ (x @ W)
    assert grad_check(lambda: np.sum((x @ W) ** 2), [(W, dW)]) < 1e-6, "correct grad should pass"
    assert grad_check(lambda: np.sum((x @ W) ** 2), [(W, dW * 1.01)]) > 1e-3, "wrong grad should fail"
    print("gradcheck self-test passed")
