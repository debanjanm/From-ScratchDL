import numpy as np

np.random.seed(42)


class Perceptron:
    """Rosenblatt perceptron: single neuron, step activation, mistake-driven update."""

    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.lr = learning_rate

    def step(self, z):
        return (z >= 0).astype(int)

    def predict(self, X):
        return self.step(np.dot(X, self.weights) + self.bias)

    def fit(self, X, y, epochs=20):
        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.step(np.dot(xi, self.weights) + self.bias)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if errors == 0:
                break
        return self


def train_gate(name, y, epochs=20):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    p = Perceptron(input_size=2).fit(X, y, epochs=epochs)
    preds = p.predict(X)
    print(f"{name}: weights={p.weights.round(2)} bias={p.bias:.2f} preds={preds.tolist()} target={y.tolist()}")
    return p, preds


if __name__ == "__main__":
    print("=" * 60)
    print("PERCEPTRON FROM SCRATCH WITH NUMPY")
    print("=" * 60)

    # Linearly separable gates: perceptron converges exactly.
    _, and_preds = train_gate("AND", np.array([0, 0, 0, 1]))
    assert np.array_equal(and_preds, [0, 0, 0, 1]), "perceptron failed to learn AND"

    _, or_preds = train_gate("OR", np.array([0, 1, 1, 1]))
    assert np.array_equal(or_preds, [0, 1, 1, 1]), "perceptron failed to learn OR"

    # XOR is NOT linearly separable: a single perceptron provably cannot
    # solve it (Minsky & Papert, 1969). This motivates 02_backpropagation,
    # which adds a hidden layer to learn non-linear boundaries.
    _, xor_preds = train_gate("XOR", np.array([0, 1, 1, 0]), epochs=50)
    print(f"XOR accuracy: {np.mean(xor_preds == [0, 1, 1, 0]):.0%} (expected <100% — single-layer limit)")

    print("\nAll linearly-separable gates learned correctly.")
