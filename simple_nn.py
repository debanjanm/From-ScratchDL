import numpy as np

# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================
class Sigmoid:
    def __init__(self):
        self.output = None
    
    def forward(self, z):
        self.output = 1 / (1 + np.exp(-z))
        return self.output
    
    def backward(self, grad_output):
        # Gradient: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        return grad_output * self.output * (1 - self.output)

class ReLU:
    def __init__(self):
        self.input = None
    
    def forward(self, z):
        self.input = z
        return np.maximum(0, z)
    
    def backward(self, grad_output):
        # Gradient: relu'(z) = 1 if z > 0 else 0
        return grad_output * (self.input > 0).astype(float)

# ============================================================
# LOSS FUNCTIONS
# ============================================================
class MSELoss:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]

# ============================================================
# LAYER IMPLEMENTATIONS
# ============================================================
class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.input = None
        self.has_params = True  # Flag to identify trainable layers
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output, learning_rate):
        # Compute gradients
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

# ============================================================
# NEURAL NETWORK MODEL
# ============================================================
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            if hasattr(layer, 'has_params') and layer.has_params:
                # Layer with parameters (Dense)
                grad = layer.backward(grad, lr)
            else:
                # Activation function (no parameters)
                grad = layer.backward(grad)

# ============================================================
# TRAINING UTILITIES
# ============================================================
def train(model, X, y, epochs, lr, loss_fn):
    """Training loop"""
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, y)
        
        # Backward pass
        grad = loss_fn.backward(predictions, y)
        model.backward(grad, lr)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Load and preprocess dataset"""
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    """Main execution function"""
    np.random.seed(42)
    
    # Load data
    X_train, y_train = load_data()
    
    # Build model
    model = NeuralNetwork([
        Dense(2, 64),
        ReLU(),
        Dense(64, 1),
        Sigmoid()
    ])
    
    # Train model
    loss_fn = MSELoss()
    train(model, X_train, y_train, epochs=1000, lr=0.01, loss_fn=loss_fn)
    
    # Evaluate
    predictions = model.forward(X_train)
    accuracy = np.mean((predictions > 0.5) == y_train)
    print(f"\nFinal Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
