import os

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    """
    A simple feedforward neural network with backpropagation implemented from scratch using NumPy.
    Architecture: Input -> Hidden -> Output (with sigmoid activation)
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store intermediate values for backpropagation
        self.cache = {}
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """Derivative of sigmoid (a is already sigmoid(z))"""
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def forward(self, X):
        """
        Forward propagation
        X: input data of shape (batch_size, input_size)
        Returns: output predictions
        """
        # Layer 1: Input -> Hidden
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self.sigmoid(self.cache['Z1'])  # Using sigmoid for hidden layer
        
        # Layer 2: Hidden -> Output
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])  # Sigmoid for binary classification
        
        self.cache['X'] = X  # Store input for backprop
        return self.cache['A2']
    
    def compute_loss(self, y_pred, y_true):
        """
        Binary cross-entropy loss
        y_pred: predictions (batch_size, output_size)
        y_true: true labels (batch_size, output_size)
        """
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, y_true):
        """
        Backpropagation - computes gradients of loss with respect to all parameters
        y_true: true labels of shape (batch_size, output_size)
        Returns: dictionary containing gradients
        """
        m = y_true.shape[0]  # batch size
        
        # Output layer gradients
        # dL/dZ2 = dL/dA2 * dA2/dZ2
        dZ2 = self.cache['A2'] - y_true  # Simplified for sigmoid + cross-entropy
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients (backpropagating the error)
        # dL/dA1 = dL/dZ2 * dZ2/dA1 = dZ2 * W2.T
        dA1 = np.dot(dZ2, self.W2.T)
        # dL/dZ1 = dL/dA1 * dA1/dZ1
        dZ1 = dA1 * self.sigmoid_derivative(self.cache['A1'])
        dW1 = np.dot(self.cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update weights using gradient descent"""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def train_step(self, X, y):
        """Single training step: forward -> loss -> backward -> update"""
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(y_pred, y)
        
        # Backward pass
        gradients = self.backward(y)
        
        # Update parameters
        self.update_parameters(gradients)
        
        return loss, y_pred
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def get_accuracy(self, y_pred, y_true, threshold=0.5):
        """Calculate accuracy"""
        predictions = (y_pred >= threshold).astype(int)
        return np.mean(predictions == y_true)


# ==========================================
# DEMONSTRATION: XOR Problem
# ==========================================

print("=" * 60)
print("BACKPROPAGATION FROM SCRATCH WITH NUMPY")
print("=" * 60)

# XOR dataset - classic non-linearly separable problem
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])  # XOR outputs

print(f"\nDataset: XOR Problem")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
print(f"\nTraining data:")
for i in range(len(X)):
    print(f"  Input: {X[i]} -> Output: {y[i][0]}")

# Initialize network
input_size = 2
hidden_size = 4  # 4 neurons in hidden layer
output_size = 1
learning_rate = 0.5

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

print(f"\nNetwork Architecture:")
print(f"  Input layer: {input_size} neurons")
print(f"  Hidden layer: {hidden_size} neurons (sigmoid)")
print(f"  Output layer: {output_size} neuron (sigmoid)")
print(f"  Learning rate: {learning_rate}")

# Training loop
epochs = 10000
loss_history = []
accuracy_history = []

print(f"\nTraining for {epochs} epochs...")

for epoch in range(epochs):
    loss, y_pred = nn.train_step(X, y)
    loss_history.append(loss)
    
    # Calculate accuracy every 100 epochs
    if epoch % 100 == 0:
        acc = nn.get_accuracy(y_pred, y)
        accuracy_history.append(acc)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f} | Accuracy: {acc:.2%}")

# Final evaluation
final_pred = nn.predict(X)
final_loss = nn.compute_loss(final_pred, y)
final_acc = nn.get_accuracy(final_pred, y)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"Final Loss: {final_loss:.6f}")
print(f"Final Accuracy: {final_acc:.2%}")
print(f"\nPredictions:")
for i in range(len(X)):
    pred = final_pred[i][0]
    binary = 1 if pred >= 0.5 else 0
    print(f"  Input: {X[i]} -> Predicted: {pred:.4f} ({binary}) | True: {y[i][0]}")

# Visualize training progress
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss curve
axes[0].plot(loss_history, color='#e74c3c', linewidth=2)
axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Plot accuracy curve
acc_epochs = np.arange(0, epochs, 100)
axes[1].plot(acc_epochs, accuracy_history, color='#2ecc71', linewidth=2)
axes[1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1.05])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backpropagation_training.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to: {out_path}")

# ==========================================
# DETAILED BACKPROPAGATION EXPLANATION
# ==========================================

print(f"\n{'='*60}")
print("BACKPROPAGATION MATHEMATICS")
print(f"{'='*60}")

print("""
BACKPROPAGATION STEP-BY-STEP:

1. FORWARD PASS:
   Z1 = X · W1 + b1          (linear transformation)
   A1 = σ(Z1)                (sigmoid activation)
   Z2 = A1 · W2 + b2         (linear transformation)
   A2 = σ(Z2)                (sigmoid activation, output)

2. LOSS COMPUTATION (Binary Cross-Entropy):
   L = -[y·log(A2) + (1-y)·log(1-A2)]

3. BACKWARD PASS (Chain Rule):
   
   Output Layer:
   dZ2 = A2 - y              (derivative of loss w.r.t Z2)
   dW2 = (A1^T · dZ2) / m    (gradient of W2)
   db2 = sum(dZ2) / m        (gradient of b2)
   
   Hidden Layer:
   dA1 = dZ2 · W2^T          (backpropagate error)
   dZ1 = dA1 ⊙ σ'(Z1)        (element-wise multiplication)
   dW1 = (X^T · dZ1) / m     (gradient of W1)
   db1 = sum(dZ1) / m        (gradient of b1)

4. PARAMETER UPDATE (Gradient Descent):
   W = W - α · dW
   b = b - α · db
   where α = learning rate

KEY INSIGHT: Backpropagation efficiently computes gradients by 
reusing intermediate values from the forward pass (dynamic programming).
""")

# Print final weights
print(f"\n{'='*60}")
print("LEARNED PARAMETERS")
print(f"{'='*60}")
print(f"W1 (Input to Hidden):\n{nn.W1.round(4)}")
print(f"\nb1 (Hidden bias):\n{nn.b1.round(4)}")
print(f"\nW2 (Hidden to Output):\n{nn.W2.round(4)}")
print(f"\nb2 (Output bias):\n{nn.b2.round(4)}")