import numpy as np

def min_max_normalize(X, min_vals=None, max_vals=None):
    if min_vals is None or max_vals is None:
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8
    X_norm = (X - min_vals) / range_vals
    return X_norm, min_vals, max_vals

class ApplicantScreeningNN:
    """
    Multi-Layer Perceptron for Applicant Screening built from scratch.
    """
    def __init__(self, input_size=4, hidden_size=6, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # He Initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Xavier Initialization
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        self.min_vals = None
        self.max_vals = None
        
    def relu(self, Z):
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
        
    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))
        
    def binary_cross_entropy(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
        return loss

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
        
    def backward_propagation(self, X, Y):
        m = X.shape[0]
        
        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, Y, epochs=1000, print_interval=100):
        print(f"Starting Training for {epochs} epochs...")
        for epoch in range(epochs):
            Y_pred = self.forward_propagation(X)
            loss = self.binary_cross_entropy(Y, Y_pred)
            self.backward_propagation(X, Y)
            
            if epoch % print_interval == 0 or epoch == epochs - 1:
                # Calculate training accuracy
                preds = (Y_pred >= 0.5).astype(int)
                acc = np.mean(preds == Y) * 100
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Train Acc: {acc:.2f}%")
                
    def predict(self, X):
        probabilities = self.forward_propagation(X)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions, probabilities

    def save_model(self, filepath):
        """Saves weights and normalization params to npz"""
        np.savez(filepath, 
                 W1=self.W1, b1=self.b1, 
                 W2=self.W2, b2=self.b2,
                 min_vals=self.min_vals, max_vals=self.max_vals)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads weights and normalization params from npz"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.min_vals = data['min_vals']
        self.max_vals = data['max_vals']
        # Set hidden/output dimensions from weights
        self.hidden_size = self.W1.shape[1]
        self.input_size = self.W1.shape[0]
        self.output_size = self.W2.shape[1]
