import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_samples=200):
    """
    Generates synthetic recruitment dataset.
    Features: Years of Experience (0-10), Technical Score (0-100), Soft Skills Score (0-100), Projects Completed (0-6).
    Target: 1 (Shortlist) or 0 (Reject).
    """
    # Initialize arrays
    X = np.zeros((num_samples, 4))
    y = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        exp = np.random.uniform(0, 10)
        tech = np.random.uniform(0, 100)
        soft = np.random.uniform(0, 100)
        proj = np.random.randint(0, 7)
        
        X[i] = [exp, tech, soft, proj]
        
        # Hiring Logic: 
        # A combination of scores determines a latent 'overall suitability' score
        # Weighting: Tech (40%), Soft (30%), Exp (15%), Proj (15%)
        # Experience is out of 10, so exp/10*100 scales it to 100
        # Projects is out of 6, so proj/6*100 scales it to 100
        score = (tech * 0.4) + (soft * 0.3) + ((exp / 10 * 100) * 0.15) + ((proj / 6 * 100) * 0.15)
        
        # Realistic noise and thresholds
        # Must have a decent overall score, but also minimum technical and soft skills
        if score > 60 and tech >= 50 and soft >= 40:
            y[i] = 1
        else:
            y[i] = 0
            
    return X, y

def min_max_normalize(X, min_vals=None, max_vals=None):
    """
    Scales features to be between 0 and 1 manually.
    If min_vals and max_vals are not provided, computes them from X.
    """
    if min_vals is None or max_vals is None:
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
    
    # Range (max - min), add small epsilon to avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8
    
    X_norm = (X - min_vals) / range_vals
    return X_norm, min_vals, max_vals

def train_test_split(X, y, test_ratio=0.2):
    """
    Splits the dataset into training and testing sets manually.
    """
    num_samples = X.shape[0]
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    test_size = int(num_samples * test_ratio)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


class ApplicantScreeningNN:
    """
    Multi-Layer Perceptron for Applicant Screening built from scratch.
    Architecture:
      - Input Layer: 4 neurons
      - Hidden Layer: Configurable neurons with ReLU activation
      - Output Layer: 1 neuron with Sigmoid activation
    """
    def __init__(self, input_size=4, hidden_size=6, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # --- Weight Initialization ---
        # W1: He Initialization (good for ReLU)
        # Shape: (input_size, hidden_size)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        # b1: Zero Initialization, Shape: (1, hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        # W2: Xavier/Glorot Initialization (good for Sigmoid)
        # Shape: (hidden_size, output_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        # b2: Zero Initialization, Shape: (1, output_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # Variables to store normalization parameters
        self.min_vals = None
        self.max_vals = None
        
    def relu(self, Z):
        """ReLU Activation Function"""
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        """Derivative of ReLU Activation Function"""
        return (Z > 0).astype(float)
        
    def sigmoid(self, Z):
        """Sigmoid Activation Function with numerical stability clipping"""
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))
        
    def binary_cross_entropy(self, Y_true, Y_pred):
        """
        Computes the Binary Cross-Entropy Loss.
        Uses epsilon to prevent log(0) which causes NaN errors.
        """
        m = Y_true.shape[0]
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
        return loss

    def forward_propagation(self, X):
        """
        Executes explicit forward propagation using matrix dot products.
        """
        # --- Hidden Layer ---
        # Z1 = X.W1 + b1
        self.Z1 = np.dot(X, self.W1) + self.b1
        # A1 = ReLU(Z1)
        self.A1 = self.relu(self.Z1)
        
        # --- Output Layer ---
        # Z2 = A1.W2 + b2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # A2 = Sigmoid(Z2)
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
        
    def backward_propagation(self, X, Y):
        """
        Executes explicit backward propagation using the Chain Rule.
        """
        m = X.shape[0] # Number of samples
        
        # --- Output Layer Gradients ---
        # Derivative of Loss wrt Z2 (Simiplification for BCE with Sigmoid)
        # dZ2 = A2 - Y
        dZ2 = self.A2 - Y
        
        # Derivative of Loss wrt W2
        # dW2 = (A1^T . dZ2) / m
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        
        # Derivative of Loss wrt b2
        # db2 = sum(dZ2) / m
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # --- Hidden Layer Gradients ---
        # Derivative of Loss wrt A1
        # dA1 = dZ2 . W2^T
        dA1 = np.dot(dZ2, self.W2.T)
        
        # Derivative of Loss wrt Z1
        # dZ1 = dA1 * g'(Z1) where g' is ReLU derivative
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        
        # Derivative of Loss wrt W1
        # dW1 = (X^T . dZ1) / m
        dW1 = (1/m) * np.dot(X.T, dZ1)
        
        # Derivative of Loss wrt b1
        # db1 = sum(dZ1) / m
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Vanilla Gradient Descent Update ---
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, Y, epochs=1000, print_interval=100):
        """
        Trains the neural network using Gradient Descent.
        """
        print(f"Starting Training for {epochs} epochs...")
        for epoch in range(epochs):
            # 1. Forward pass
            Y_pred = self.forward_propagation(X)
            
            # 2. Loss computation
            loss = self.binary_cross_entropy(Y, Y_pred)
            
            # 3. Backward pass
            self.backward_propagation(X, Y)
            
            # 4. Logging
            if epoch % print_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")
                
    def predict(self, X):
        """
        Generates binary predictions (0 or 1) and raw probabilities.
        """
        probabilities = self.forward_propagation(X)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions, probabilities

    def evaluate(self, X_test, y_test):
        """
        Computes Evaluation Metrics manually and prints ASCII Confusion Matrix.
        """
        y_pred, _ = self.predict(X_test)
        
        # Confusion Matrix Elements
        TP = np.sum((y_test == 1) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))
        
        # Metrics Calculation
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1_score:.4f}")
        
        print("\n=== ASCII Confusion Matrix ===")
        print(f"                 Predicted: 0   Predicted: 1")
        print(f"Actual: 0 (Rej) |    {TN:4d}      |    {FP:4d}      |")
        print(f"Actual: 1 (Acc) |    {FN:4d}      |    {TP:4d}      |")

    def screen_candidates(self, candidates):
        """
        Batch Recruitment Screening function.
        Accepts raw candidates (unscaled), scales them using training min/max,
        predicts probability, and outputs a ranked leaderboard.
        """
        if isinstance(candidates, list):
            X_raw = np.array(candidates)
        else:
            X_raw = candidates
            
        # Normalize using the Min/Max from the training dataset
        X_norm, _, _ = min_max_normalize(X_raw, self.min_vals, self.max_vals)
        
        # Forward pass to get match probability
        _, probabilities = self.predict(X_norm)
        
        results = []
        for i in range(X_raw.shape[0]):
            score = probabilities[i][0] * 100 # Convert to Percentage
            decision = "Shortlisted" if score >= 50 else "Rejected"
            results.append({
                "Candidate ID": f"CAND-{i + 1:03d}",
                "Exp": X_raw[i][0],
                "Tech": X_raw[i][1],
                "Soft": X_raw[i][2],
                "Proj": X_raw[i][3],
                "Score": score,
                "Decision": decision
            })
            
        # Rank by Match Score (Highest to Lowest)
        results_sorted = sorted(results, key=lambda x: x["Score"], reverse=True)
        
        print("\n=== Recruitment Screening Leaderboard ===")
        print(f"{'Rank':<5} | {'ID':<10} | {'Exp':<4} | {'Tech':<5} | {'Soft':<5} | {'Proj':<4} | {'Match Score (%)':<15} | {'Decision'}")
        print("-" * 85)
        for rank, res in enumerate(results_sorted, 1):
            print(f"{rank:<5} | {res['Candidate ID']:<10} | {res['Exp']:<4.1f} | {res['Tech']:<5.1f} | {res['Soft']:<5.1f} | {res['Proj']:<4.0f} | {res['Score']:>14.2f}% | {res['Decision']}")


if __name__ == "__main__":
    print(">>> 1. Generating Synthetic Recruitment Dataset...")
    X_data, y_data = generate_synthetic_data(num_samples=500)
    
    print(">>> 2. Splitting and Normalizing Data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_data, y_data, test_ratio=0.2)
    
    # Normalize Training Data
    X_train, min_vals, max_vals = min_max_normalize(X_train_raw)
    # Normalize Testing Data with Training Min/Max
    X_test, _, _ = min_max_normalize(X_test_raw, min_vals, max_vals)
    
    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Testing Set:  {X_test.shape[0]} samples")
    print("-" * 30)
    
    print(">>> 3. Initializing and Training MLP Neural Network...")
    # Initialize Model (Input: 4 features, Hidden: 6 neurons, Output: 1)
    nn = ApplicantScreeningNN(input_size=4, hidden_size=6, output_size=1, learning_rate=0.1)
    
    # We assign the min and max values to the network for future screening
    nn.min_vals = min_vals
    nn.max_vals = max_vals
    
    # Train the Network
    nn.train(X_train, y_train, epochs=2000, print_interval=200)
    
    print("\n>>> 4. Evaluating Model Performance...")
    nn.evaluate(X_test, y_test)
    
    print("\n>>> 5. Batch Recruitment Screening Simulation...")
    # Raw Candidate Profiles: [Experience(0-10), Tech(0-100), Soft(0-100), Projects(0-6)]
    sample_candidates = [
        [2.5, 88.0, 75.0, 2],  # Strong junior
        [8.0, 95.0, 90.0, 6],  # Exceptional senior (Must Shortlist)
        [1.0, 45.0, 60.0, 1],  # Underqualified
        [5.0, 55.0, 30.0, 3],  # Poor soft skills
        [4.0, 80.0, 85.0, 4]   # Solid mid-level
    ]
    
    nn.screen_candidates(sample_candidates)
