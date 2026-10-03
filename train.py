import numpy as np
import csv
from model_base import ApplicantScreeningNN, min_max_normalize

def load_csv(filepath):
    """Loads CSV and returns X (features) and y (labels)."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = []
        for row in reader:
            data.append([float(x) for x in row])
            
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    return X, y

def evaluate_metrics(y_true, y_pred):
    """Computes Evaluation Metrics manually and prints ASCII Confusion Matrix."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
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

def main():
    print(">>> 1. Loading Datasets...")
    X_train_raw, y_train = load_csv('train_candidates.csv')
    X_test_raw, y_test = load_csv('test_candidates.csv')
    
    print(f"Training Set: {X_train_raw.shape[0]} samples")
    print(f"Testing Set:  {X_test_raw.shape[0]} samples")
    print("-" * 30)
    
    print(">>> 2. Normalizing Data...")
    X_train, min_vals, max_vals = min_max_normalize(X_train_raw)
    X_test, _, _ = min_max_normalize(X_test_raw, min_vals, max_vals)
    
    print(">>> 3. Initializing and Training MLP Neural Network...")
    nn = ApplicantScreeningNN(input_size=4, hidden_size=6, output_size=1, learning_rate=0.1)
    nn.min_vals = min_vals
    nn.max_vals = max_vals
    
    # Train the Network
    nn.train(X_train, y_train, epochs=2000, print_interval=200)
    
    print("\n>>> 4. Evaluating Model Performance...")
    y_pred, _ = nn.predict(X_test)
    evaluate_metrics(y_test, y_pred)
    
    print("\n>>> 5. Saving Model...")
    nn.save_model('model_weights.npz')

if __name__ == '__main__':
    main()
