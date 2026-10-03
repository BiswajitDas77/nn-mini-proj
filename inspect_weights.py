import numpy as np
data = np.load("model_weights.npz")

print("Keys stored in this model file:")
print(data.files)
print("-" * 50)

for key in data.files:
    arr = data[key]
    print(f"Key: {key:<10} | Shape: {str(arr.shape):<12} | Dtype: {arr.dtype}")
    print(f"Sample values:\n{arr}\n{'-' * 50}")