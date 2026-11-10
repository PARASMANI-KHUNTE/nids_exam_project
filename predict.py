import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("nids_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load test data
df = pd.read_csv("network_intrusion.csv")
X = df.drop("is_attack", axis=1)
y = df["is_attack"]

# Get test indices (same split as training)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Find one normal (0) and one attack (1) sample
normal_idx = X_test[y_test == 0].index[0]
attack_idx = X_test[y_test == 1].index[0]

# Predict normal sample
normal_sample = X_test_scaled[X_test.index.get_loc(normal_idx)].reshape(1, -1)
normal_pred = model.predict(normal_sample)[0]
normal_actual = y_test.loc[normal_idx]

# Predict attack sample
attack_sample = X_test_scaled[X_test.index.get_loc(attack_idx)].reshape(1, -1)
attack_pred = model.predict(attack_sample)[0]
attack_actual = y_test.loc[attack_idx]

print("=== Live Inference Demo ===")
print(f"\nSample 1 (Normal traffic):")
print(f"  Predicted: {normal_pred} ({'Attack' if normal_pred else 'Normal'})")
print(f"  Actual:    {normal_actual} ({'Attack' if normal_actual else 'Normal'})")
print(f"  Match: {'✓' if normal_pred == normal_actual else '✗'}")

print(f"\nSample 2 (Attack traffic):")
print(f"  Predicted: {attack_pred} ({'Attack' if attack_pred else 'Normal'})")
print(f"  Actual:    {attack_actual} ({'Attack' if attack_actual else 'Normal'})")
print(f"  Match: {'✓' if attack_pred == attack_actual else '✗'}")