# NIDS Exam Cheatsheet

---

## 1. Written Exam Code Block (20-line `train_ml.py`)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("network_intrusion.csv")
X, y = df.drop("is_attack", axis=1), df["is_attack"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "nids_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

---

## 2. Dataset & Feature Summary

| Metric | Value |
|--------|-------|
| **Total Rows** | 10,000 (5,339 normal / 4,661 attack) |
| **Features** | 10 numerical |
| **Test Split** | 20% (2,000 samples) |
| **RF Accuracy** | 99.0% |
| **RF Precision** | 99% (both classes) |
| **RF Recall** | Normal: 100%, Attack: 98% |
| **RF F1-Score** | 99% (both classes) |
| **DL Accuracy** | 90.15% |
| **DL Test Loss** | 0.2095 |

### Feature Meanings (Network Security Context)

| Feature | Description | Security Relevance |
|---------|-------------|-------------------|
| `duration` | Connection duration (seconds) | Long durations = data exfiltration; zero = scan/probe |
| `src_bytes` | Bytes sent from source | High = data upload/exfil; Low = reconnaissance |
| `dst_bytes` | Bytes sent to destination | High = download/response; Asymmetry = anomaly |
| `logged_in` | Successful login (0/1) | 0 + admin port = brute force attempt |
| `count` | Connections to same host (2s) | High = DoS/flood; Low = stealth scan |
| `srv_count` | Connections to same service (2s) | High = service-specific DoS |
| `serror_rate` | % SYN errors | High = SYN flood (DoS) |
| `srv_serror_rate` | % SYN errors for service | Service-targeted SYN flood |
| `same_srv_rate` | % connections to same service | Low = port scan; High = focused attack |
| `dst_host_srv_count` | Connections to dest host/service | High = distributed attack on service |

---

## 3. 15 Viva Voce Q&A

### Model Selection
**Q1: Why Random Forest for this NIDS task?**
A: Handles mixed feature scales without tuning, robust to outliers in network traffic, provides feature importance for explainability, and ensembles reduce variance—critical for detecting diverse attack types with limited data.

**Q2: Why a 3-layer MLP (10→32→16→1) for deep learning?**
A: 10 inputs match our features; 32→16 provides progressive bottleneck for compressed threat representation; single sigmoid output gives calibrated attack probability. Shallow enough to train fast on CPU, deep enough to learn non-linear attack boundaries.

### Data Handling
**Q3: What is data leakage and how does `fit_transform` vs `transform` prevent it?**
A: Leakage occurs when test info influences training. `fit_transform` on training computes mean/std **only from train data**; `transform` on test applies those exact params—preventing test distribution from contaminating the scaler.

**Q4: Why stratified sampling?**
A: Preserves class ratio (53:47) in both train/test. Without it, random split could yield test set with 90% normal, inflating accuracy while missing attacks.

### Confusion Matrix & Metrics
**Q5: Explain False Positives vs False Negatives in NIDS context.**
A: FP = Benign flagged as attack → analyst fatigue, wasted resources. FN = Attack missed → breach, data theft. **FN cost >> FP cost** in security; hence we prioritize Recall.

**Q6: Why is Recall more important than Precision for intrusion detection?**
A: Missing an attack (low recall) causes direct harm; false alarms (low precision) cause operational overhead. We tune threshold to maximize recall, accepting more FPs.

**Q7: What does the confusion matrix [[1063, 5], [15, 917]] tell you?**
A: 5 normal→attack (FP), 15 attack→normal (FN). Model catches 98% of attacks (recall) with only 0.5% false alarm rate on normal traffic.

### Deep Learning Components
**Q8: Role of `StandardScaler`?**
A: Normalizes features to zero mean, unit variance. Critical for neural nets—unscaled inputs cause gradient explosion/vanishing and slow convergence.

**Q9: Why `ReLU` activation?**
A: Non-linear, cheap (max(0,x)), mitigates vanishing gradient vs sigmoid/tanh, enables sparse activation for efficient representation learning.

**Q10: Why `Sigmoid` on output layer?**
A: Squashes logits to [0,1] probability for binary classification; pairs with BCELoss for proper probabilistic interpretation.

**Q11: Why `BCELoss` (Binary Cross Entropy)?**
A: Measures distance between predicted probability and true label (0/1). Penalizes confident wrong predictions exponentially—drives model toward calibrated probabilities.

**Q12: Why `Adam` optimizer?**
A: Adaptive learning rates per parameter (momentum + RMSProp). Handles varying gradient scales across layers, converges faster than SGD on this small network.

### Practical Deployment
**Q13: How would you handle concept drift in production NIDS?**
A: Monitor prediction distribution shift; retrain weekly with sliding window of recent labeled data; use online learning or ensemble with decaying weights for older patterns.

**Q14: What if attack classes are highly imbalanced (e.g., 99.9% normal)?**
A: Use class_weight='balanced', oversample minority (SMOTE), or anomaly detection (Isolation Forest) instead of supervised classification; adjust decision threshold post-training.

**Q15: How to explain model decisions to a SOC analyst?**
A: Random Forest: show feature importance (e.g., `serror_rate` top for DoS). SHAP values per prediction: "This connection flagged because `src_bytes=0`, `serror_rate=1.0`, `count=500` matches SYN flood signature."

---

## Quick Reference: File Structure

```
nids_exam_project/
├── network_intrusion.csv      # 10K rows × 11 cols (10 features + label)
├── prepare_data.py            # Downloads NSL-KDD, creates CSV
├── train_ml.py                # Random Forest (99% acc)
├── train_dl.py                # PyTorch MLP (90% acc)
├── predict.py                 # Live inference demo
├── nids_rf_model.pkl          # Trained RF model
├── scaler.pkl                 # Fitted StandardScaler
├── nids_dl_model.pth          # Trained MLP weights
├── confusion_matrix.png       # RF confusion matrix plot
└── EXAM_CHEATSHEET.md         # This file
```

---

**Exam Tip**: Memorize the 20-line code block. Key lines examiners check: `stratify=y`, `fit_transform`/`transform` split, `random_state=42`, `joblib.dump`.