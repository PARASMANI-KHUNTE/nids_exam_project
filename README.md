# Network Intrusion Detection System (NIDS)

End-to-end ML/DL project for practical exam preparation. Detects network intrusions using Random Forest and PyTorch MLP on NSL-KDD dataset.

## Quick Start

```bash
# Install dependencies
pip install pandas scikit-learn matplotlib seaborn joblib torch requests

# 1. Prepare dataset (downloads NSL-KDD 20% or uses sklearn fallback)
python prepare_data.py

# 2. Train Random Forest (99% accuracy)
python train_ml.py

# 3. Train PyTorch MLP (90% accuracy)
python train_dl.py

# 4. Live inference demo
python predict.py
```

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | **99.0%** | 99% | 99% | 99% |
| PyTorch MLP | **90.1%** | 90% | 90% | 90% |

## Features (10 Core Network Security Features)

| Feature | Description |
|---------|-------------|
| `duration` | Connection duration (seconds) |
| `src_bytes` | Bytes sent from source |
| `dst_bytes` | Bytes sent to destination |
| `logged_in` | Successful login (0/1) |
| `count` | Connections to same host (2s window) |
| `srv_count` | Connections to same service (2s window) |
| `serror_rate` | % SYN errors (SYN flood indicator) |
| `srv_serror_rate` | Service-specific SYN error rate |
| `same_srv_rate` | % connections to same service |
| `dst_host_srv_count` | Connections to destination host/service |

## Project Structure

```
nids_exam_project/
├── network_intrusion.csv      # 10K samples (5339 normal / 4661 attack)
├── prepare_data.py            # Dataset download & preprocessing
├── train_ml.py                # Random Forest training (20 lines for exam)
├── train_dl.py                # PyTorch MLP training
├── predict.py                 # Live inference demo
├── nids_exam_notebook.ipynb   # Complete end-to-end notebook
├── EXAM_CHEATSHEET.md         # 20-line exam code + 15 Viva Q&A
├── requirements.txt           # Python dependencies
├── nids_rf_model.pkl          # Trained RF model
├── scaler.pkl                 # Fitted StandardScaler
├── nids_dl_model.pth          # Trained MLP weights
└── confusion_matrix.png       # RF confusion matrix visualization
```

## Exam Preparation

- **Written Exam**: Memorize the 20-line `train_ml.py` in `EXAM_CHEATSHEET.md`
- **Viva Voce**: 15 high-probability questions with crisp 2-sentence answers
- **Live Demo**: Run `python predict.py` for 2-second inference demonstration

## Dataset

- **Primary**: NSL-KDD 20% (25,192 rows) from GitHub
- **Fallback**: sklearn `fetch_kddcup99(percent10=True)`
- **Processed**: 10,000 stratified samples, 10 numerical features, binary labels (0=normal, 1=attack)

## Requirements

- Python 3.8+
- pandas, scikit-learn, matplotlib, seaborn, joblib, torch, requests

---

*Generated for practical exam (Written + Viva Voce) preparation*