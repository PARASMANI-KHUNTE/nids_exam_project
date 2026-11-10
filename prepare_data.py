import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split

# Column names for NSL-KDD dataset
col_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# 10 core numerical features to extract
core_features = [
    'duration', 'src_bytes', 'dst_bytes', 'logged_in', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'dst_host_srv_count'
]


def download_nsl_kdd():
    """Download NSL-KDD 20% dataset from GitHub"""
    url = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B_20Percent.txt"
    try:
        print("Downloading NSL-KDD 20% dataset...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), names=col_names)
        print(f"Downloaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def load_kddcup99_fallback():
    """Fallback: Load KDD Cup 99 dataset from sklearn"""
    print("Using sklearn fallback: fetch_kddcup99(percent10=True)...")
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame
    df.columns = col_names[:-1]  # No difficulty column in sklearn version
    print(f"Loaded fallback dataset! Shape: {df.shape}")
    return df


def to_binary_label(value):
    """Map a dataset label (str or bytes) to 0 = normal, 1 = attack."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return 0 if str(value).strip().strip(".").lower() == "normal" else 1


def preprocess_data(df, n_samples=10000, random_state=42):
    """Preprocess the dataframe: select features, create binary label, sample."""
    # Select core features
    X = df[core_features].copy()

    # Create binary label: 0 for normal, 1 for attack
    y = df['label'].apply(to_binary_label)

    # Combine and drop NaNs
    df_clean = pd.concat([X, y.rename('is_attack')], axis=1).dropna().reset_index(drop=True)

    # Stratified sample that preserves the original class ratio.
    # train_test_split already shuffles the rows.
    if len(df_clean) > n_samples:
        df_sampled = train_test_split(
            df_clean,
            train_size=n_samples,
            stratify=df_clean['is_attack'],
            random_state=random_state,
        )[0]
    else:
        df_sampled = df_clean

    return df_sampled.reset_index(drop=True)


# Main execution
if __name__ == "__main__":
    # Try primary source
    df = download_nsl_kdd()

    # Fallback if primary fails
    if df is None:
        df = load_kddcup99_fallback()

    # Preprocess
    df_final = preprocess_data(df)

    # Save to CSV
    output_path = "network_intrusion.csv"
    df_final.to_csv(output_path, index=False)

    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"Shape: {df_final.shape}")
    print(f"Features: {core_features}")
    print(f"\nClass Distribution:")
    print(df_final['is_attack'].value_counts())
    print(f"\nSaved to: {output_path}")
