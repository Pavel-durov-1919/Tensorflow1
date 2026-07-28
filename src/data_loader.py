"""
C-MAPSS FD001 Data Loader & Validation Module
Author: Member A (Data & ML Lead)
"""

import hashlib
import os
import urllib.request
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")

URLS = {
    "train_FD001.txt": "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/raw/main/train_FD001.txt",
    "test_FD001.txt": "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/raw/main/test_FD001.txt",
    "RUL_FD001.txt": "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/raw/main/RUL_FD001.txt",
}

COLUMNS = ["unit_nr", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
MAX_RUL_CLIP = 125


def compute_checksums(file_path: str) -> dict:
    """Compute MD5 and SHA256 checksums of a file."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download_fd001(raw_dir: str = RAW_DATA_DIR) -> dict:
    """Download FD001 dataset files if not already present and compute checksums."""
    os.makedirs(raw_dir, exist_ok=True)
    checksums = {}
    
    for filename, url in URLS.items():
        dest_path = os.path.join(raw_dir, filename)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            print(f"Downloading {filename} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
        
        checksums[filename] = compute_checksums(dest_path)
        checksums[filename]["size_bytes"] = os.path.getsize(dest_path)
        print(f"Verified {filename}: {checksums[filename]['size_bytes']} bytes, SHA256: {checksums[filename]['sha256'][:16]}...")
        
    return checksums


def load_raw_df(file_path: str, names: list = COLUMNS) -> pd.DataFrame:
    """Load space-delimited text file into DataFrame."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=names)
    if df.shape[1] < len(names):
        raise ValueError(f"Schema mismatch in {file_path}: expected at least {len(names)} columns, got {df.shape[1]}")
    return df


def calculate_rul(df: pd.DataFrame, max_rul_clip: int = MAX_RUL_CLIP) -> pd.DataFrame:
    """Calculate RUL_raw and RUL_clipped per engine unit."""
    max_cycles = df.groupby("unit_nr")["cycle"].transform("max")
    df = df.copy()
    df["RUL_raw"] = max_cycles - df["cycle"]
    df["RUL_clipped"] = np.minimum(df["RUL_raw"], max_rul_clip)
    return df


def select_useful_sensors(df: pd.DataFrame, threshold: float = 0.01) -> tuple:
    """Identify constant/low-variance sensors and useful feature columns."""
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    stds = df[sensor_cols].std()
    
    constant_sensors = stds[stds < threshold].index.tolist()
    useful_sensors = stds[stds >= threshold].index.tolist()
    
    # op_setting_1 and 2 are useful operational settings; op_setting_3 is constant in FD001
    useful_op_settings = ["op_setting_1", "op_setting_2"]
    useful_features = useful_op_settings + useful_sensors
    
    metadata = {
        "constant_sensors": constant_sensors,
        "useful_sensors": useful_sensors,
        "useful_features": useful_features,
        "sensor_stds": stds.to_dict()
    }
    return useful_features, metadata


def get_validated_dataset(raw_dir: str = RAW_DATA_DIR) -> tuple:
    """Complete data ingestion, validation, and RUL calculation pipeline."""
    checksums = download_fd001(raw_dir)
    
    train_path = os.path.join(raw_dir, "train_FD001.txt")
    test_path = os.path.join(raw_dir, "test_FD001.txt")
    rul_path = os.path.join(raw_dir, "RUL_FD001.txt")
    
    train_df = load_raw_df(train_path)
    test_df = load_raw_df(test_path)
    test_rul_series = pd.read_csv(rul_path, header=None, names=["RUL_true"])["RUL_true"]
    
    # Data Validation Checks
    assert not train_df.isnull().values.any(), "Missing values detected in training set!"
    assert not test_df.isnull().values.any(), "Missing values detected in test set!"
    assert train_df["unit_nr"].nunique() == 100, f"Expected 100 engines in train, got {train_df['unit_nr'].nunique()}"
    
    train_df = calculate_rul(train_df)
    useful_features, sensor_meta = select_useful_sensors(train_df)
    
    return train_df, test_df, test_rul_series, checksums, sensor_meta, useful_features


if __name__ == "__main__":
    tr, te, rul, chk, meta, feats = get_validated_dataset()
    print(f"Dataset successfully loaded. Train shape: {tr.shape}, Useful features count: {len(feats)}")
