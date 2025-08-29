#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal Depression-Score Prediction
-------------------------------------
• 视频特征：BGR 均值/标准差 + 边缘密度
• 音频特征：MFCC 均值 + 标准差
• 模型：两层 MLP 回归
依赖：opencv-python, librosa, scipy, torch, scikit-learn, matplotlib
"""
import os
import logging
import numpy as np
import cv2
import librosa
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_BASE   = r"C:\AAFujiancankao"
AUDIO_BASE  = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\audio"

# 标签文件名统一配置
LABEL_FILES = {
    "train":   "train_label.mat",
    "dev":     "develop_label.mat",   # ← 唯一正确写法
    "test":    "test_label.mat"
}

# ----------------- 日志 -----------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------- 工具：防呆路径检查 ----------
def checked_path(p, desc="file"):
    if not os.path.isfile(p):
        logging.error(f"{desc} 不存在: {p}")
        raise FileNotFoundError(p)
    return p

# --------- 载入标签 ----------
def load_labels(label_path):
    """
    Load labels from .mat file. Accepts keys 'label' / 'labels' /
    first numeric array found.
    """
    label_path = checked_path(label_path, "Label file")
    mat = sio.loadmat(label_path)
    arrays = {k: v for k, v in mat.items() if isinstance(v, np.ndarray)}
    if "label" in arrays:
        labels = arrays["label"]
    elif "labels" in arrays:
        labels = arrays["labels"]
    else:
        labels = next(iter(arrays.values()))
    labels = np.asarray(labels).squeeze()
    logging.info(f"Loaded labels from {label_path}, shape={labels.shape}")
    return labels

# --------- 音频特征 ----------
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feat = np.concatenate([mfcc.mean(1), mfcc.std(1)])  # (26,)
        return feat
    except Exception as e:
        logging.error(f"Audio error {audio_path}: {e}")
        return np.zeros(26, dtype=np.float32)

# --------- 视频特征 ----------
def extract_video_features(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("cannot open video")

        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = int(fps)
        feats = []
        for fid in range(0, frame_cnt, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frm = cap.read()
            if not ok:
                continue
            frm = cv2.resize(frm, (64, 64))
            m, s = cv2.meanStdDev(frm)
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 100, 200)
            dens = edge.mean()
            feats.append(np.hstack([m.flatten(), s.flatten(), dens]))
        cap.release()
        return np.mean(feats, axis=0) if feats else np.zeros(7, dtype=np.float32)
    except Exception as e:
        logging.error(f"Video error {video_path}: {e}")
        return np.zeros(7, dtype=np.float32)

# --------- 数据集 ----------
class MultiModalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --------- 模型 ----------
class MLPModel(nn.Module):
    def __init__(self, in_dim, h1=64, h2=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --------- 训练 ----------
def train(model, tr_loader, val_loader, epochs=50, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0
        for Xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(Xb).squeeze(), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*len(Xb)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += crit(model(Xb).squeeze(), yb).item()*len(Xb)
        val_loss /= len(val_loader.dataset)
        logging.info(f"Epoch {ep:02d} | Train {tr_loss:.4f} | Val {val_loss:.4f}")

# --------- 评估 ----------
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    print_scores(y, preds)
    return preds

def print_scores(y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    logging.info(f"Test R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

# --------- 可视化 ----------
def plot_scatter(y, y_hat, title="Actual vs Predicted"):
    plt.figure(figsize=(5,5))
    plt.scatter(y, y_hat, alpha=.7)
    lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_plot.png")
    plt.show()

# --------- 主流程 ----------
def main():
    # ---------- 路径拼装 ----------
    paths = {
        "train": {
            "video": {
                "free":  os.path.join(DATA_BASE, "train", "Freeform"),
                "north": os.path.join(DATA_BASE, "train", "Northwind")
            },
            "audio": {
                "free":  os.path.join(AUDIO_BASE, "train", "Freeform"),
                "north": os.path.join(AUDIO_BASE, "train", "Northwind")
            },
            "label": os.path.join(DATA_BASE, "label", LABEL_FILES["train"])
        },
        "dev": {
            "video": {
                "free":  os.path.join(DATA_BASE, "dev", "Freeform"),
                "north": os.path.join(DATA_BASE, "dev", "Northwind")
            },
            "audio": {
                "free":  os.path.join(AUDIO_BASE, "dev", "Freeform"),
                "north": os.path.join(AUDIO_BASE, "dev", "Northwind")
            },
            "label": os.path.join(DATA_BASE, "label", LABEL_FILES["dev"])
        },
        "test": {
            "video": {
                "free":  os.path.join(DATA_BASE, "test", "Freeform"),
                "north": os.path.join(DATA_BASE, "test", "Northwind")
            },
            "audio": {
                "free":  os.path.join(AUDIO_BASE, "test", "Freeform"),
                "north": os.path.join(AUDIO_BASE, "test", "Northwind")
            },
            "label": os.path.join(DATA_BASE, "label", LABEL_FILES["test"])
        }
    }

    # ---------- 加载标签 ----------
    y_train = load_labels(paths["train"]["label"])
    y_dev   = load_labels(paths["dev"]["label"])
    y_test  = load_labels(paths["test"]["label"])

    # ---------- 提取特征 ----------
    def build_features(split):
        vids_free  = sorted(os.listdir(paths[split]["video"]["free"]))
        vids_north = sorted(os.listdir(paths[split]["video"]["north"]))
        auds_free  = sorted(os.listdir(paths[split]["audio"]["free"]))
        auds_north = sorted(os.listdir(paths[split]["audio"]["north"]))
        y_split = {"train":y_train,"dev":y_dev,"test":y_test}[split]
        feats = []
        for i in range(len(y_split)):
            v_free  = os.path.join(paths[split]["video"]["free"],  vids_free[i])
            v_north = os.path.join(paths[split]["video"]["north"], vids_north[i])
            a_free  = os.path.join(paths[split]["audio"]["free"],  auds_free[i])
            a_north = os.path.join(paths[split]["audio"]["north"], auds_north[i])
            feat = np.concatenate([
                extract_audio_features(a_free),
                extract_audio_features(a_north),
                extract_video_features(v_free),
                extract_video_features(v_north)
            ])
            feats.append(feat.astype(np.float32))
        return np.vstack(feats)

    X_train = build_features("train")
    X_dev   = build_features("dev")
    X_test  = build_features("test")

    # ---------- 标准化 ----------
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_dev   = scaler.transform(X_dev)
    X_test  = scaler.transform(X_test)

    # ---------- 数据加载器 ----------
    train_loader = DataLoader(MultiModalDataset(X_train, y_train), batch_size=8, shuffle=True)
    dev_loader   = DataLoader(MultiModalDataset(X_dev,   y_dev),   batch_size=8, shuffle=False)

    # ---------- 建模 ----------
    model = MLPModel(in_dim=X_train.shape[1])
    logging.info(f"Model input dim = {X_train.shape[1]}")
    train(model, train_loader, dev_loader, epochs=50, lr=1e-3)

    # ---------- 评估 ----------
    y_pred = evaluate(model, X_test, y_test)

    # ---------- 图 ----------
    plot_scatter(y_test, y_pred)

if __name__ == "__main__":
    main()



