#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025 中青杯 C 题 —— 批量预处理脚本
功能：提取视频分块-LBP(944) + MFCC-Δ-ΔΔ(39) = 983 维特征
      并按 4 类标签保存为 .pt 文件，供后续 PyTorch 训练/推理。
"""

import os, re, glob, logging
from collections import Counter

import numpy as np
import scipy.io as sio


import cv2
import librosa
import torch
from tqdm import tqdm

# ---------------- 尝试导入 moviepy ----------------
try:
    from moviepy.editor import VideoFileClip
    USE_MOVIEPY = True
except ModuleNotFoundError:
    import soundfile as sf
    USE_MOVIEPY = False
    logging.warning("moviepy 不可用，改用 soundfile 读取音轨（需 ffmpeg）")
# -------------------------------------------------

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# ========== 路径 ==========
BASE_DIR = r"C:"

DATA_DIR = os.path.join(
    BASE_DIR,
    r"C:\AAFujiancankao")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
DEV_DIR   = os.path.join(DATA_DIR, "dev")
TEST_DIR  = os.path.join(DATA_DIR, "test")

LABEL_DIR = os.path.join(DATA_DIR, "label")
TRAIN_LAB = os.path.join(LABEL_DIR, "train_label.mat")
DEV_LAB   = os.path.join(LABEL_DIR, "develop_label.mat")
TEST_LAB  = os.path.join(LABEL_DIR, "test_label.mat")

OUT_DIR = os.path.join(BASE_DIR, "逐帧数据处理")
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 处理参数 ==========
MAX_FRAMES    = 64
FACE_SIZE     = (128, 128)
AUDIO_SR      = 16_000
N_MFCC        = 13           # 13 × (静音/Δ/ΔΔ) = 39
AUDIO_FRAMES  = 128          # 对齐长度
LBP_P, LBP_R  = 8, 1
GRID_SIZE     = 4            # 4×4 分块
LBP_DIM       = 59 * GRID_SIZE * GRID_SIZE  # 59 ×16 = 944
TOTAL_DIM     = LBP_DIM + 39                # 983

# ========== 工具函数 ==========
def load_labels(mat_path: str) -> np.ndarray:
    """读取 .mat 文件 → 一维 score 数组 (索引即受试者 ID)"""
    data = sio.loadmat(mat_path)
    keys = [k for k in data if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{mat_path} 无有效变量")
    if {"id", "label"}.issubset(keys):
        idx = np.squeeze(data["id"]).astype(int)
        lab = np.squeeze(data["label"]).astype(float)
        arr = np.zeros(idx.max() + 1, dtype=float)
        arr[idx] = lab
        return arr
    return np.squeeze(data[keys[0]]).astype(float)

def score_to_cls(score: float) -> int:
    """BDI-II → 4 类"""
    return 0 if score <= 13 else 1 if score <= 19 else 2 if score <= 28 else 3

def extract_block_lbp(gray: np.ndarray) -> np.ndarray:
    """16 块 × 59 bin -> 944 维"""
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    h, w = gray.shape
    bh, bw = h // GRID_SIZE, w // GRID_SIZE
    feats = []
    for by in range(GRID_SIZE):
        for bx in range(GRID_SIZE):
            patch = lbp[by*bh:(by+1)*bh, bx*bw:(bx+1)*bw]
            hist, _ = np.histogram(
                patch.ravel(), bins=59, range=(0, 58), density=True)
            feats.append(hist)
    return np.hstack(feats).astype(np.float32)

def eye_align(face: np.ndarray) -> np.ndarray:
    """双眼矫正"""
    eye_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_det.detectMultiScale(face, 1.1, 3)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]  # 最左 & 最右
        (x1,y1,w1,h1), (x2,y2,w2,h2) = eyes
        cx1, cy1 = x1+w1/2, y1+h1/2
        cx2, cy2 = x2+w2/2, y2+h2/2
        ang = np.degrees(np.arctan2(cy2-cy1, cx2-cx1))
        M = cv2.getRotationMatrix2D((face.shape[1]/2, face.shape[0]/2), ang, 1.0)
        face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return face

def read_audio(vpath: str) -> np.ndarray:
    """返回 mono 声道、16 kHz 浮点波形"""
    if USE_MOVIEPY:
        clip = VideoFileClip(vpath, audio=True)
        if clip.audio is None:
            clip.close(); raise RuntimeError("无音轨")
        aud = clip.audio.to_soundarray(fps=AUDIO_SR)
        clip.close()
    else:
        import soundfile as sf
        aud, sr0 = sf.read(vpath)
        if sr0 != AUDIO_SR:
            aud = librosa.resample(aud.T, sr0, AUDIO_SR).T
    if aud.ndim == 2:
        aud = aud.mean(axis=1)
    return aud.astype(float)

def video_feature(vpath: str) -> np.ndarray:
    """返回 983 维特征"""
    # ---- 视觉分块 LBP 均值 ----
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise IOError("无法打开")
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES
    idxs = np.linspace(0, tot-1, num=min(tot, MAX_FRAMES)).astype(int)
    face_det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    vfeat = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbs  = face_det.detectMultiScale(gray, 1.1, 4)
        if len(bbs)==0:
            continue
        x,y,w,h = max(bbs, key=lambda r: r[2]*r[3])
        face = gray[y:y+h, x:x+w]
        face = eye_align(face)
        face = cv2.resize(face, FACE_SIZE)
        vfeat.append(extract_block_lbp(face))
    cap.release()
    if not vfeat:
        raise RuntimeError("无人脸帧")
    vvec = np.mean(np.vstack(vfeat), axis=0)          # 944

    # ---- 音频 39 维 ----
    aud = read_audio(vpath)
    mfcc = librosa.feature.mfcc(aud, sr=AUDIO_SR, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    T = mfcc.shape[1]
    feats = np.vstack([mfcc, d1, d2]).T               # (T,39)
    if T < AUDIO_FRAMES:
        feats = librosa.util.fix_length(feats, AUDIO_FRAMES, axis=0)
    else:
        feats = feats[:AUDIO_FRAMES]
    avec = feats.mean(axis=0).astype(np.float32)      # 39
    return np.hstack([vvec, avec])

def process_split(vdir: str, label_arr: np.ndarray, save_to: str):
    out = []
    vids = glob.glob(os.path.join(vdir, "Freeform", "*.mp4")) + \
           glob.glob(os.path.join(vdir, "Northwind", "*.mp4"))
    vids.sort(key=lambda p: (int(re.match(r".*?(\d+)_", p).group(1)),
                             "Freeform" not in p))
    logger.info(f"{vdir} 发现 {len(vids)} 个视频")
       # ------ 用 enumerate 代替直用 vid_id ------
    for idx, vpath in enumerate(tqdm(vids, desc=os.path.basename(vdir), ncols=80)):
        name = os.path.splitext(os.path.basename(vpath))[0]  # e.g. '205_1_Freeform'

        # 两条视频共用一个标签 —— 整除2
        person_idx = idx // 2              # 0,0,1,1,2,2,…
        score      = label_arr[person_idx] # label_arr 长度正好是参与者数
        label      = score_to_cls(score)

        try:
            feat = video_feature(vpath)
        except Exception as e:
            logger.warning(f"{name} 跳过：{e}")
            continue

        out.append({
            "feature": torch.from_numpy(feat),
            "label"  : int(label),
            "vid"    : name
        })

    torch.save(out, save_to)
    logger.info(f"写入 {len(out)} 样本 → {save_to}")

def show_dist(arr, title):
    dist = Counter(score_to_cls(float(s)) for s in arr if s>0)
    logger.info(f"{title} 标签分布: " + ", ".join(f"{k}:{v}" for k,v in dist.items()))

# ===================== 主程序 =====================
if __name__ == "__main__":
    train_lbl = load_labels(TRAIN_LAB)
    dev_lbl   = load_labels(DEV_LAB)
    test_lbl  = load_labels(TEST_LAB)

    show_dist(train_lbl, "Train")
    show_dist(dev_lbl,   "Dev")
    show_dist(test_lbl,  "Test")

    process_split(TRAIN_DIR, train_lbl, os.path.join(OUT_DIR, "train_data.pt"))
    process_split(DEV_DIR,   dev_lbl,   os.path.join(OUT_DIR, "develop_data.pt"))
    process_split(TEST_DIR,  test_lbl,  os.path.join(OUT_DIR, "test_data.pt"))

    logger.info("预处理完成 ✔")



import torch

# 指定数据路径（使用原始字符串避免转义反斜杠）
train_path = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\逐帧数据处理\train_data.pt"
dev_path   = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\逐帧数据处理\develop_data.pt"
test_path  = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\逐帧数据处理\test_data.pt"

# 加载数据
train_data = torch.load(train_path)
dev_data   = torch.load(dev_path)
test_data  = torch.load(test_path)

# 检查数据格式（例如输出第一个样本的键）
print(train_data[0].keys())  # 期望输出: dict_keys(['feature','label','vid'])
print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(dev_data)}")
print(f"测试集样本数: {len(test_data)}")



import torch

# 将训练集中所有样本的特征堆叠成二维张量 (N_train, 983)
train_features = torch.stack([sample['feature'] for sample in train_data])  # shape: (N_train, 983)
# 计算训练集的均值和标准差 (按列，即每个特征维度)
feat_mean = train_features.mean(dim=0)
feat_std  = train_features.std(dim=0)

# 避免除0错误：将标准差为0的维度设为1（表示该维度在训练集上是常数）
feat_std[feat_std == 0] = 1.0

# 对训练集、验证集和测试集的特征进行标准化（原地修改）
for sample in train_data:
    sample['feature'] = (sample['feature'] - feat_mean) / feat_std
for sample in dev_data:
    sample['feature'] = (sample['feature'] - feat_mean) / feat_std
for sample in test_data:
    sample['feature'] = (sample['feature'] - feat_mean) / feat_std


from torch.utils.data import Dataset, DataLoader

class BDIIDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # 保存样本列表（其中每个元素是字典）
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        # 返回特征和标签张量。忽略 'vid' 字段，因为训练不需要用到
        return sample['feature'], sample['label']

# 构建Dataset对象
train_dataset = BDIIDataset(train_data)
dev_dataset   = BDIIDataset(dev_data)
test_dataset  = BDIIDataset(test_data)

# 使用DataLoader实现批量迭代
batch_size = 32  # 可以根据数据量和硬件调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch.nn as nn

# 定义MLP回归模型
class BDIIPredictor(nn.Module):
    def __init__(self, input_dim=983, hidden1=128, hidden2=64):
        super(BDIIPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)  # 最后一层输出1维（回归预测一个值）
        )
    def forward(self, x):
        return self.net(x)

# 初始化模型和训练设置
model = BDIIPredictor(input_dim=983)
# 检查是否有GPU可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数（均方误差）和优化器（使用Adam）
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


import torch.nn as nn

# 定义MLP回归模型
class BDIIPredictor(nn.Module):
    def __init__(self, input_dim=983, hidden1=128, hidden2=64):
        super(BDIIPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)  # 最后一层输出1维（回归预测一个值）
        )
    def forward(self, x):
        return self.net(x)

# 初始化模型和训练设置
model = BDIIPredictor(input_dim=983)
# 检查是否有GPU可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数（均方误差）和优化器（使用Adam）
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


import numpy as np
import copy

num_epochs = 100          # 训练的最多epoch数上限
patience = 5              # Early Stopping耐心: 连续5个epoch验证集无提升则停止
best_val_loss = float('inf')
best_model_wts = None
no_improve_epochs = 0

for epoch in range(1, num_epochs+1):
    # === 训练阶段 ===
    model.train()  # 切换到训练模式
    total_train_loss = 0.0
    for features, labels in train_loader:
        # 将数据移动到设备
        features = features.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()        # 清空上一步的梯度
        preds = model(features)      # 前向传播得到预测
        loss = criterion(preds.squeeze(), labels)  # 计算当前批次的MSE损失
        loss.backward()             # 反向传播计算梯度
        optimizer.step()            # 更新模型参数
        total_train_loss += loss.item() * features.size(0)  # 累积训练损失总和（方便计算平均损失）

    avg_train_loss = total_train_loss / len(train_dataset)  # 训练集平均损失

    # === 验证阶段 ===
    model.eval()  # 切换到评估模式
    total_val_loss = 0.0
    with torch.no_grad():  # 评估不需要计算梯度
        for features, labels in dev_loader:
            features = features.to(device)
            labels   = labels.to(device)
            preds = model(features)
            val_loss = criterion(preds.squeeze(), labels)
            total_val_loss += val_loss.item() * features.size(0)
    avg_val_loss = total_val_loss / len(dev_dataset)  # 验证集平均损失

    # 输出当前Epoch的损失情况
    print(f"Epoch {epoch}: Train MSE = {avg_train_loss:.4f}, Val MSE = {avg_val_loss:.4f}")

    # === Early Stopping 判定 ===
    if avg_val_loss < best_val_loss:
        # 验证损失取得新的最好成绩，保存模型权重
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝模型参数:contentReference[oaicite:0]{index=0}
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"验证损失连续 {patience} 个epoch未提升，提前停止训练。")
            break

# 恢复验证最佳模型权重
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)


    import math

model.eval()  # 切换模型为评估模式
test_preds = []
test_targets = []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels   = labels.to(device)
        preds = model(features)
        # 将结果搬回CPU并转换为numpy列表，方便后续计算
        test_preds.extend(preds.squeeze().cpu().numpy().tolist())
        test_targets.extend(labels.cpu().numpy().tolist())

# 转换为张量以方便计算指标
test_preds = torch.tensor(test_preds)
test_targets = torch.tensor(test_targets)

# 计算 MSE、MAE
mse = torch.mean((test_preds - test_targets) ** 2).item()
mae = torch.mean(torch.abs(test_preds - test_targets)).item()

# 计算 R^2
target_mean = torch.mean(test_targets)
# 总变异TSS和残差变异RSS
TSS = torch.sum((test_targets - target_mean) ** 2)
RSS = torch.sum((test_targets - test_preds) ** 2)
r2 = 1 - RSS / TSS
r2 = r2.item()

print(f"测试集指标: MSE = {mse:.4f}, MAE = {mae:.4f}, R^2 = {r2:.4f}")

import matplotlib.pyplot as plt

# 将数据转换为 numpy 数组 (确保在CPU上)
y_true = test_targets.numpy()
y_pred = test_preds.numpy()

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, c='orange', alpha=0.7, edgecolors='k', label='samples')
# 绘制 y=x 参考线
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='ideal')
plt.xlabel('真实 BDI-II 分数')
plt.ylabel('预测 BDI-II 分数')
plt.title('真实值 vs 预测值')
plt.legend()
plt.grid(True)
plt.show()  # 在交互环境下显示图形；如在脚本中运行可改为 plt.savefig('scatter.png')


# 保存模型参数
torch.save(model.state_dict(), "BDII_MLP_model.pt")
print("模型已保存为 BDII_MLP_model.pt")
