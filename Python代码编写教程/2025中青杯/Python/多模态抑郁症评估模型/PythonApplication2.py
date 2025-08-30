#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025 中青杯 C 题 —— 批量预处理脚本
功能：提取视频分块-LBP(944) + MFCC-Δ-ΔΔ(39) = 983 维特征
      并按 4 类标签保存为 .pt 文件，供后续 PyTorch 训练/推理。
"""

# ========== 步骤1：导入必要的库（按功能分组） ==========
# 1.1 系统和文件操作库（并列导入）
import os, re, glob, logging
from collections import Counter

# 1.2 数值计算和数据处理库（并列导入）
import numpy as np
import scipy.io as sio

# 1.3 计算机视觉和音频处理库（并列导入）
import cv2
import librosa
import torch
from tqdm import tqdm

# ========== 步骤2：条件导入音频处理库（优先级选择） ==========
# 2.1 尝试导入moviepy（优先选择）
try:
    from moviepy.editor import VideoFileClip
    USE_MOVIEPY = True
# 2.2 如果moviepy不可用，则使用soundfile作为备选（并列备选方案）
except ModuleNotFoundError:
    import soundfile as sf
    USE_MOVIEPY = False
    logging.warning("moviepy 不可用，改用 soundfile 读取音轨（需 ffmpeg）")

# ========== 步骤3：配置日志系统 ==========
# 3.1 设置日志基本配置（级别、格式、时间格式的并列配置）
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
# 3.2 创建日志记录器实例
logger = logging.getLogger(__name__)

# ========== 步骤4：配置数据路径 ==========
# 4.1 设置基础目录
BASE_DIR = r"C:"

# 4.2 设置数据根目录
DATA_DIR = os.path.join(
    BASE_DIR,
    r"C:\AAFujiancankao")

# 4.3 设置数据集目录（训练集、验证集、测试集的并列配置）
TRAIN_DIR = os.path.join(DATA_DIR, "train")
DEV_DIR   = os.path.join(DATA_DIR, "dev")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# 4.4 设置标签文件路径（标签目录和各数据集标签文件的并列配置）
LABEL_DIR = os.path.join(DATA_DIR, "label")
TRAIN_LAB = os.path.join(LABEL_DIR, "train_label.mat")
DEV_LAB   = os.path.join(LABEL_DIR, "develop_label.mat")
TEST_LAB  = os.path.join(LABEL_DIR, "test_label.mat")

# 4.5 设置输出目录并创建
OUT_DIR = os.path.join(BASE_DIR, "逐帧数据处理")
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 步骤5：配置特征提取参数 ==========
# 5.1 视频处理参数（并列配置）
MAX_FRAMES    = 64           # 最大帧数
FACE_SIZE     = (128, 128)   # 人脸尺寸

# 5.2 音频处理参数（并列配置）
AUDIO_SR      = 16_000       # 音频采样率
N_MFCC        = 13           # MFCC系数数量：13 × (静音/Δ/ΔΔ) = 39
AUDIO_FRAMES  = 128          # 音频帧对齐长度

# 5.3 LBP特征参数（并列配置）
LBP_P, LBP_R  = 8, 1         # LBP参数：邻域点数和半径
GRID_SIZE     = 4            # 分块网格大小：4×4 分块

# 5.4 特征维度计算（依次计算）
LBP_DIM       = 59 * GRID_SIZE * GRID_SIZE  # LBP维度：59 ×16 = 944
TOTAL_DIM     = LBP_DIM + 39                # 总特征维度：944 + 39 = 983

# ========== 步骤6：定义工具函数 ==========
def load_labels(mat_path: str) -> np.ndarray:
    """读取 .mat 文件 → 一维 score 数组 (索引即受试者 ID)"""
    # 6.1 加载MATLAB文件数据
    data = sio.loadmat(mat_path)
    # 6.2 获取有效变量键（过滤系统变量）
    keys = [k for k in data if not k.startswith("__")]
    # 6.3 验证数据有效性
    if not keys:
        raise ValueError(f"{mat_path} 无有效变量")
    # 6.4 处理包含id和label的结构化数据
    if {"id", "label"}.issubset(keys):
        # 6.4.1 提取ID和标签数据（并列提取）
        idx = np.squeeze(data["id"]).astype(int)
        lab = np.squeeze(data["label"]).astype(float)
        # 6.4.2 创建索引数组并填充标签
        arr = np.zeros(idx.max() + 1, dtype=float)
        arr[idx] = lab
        return arr
    # 6.5 处理简单数组数据（备选方案）
    return np.squeeze(data[keys[0]]).astype(float)

def score_to_cls(score: float) -> int:
    """BDI-II分数转换为4类标签"""
    # 6.6 按阈值分类（依次判断：0-13, 14-19, 20-28, 29+）
    return 0 if score <= 13 else 1 if score <= 19 else 2 if score <= 28 else 3

def extract_block_lbp(gray: np.ndarray) -> np.ndarray:
    """提取分块LBP特征：16块 × 59bin -> 944维"""
    # 6.7 导入LBP特征提取函数
    from skimage.feature import local_binary_pattern
    # 6.8 计算LBP特征图
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    # 6.9 获取图像尺寸并计算分块大小
    h, w = gray.shape
    bh, bw = h // GRID_SIZE, w // GRID_SIZE
    # 6.10 提取各分块的LBP直方图特征
    feats = []
    for by in range(GRID_SIZE):
        for bx in range(GRID_SIZE):
            # 6.10.1 提取当前分块
            patch = lbp[by*bh:(by+1)*bh, bx*bw:(bx+1)*bw]
            # 6.10.2 计算分块直方图
            hist, _ = np.histogram(
                patch.ravel(), bins=59, range=(0, 58), density=True)
            # 6.10.3 添加到特征列表
            feats.append(hist)
    # 6.11 拼接所有分块特征并返回
    return np.hstack(feats).astype(np.float32)

def eye_align(face: np.ndarray) -> np.ndarray:
    """基于双眼检测的人脸对齐矫正"""
    # 6.12 初始化眼部检测器
    eye_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    # 6.13 检测眼部区域
    eyes = eye_det.detectMultiScale(face, 1.1, 3)
    # 6.14 如果检测到至少两只眼睛，进行对齐
    if len(eyes) >= 2:
        # 6.14.1 选择最左和最右的两只眼睛
        eyes = sorted(eyes, key=lambda e: e[0])[:2]  # 最左 & 最右
        # 6.14.2 提取两眼的位置信息（并列提取）
        (x1,y1,w1,h1), (x2,y2,w2,h2) = eyes
        # 6.14.3 计算两眼中心点坐标（并列计算）
        cx1, cy1 = x1+w1/2, y1+h1/2
        cx2, cy2 = x2+w2/2, y2+h2/2
        # 6.14.4 计算两眼连线角度
        ang = np.degrees(np.arctan2(cy2-cy1, cx2-cx1))
        # 6.14.5 构建旋转矩阵
        M = cv2.getRotationMatrix2D((face.shape[1]/2, face.shape[0]/2), ang, 1.0)
        # 6.14.6 应用仿射变换进行对齐
        face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    # 6.15 返回对齐后的人脸
    return face

def read_audio(vpath: str) -> np.ndarray:
    """从视频文件中提取单声道16kHz音频波形"""
    # 6.16 根据可用库选择音频提取方法
    if USE_MOVIEPY:
        # 6.16.1 使用moviepy提取音频（优先方案）
        clip = VideoFileClip(vpath, audio=True)
        # 6.16.2 验证音轨存在性
        if clip.audio is None:
            clip.close(); raise RuntimeError("无音轨")
        # 6.16.3 提取音频数组
        aud = clip.audio.to_soundarray(fps=AUDIO_SR)
        # 6.16.4 释放资源
        clip.close()
    else:
        # 6.16.5 使用soundfile提取音频（备选方案）
        import soundfile as sf
        # 6.16.6 读取音频文件
        aud, sr0 = sf.read(vpath)
        # 6.16.7 如果采样率不匹配，进行重采样
        if sr0 != AUDIO_SR:
            aud = librosa.resample(aud.T, sr0, AUDIO_SR).T
    # 6.17 转换为单声道（如果是立体声）
    if aud.ndim == 2:
        aud = aud.mean(axis=1)
    # 6.18 返回浮点型音频波形
    return aud.astype(float)

def video_feature(vpath: str) -> np.ndarray:
    """提取视频的多模态特征：视觉LBP(944维) + 音频MFCC(39维) = 983维"""
    # ========== 6.19 视觉特征提取部分 ==========
    # 6.19.1 打开视频文件
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise IOError("无法打开")
    # 6.19.2 获取视频帧数并计算采样索引
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES
    idxs = np.linspace(0, tot-1, num=min(tot, MAX_FRAMES)).astype(int)
    # 6.19.3 初始化人脸检测器
    face_det = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # 6.19.4 逐帧提取视觉特征
    vfeat = []
    for i in idxs:
        # 6.19.4.1 定位到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        # 6.19.4.2 读取帧数据
        ok, frame = cap.read()
        if not ok:
            continue
        # 6.19.4.3 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 6.19.4.4 检测人脸区域
        bbs  = face_det.detectMultiScale(gray, 1.1, 4)
        if len(bbs)==0:
            continue
        # 6.19.4.5 选择最大的人脸区域
        x,y,w,h = max(bbs, key=lambda r: r[2]*r[3])
        # 6.19.4.6 裁剪人脸区域
        face = gray[y:y+h, x:x+w]
        # 6.19.4.7 进行眼部对齐
        face = eye_align(face)
        # 6.19.4.8 调整人脸尺寸
        face = cv2.resize(face, FACE_SIZE)
        # 6.19.4.9 提取LBP特征并添加到列表
        vfeat.append(extract_block_lbp(face))
    # 6.19.5 释放视频资源
    cap.release()
    # 6.19.6 验证是否提取到有效特征
    if not vfeat:
        raise RuntimeError("无人脸帧")
    # 6.19.7 计算所有帧的平均LBP特征（944维）
    vvec = np.mean(np.vstack(vfeat), axis=0)

    # ========== 6.20 音频特征提取部分 ==========
    # 6.20.1 提取音频波形
    aud = read_audio(vpath)
    # 6.20.2 计算MFCC特征及其一阶、二阶差分（并列计算）
    mfcc = librosa.feature.mfcc(aud, sr=AUDIO_SR, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    # 6.20.3 获取时间帧数
    T = mfcc.shape[1]
    # 6.20.4 拼接MFCC及其差分特征
    feats = np.vstack([mfcc, d1, d2]).T               # (T,39)
    # 6.20.5 对齐音频特征长度
    if T < AUDIO_FRAMES:
        feats = librosa.util.fix_length(feats, AUDIO_FRAMES, axis=0)
    else:
        feats = feats[:AUDIO_FRAMES]
    # 6.20.6 计算音频特征的时间平均值（39维）
    avec = feats.mean(axis=0).astype(np.float32)
    
    # 6.21 拼接视觉和音频特征并返回（944 + 39 = 983维）
    return np.hstack([vvec, avec])

def process_split(vdir: str, label_arr: np.ndarray, save_to: str):
    """处理指定数据集分割：提取特征并保存为.pt文件"""
    # 6.22 初始化输出列表
    out = []
    # 6.23 搜索视频文件（Freeform和Northwind两种类型的并列搜索）
    vids = glob.glob(os.path.join(vdir, "Freeform", "*.mp4")) + \
           glob.glob(os.path.join(vdir, "Northwind", "*.mp4"))
    # 6.24 按参与者ID和视频类型排序
    vids.sort(key=lambda p: (int(re.match(r".*?(\d+)_", p).group(1)),
                             "Freeform" not in p))
    # 6.25 记录发现的视频数量
    logger.info(f"{vdir} 发现 {len(vids)} 个视频")
    
    # 6.26 逐个处理视频文件
    for idx, vpath in enumerate(tqdm(vids, desc=os.path.basename(vdir), ncols=80)):
        # 6.26.1 提取视频文件名
        name = os.path.splitext(os.path.basename(vpath))[0]  # e.g. '205_1_Freeform'

        # 6.26.2 计算对应的参与者索引（两个视频共用一个标签）
        person_idx = idx // 2              # 0,0,1,1,2,2,…
        # 6.26.3 获取对应的分数和标签（并列获取）
        score      = label_arr[person_idx] # label_arr 长度正好是参与者数
        label      = score_to_cls(score)

        # 6.26.4 提取视频特征（异常处理）
        try:
            feat = video_feature(vpath)
        except Exception as e:
            logger.warning(f"{name} 跳过：{e}")
            continue

        # 6.26.5 构建样本数据并添加到输出列表
        out.append({
            "feature": torch.from_numpy(feat),
            "label"  : int(label),
            "vid"    : name
        })

    # 6.27 保存处理结果并记录日志
    torch.save(out, save_to)
    logger.info(f"写入 {len(out)} 样本 → {save_to}")

def show_dist(arr, title):
    """显示标签分布统计"""
    # 6.28 统计各类别标签数量
    dist = Counter(score_to_cls(float(s)) for s in arr if s>0)
    # 6.29 输出分布信息
    logger.info(f"{title} 标签分布: " + ", ".join(f"{k}:{v}" for k,v in dist.items()))

# ========== 步骤7：主程序执行 ==========
if __name__ == "__main__":
    # 7.1 加载各数据集的标签（并列加载）
    train_lbl = load_labels(TRAIN_LAB)
    dev_lbl   = load_labels(DEV_LAB)
    test_lbl  = load_labels(TEST_LAB)

    # 7.2 显示各数据集的标签分布（并列显示）
    show_dist(train_lbl, "Train")
    show_dist(dev_lbl,   "Dev")
    show_dist(test_lbl,  "Test")

    # 7.3 处理各数据集并保存特征文件（并列处理）
    process_split(TRAIN_DIR, train_lbl, os.path.join(OUT_DIR, "train_data.pt"))
    process_split(DEV_DIR,   dev_lbl,   os.path.join(OUT_DIR, "develop_data.pt"))
    process_split(TEST_DIR,  test_lbl,  os.path.join(OUT_DIR, "test_data.pt"))

    # 7.4 记录预处理完成
    logger.info("预处理完成 ✔")



# ========== 步骤8：数据加载和预处理 ==========
import torch

# 8.1 设置预处理后的数据文件路径（并列设置）
train_path = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\逐帧数据处理\train_data.pt"
dev_path   = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\逐帧数据处理\develop_data.pt"
test_path  = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\逐帧数据处理\test_data.pt"

# 8.2 加载预处理后的数据（并列加载）
train_data = torch.load(train_path)
dev_data   = torch.load(dev_path)
test_data  = torch.load(test_path)

# 8.3 查看数据结构和统计信息（并列查看）
print(train_data[0].keys())  # 期望输出: dict_keys(['feature','label','vid'])
print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(dev_data)}")
print(f"测试集样本数: {len(test_data)}")



# ========== 步骤9：特征标准化 ==========
import torch

# 9.1 提取训练集特征矩阵
train_features = torch.stack([sample['feature'] for sample in train_data])  # shape: (N_train, 983)
# 9.2 基于训练集计算标准化参数（均值和标准差的并列计算）
feat_mean = train_features.mean(dim=0)
feat_std  = train_features.std(dim=0)

# 9.3 处理标准差为零的特征（避免除零错误）
feat_std[feat_std == 0] = 1.0

# 9.4 对所有数据集应用标准化（基于训练集参数的并列标准化）
for sample in train_data:
    sample['feature'] = (sample['feature'] - feat_mean) / feat_std
for sample in dev_data:
    sample['feature'] = (sample['feature'] - feat_mean) / feat_std
for sample in test_data:
    sample['feature'] = (sample['feature'] - feat_mean) / feat_std


# ========== 步骤10：构建数据集和数据加载器 ==========
from torch.utils.data import Dataset, DataLoader

# 10.1 定义BDII数据集类
class BDIIDataset(Dataset):
    def __init__(self, data_list):
        # 10.1.1 保存样本列表（其中每个元素是字典）
        self.data_list = data_list
    def __len__(self):
        # 10.1.2 返回数据集大小
        return len(self.data_list)
    def __getitem__(self, idx):
        # 10.1.3 获取指定索引的样本
        sample = self.data_list[idx]
        # 10.1.4 返回特征和标签张量（忽略 'vid' 字段，因为训练不需要用到）
        return sample['feature'], sample['label']

# 10.2 构建Dataset对象（并列构建）
train_dataset = BDIIDataset(train_data)
dev_dataset   = BDIIDataset(dev_data)
test_dataset  = BDIIDataset(test_data)

# 10.3 设置批次大小
batch_size = 32  # 可以根据数据量和硬件调整
# 10.4 使用DataLoader实现批量迭代（并列创建，训练集打乱，验证和测试集不打乱）
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ========== 步骤11：模型定义和初始化 ==========
import torch.nn as nn

# 11.1 定义BDII预测模型（多层感知机）
class BDIIPredictor(nn.Module):
    def __init__(self, input_dim=983, hidden1=128, hidden2=64):
        super(BDIIPredictor, self).__init__()
        # 11.1.1 构建序列化神经网络（依次定义各层）
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),  # 输入层到第一隐藏层
            nn.ReLU(),                      # 第一层ReLU激活
            nn.Linear(hidden1, hidden2),    # 第一隐藏层到第二隐藏层
            nn.ReLU(),                      # 第二层ReLU激活
            nn.Linear(hidden2, 1)           # 第二隐藏层到输出层（回归任务，输出1维）
        )
    def forward(self, x):
        # 11.1.2 前向传播计算
        return self.net(x)

# 11.2 实例化模型
model = BDIIPredictor(input_dim=983)
# 11.3 设置计算设备（GPU优先，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 11.4 定义损失函数和优化器（并列定义）
criterion = nn.MSELoss()  # 均方误差损失（回归任务）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器


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


# ========== 步骤12：模型训练（含早停机制） ==========
import numpy as np
import copy

# 12.1 设置训练参数（并列设置）
num_epochs = 100          # 训练的最多epoch数上限
patience = 5              # Early Stopping耐心: 连续5个epoch验证集无提升则停止
best_val_loss = float('inf')
best_model_wts = None
no_improve_epochs = 0

# 12.2 训练循环
for epoch in range(1, num_epochs+1):
    # 12.2.1 训练阶段
    model.train()  # 切换到训练模式
    total_train_loss = 0.0
    for features, labels in train_loader:
        # 12.2.1.1 将数据移动到设备（并列移动）
        features = features.to(device)
        labels   = labels.to(device)
        # 12.2.1.2 前向传播和损失计算（依次执行）
        optimizer.zero_grad()        # 清空上一步的梯度
        preds = model(features)      # 前向传播得到预测
        loss = criterion(preds.squeeze(), labels)  # 计算当前批次的MSE损失
        # 12.2.1.3 反向传播和参数更新（依次执行）
        loss.backward()             # 反向传播计算梯度
        optimizer.step()            # 更新模型参数
        # 12.2.1.4 累积训练损失
        total_train_loss += loss.item() * features.size(0)  # 累积训练损失总和（方便计算平均损失）

    # 12.2.2 计算平均训练损失
    avg_train_loss = total_train_loss / len(train_dataset)  # 训练集平均损失

    # 12.2.3 验证阶段
    model.eval()  # 切换到评估模式
    total_val_loss = 0.0
    with torch.no_grad():  # 评估不需要计算梯度
        for features, labels in dev_loader:
            # 12.2.3.1 将数据移动到设备（并列移动）
            features = features.to(device)
            labels   = labels.to(device)
            # 12.2.3.2 前向传播和损失计算（依次执行）
            preds = model(features)
            val_loss = criterion(preds.squeeze(), labels)
            # 12.2.3.3 累积验证损失
            total_val_loss += val_loss.item() * features.size(0)
    # 12.2.4 计算平均验证损失
    avg_val_loss = total_val_loss / len(dev_dataset)  # 验证集平均损失

    # 12.2.5 输出当前Epoch的损失情况
    print(f"Epoch {epoch}: Train MSE = {avg_train_loss:.4f}, Val MSE = {avg_val_loss:.4f}")

    # 12.2.6 Early Stopping 判定
    if avg_val_loss < best_val_loss:
        # 12.2.6.1 验证损失取得新的最好成绩，保存模型权重
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝模型参数
        no_improve_epochs = 0
    else:
        # 12.2.6.2 增加无改善计数器
        no_improve_epochs += 1
        # 12.2.6.3 检查是否触发早停
        if no_improve_epochs >= patience:
            print(f"验证损失连续 {patience} 个epoch未提升，提前停止训练。")
            break

# 12.3 恢复验证最佳模型权重
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)


# ========== 步骤13：模型评估 ==========
import math

# 13.1 模型测试阶段
model.eval()  # 切换模型为评估模式
# 13.2 初始化预测和目标列表（并列初始化）
test_preds = []
test_targets = []
# 13.3 禁用梯度计算进行推理
with torch.no_grad():
    for features, labels in test_loader:
        # 13.3.1 将数据移动到设备（并列移动）
        features = features.to(device)
        labels   = labels.to(device)
        # 13.3.2 模型前向传播
        preds = model(features)
        # 13.3.3 将结果搬回CPU并转换为numpy列表，方便后续计算（并列收集）
        test_preds.extend(preds.squeeze().cpu().numpy().tolist())
        test_targets.extend(labels.cpu().numpy().tolist())

# 13.4 转换为张量以方便计算指标（并列转换）
test_preds = torch.tensor(test_preds)
test_targets = torch.tensor(test_targets)

# 13.5 计算回归评估指标
# 13.5.1 计算 MSE、MAE（并列计算）
mse = torch.mean((test_preds - test_targets) ** 2).item()    # 均方误差
mae = torch.mean(torch.abs(test_preds - test_targets)).item() # 平均绝对误差

# 13.5.2 计算 R^2 决定系数
target_mean = torch.mean(test_targets)  # 目标值均值
# 13.5.3 计算总变异和残差变异（并列计算）
TSS = torch.sum((test_targets - target_mean) ** 2)  # 总平方和
RSS = torch.sum((test_targets - test_preds) ** 2)   # 残差平方和
# 13.5.4 计算决定系数
r2 = 1 - RSS / TSS
r2 = r2.item()

# 13.6 输出测试集评估结果
print(f"测试集指标: MSE = {mse:.4f}, MAE = {mae:.4f}, R^2 = {r2:.4f}")

# ========== 步骤14：结果可视化 ==========
import matplotlib.pyplot as plt

# 14.1 将数据转换为 numpy 数组（确保在CPU上，并列转换）
y_true = test_targets.numpy()
y_pred = test_preds.numpy()

# 14.2 创建散点图可视化预测结果
plt.figure(figsize=(6,6))
# 14.3 绘制预测值与真实值的散点图
plt.scatter(y_true, y_pred, c='orange', alpha=0.7, edgecolors='k', label='samples')
# 14.4 绘制理想预测线（y=x参考线）
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='ideal')
# 14.5 设置图表标签和标题（并列设置）
plt.xlabel('真实 BDI-II 分数')
plt.ylabel('预测 BDI-II 分数')
plt.title('真实值 vs 预测值')
# 14.6 添加图例和网格（并列添加）
plt.legend()
plt.grid(True)
# 14.7 显示图形（在交互环境下显示图形；如在脚本中运行可改为 plt.savefig('scatter.png')）
plt.show()

# ========== 步骤15：模型保存 ==========
# 15.1 保存最终模型参数
torch.save(model.state_dict(), "BDII_MLP_model.pt")
print("模型已保存为 BDII_MLP_model.pt")
