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

# ===== 步骤A：系统库和第三方库导入 =====
# 步骤A.1：系统基础库导入（并列导入）
import os                            # 文件系统操作
import logging                       # 日志记录
# 步骤A.2：数值计算库导入
import numpy as np                   # 数组计算和数值处理
# 步骤A.3：多媒体处理库导入（并列导入）
import cv2                           # OpenCV图像视频处理
import librosa                       # 音频特征提取
import scipy.io as sio               # MATLAB文件读取
# 步骤A.4：深度学习框架导入
import torch                         # PyTorch主模块
from torch import nn                 # 神经网络模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import torch.optim as optim          # 优化器模块
import torch.nn.functional as F      # 激活函数模块
# 步骤A.5：机器学习工具库导入（并列导入）
from sklearn.preprocessing import StandardScaler    # 数据标准化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 评估指标
# 步骤A.6：可视化库导入
import matplotlib.pyplot as plt      # 绘图库

# ===== 步骤B：全局配置参数设置 =====
# 步骤B.1：数据路径配置（并列配置）
DATA_BASE   = r"C:\AAFujiancankao"                                    # 视频数据根目录
AUDIO_BASE  = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\audio"      # 音频数据根目录

# 步骤B.2：标签文件映射配置
LABEL_FILES = {
    "train":   "train_label.mat",      # 训练集标签文件
    "dev":     "develop_label.mat",    # 验证集标签文件（唯一正确写法）
    "test":    "test_label.mat"        # 测试集标签文件
}

# 步骤B.3：日志系统配置
logging.basicConfig(level=logging.DEBUG,                              # 设置日志级别为DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式

# ===== 步骤C：工具函数定义 =====
# 步骤C.1：路径验证工具函数
def checked_path(p, desc="file"):
    """防呆路径检查工具"""
    # 步骤C.1.1：文件存在性检查
    if not os.path.isfile(p):
        logging.error(f"{desc} 不存在: {p}")
        raise FileNotFoundError(p)
    # 步骤C.1.2：返回有效路径
    return p

# 步骤C.2：标签数据加载函数
def load_labels(label_path):
    """
    Load labels from .mat file. Accepts keys 'label' / 'labels' /
    first numeric array found.
    """
    # 步骤C.2.1：路径验证
    label_path = checked_path(label_path, "Label file")
    
    # 步骤C.2.2：MATLAB文件数据加载
    mat = sio.loadmat(label_path)
    
    # 步骤C.2.3：数组类型数据筛选
    arrays = {k: v for k, v in mat.items() if isinstance(v, np.ndarray)}
    
    # 步骤C.2.4：标签键值查找（按优先级并列查找）
    if "label" in arrays:                    # 优先查找'label'键
        labels = arrays["label"]
    elif "labels" in arrays:                 # 次优先查找'labels'键
        labels = arrays["labels"]
    else:
        # 步骤C.2.5：备用键值自动检测
        labels = next(iter(arrays.values()))  # 取第一个数组值
    
    # 步骤C.2.6：数据格式标准化和日志记录（并列操作）
    labels = np.asarray(labels).squeeze()                              # 数据格式标准化
    logging.info(f"Loaded labels from {label_path}, shape={labels.shape}")  # 加载信息记录
    return labels

# 步骤C.3：音频特征提取函数
def extract_audio_features(audio_path):
    """
    提取音频的 MFCC 特征（均值 + 标准差）
    """
    try:
        # 步骤C.3.1：音频数据加载
        y, sr = librosa.load(audio_path, sr=None)  # 加载音频信号和采样率
        
        # 步骤C.3.2：MFCC特征计算
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 提取13维MFCC特征
        
        # 步骤C.3.3：统计特征计算和拼接（并列操作）
        feat = np.concatenate([mfcc.mean(1), mfcc.std(1)])  # 计算均值和标准差特征并拼接 (26,)
        return feat
    except Exception as e:
        # 步骤C.3.4：异常处理和默认值返回（并列操作）
        logging.error(f"Audio error {audio_path}: {e}")     # 错误信息记录
        return np.zeros(26, dtype=np.float32)               # 返回零向量默认值

# 步骤C.4：视频特征提取函数
def extract_video_features(video_path):
    """
    提取视频的视觉特征：BGR 均值/标准差 + 边缘密度
    """
    try:
        # 步骤C.4.1：视频文件打开和验证
        cap = cv2.VideoCapture(video_path)  # 打开视频文件
        if not cap.isOpened():
            raise IOError("cannot open video")

        # 步骤C.4.2：视频属性获取（并列获取）
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1  # 总帧数获取
        fps = cap.get(cv2.CAP_PROP_FPS) or 30                   # 帧率获取
        step = int(fps)                                          # 采样步长设置
        
        # 步骤C.4.3：特征容器初始化
        feats = []  # 特征列表初始化
        
        # 步骤C.4.4：视频帧循环处理
        for fid in range(0, frame_cnt, step):
            # 步骤C.4.4.1：帧位置定位和读取
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)  # 设置帧位置
            ok, frm = cap.read()                    # 读取当前帧
            if not ok:
                continue  # 跳过无效帧
            
            # 步骤C.4.4.2：帧预处理
            frm = cv2.resize(frm, (64, 64))  # 帧尺寸标准化
            
            # 步骤C.4.4.3：BGR颜色特征计算
            m, s = cv2.meanStdDev(frm)  # BGR通道均值和标准差计算
            
            # 步骤C.4.4.4：边缘密度特征计算
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)  # 灰度转换
            edge = cv2.Canny(gray, 100, 200)              # Canny边缘检测
            dens = edge.mean()                             # 边缘密度计算
            
            # 步骤C.4.4.5：帧特征拼接
            feats.append(np.hstack([m.flatten(), s.flatten(), dens]))  # 特征向量拼接
        
        # 步骤C.4.5：视频资源释放
        cap.release()
        
        # 步骤C.4.6：视频级特征聚合和返回（并列操作）
        return np.mean(feats, axis=0) if feats else np.zeros(7, dtype=np.float32)  # 特征均值计算或默认值返回
    except Exception as e:
        # 步骤C.4.7：异常处理和默认值返回（并列操作）
        logging.error(f"Video error {video_path}: {e}")  # 错误信息记录
        return np.zeros(7, dtype=np.float32)             # 返回零向量默认值

# ===== 步骤D：数据集和模型类定义 =====
# 步骤D.1：多模态数据集类定义
class MultiModalDataset(Dataset):
    def __init__(self, X, y):
        """多模态数据集初始化"""
        # 步骤D.1.1：数据张量转换（并列转换）
        self.X = torch.tensor(X, dtype=torch.float32)  # 特征数据转换为PyTorch张量
        self.y = torch.tensor(y, dtype=torch.float32)  # 标签数据转换为PyTorch张量
    
    def __len__(self):
        """数据集长度获取"""
        # 步骤D.1.2：数据集大小返回
        return len(self.X)
    
    def __getitem__(self, idx):
        """数据样本索引获取"""
        # 步骤D.1.3：样本数据返回（并列返回）
        return self.X[idx], self.y[idx]  # 特征和标签对返回

# 步骤D.2：多层感知机模型类定义
class MLPModel(nn.Module):
    def __init__(self, in_dim, h1=64, h2=32):
        """MLP模型初始化"""
        super().__init__()
        # 步骤D.2.1：神经网络层定义（按层次顺序定义）
        self.fc1 = nn.Linear(in_dim, h1)  # 第一层：输入层到隐藏层1
        self.fc2 = nn.Linear(h1, h2)      # 第二层：隐藏层1到隐藏层2
        self.fc3 = nn.Linear(h2, 1)       # 第三层：隐藏层2到输出层

    def forward(self, x):
        """前向传播计算"""
        # 步骤D.2.2：前向传播计算（按层次顺序计算）
        x = F.relu(self.fc1(x))  # 第一层计算+ReLU激活
        x = F.relu(self.fc2(x))  # 第二层计算+ReLU激活
        return self.fc3(x)       # 第三层计算（输出层，无激活函数）

# ===== 步骤E：模型训练和评估函数 =====
# 步骤E.1：模型训练函数
def train(model, tr_loader, val_loader, epochs=50, lr=1e-3):
    """模型训练主函数"""
    # 步骤E.1.1：训练组件初始化（并列初始化）
    opt = optim.Adam(model.parameters(), lr=lr)  # Adam优化器初始化
    crit = nn.MSELoss()                          # 均方误差损失函数初始化
    
    # 步骤E.1.2：训练轮次循环
    for ep in range(1, epochs+1):
        # 步骤E.1.2.1：训练阶段
        model.train()  # 设置模型为训练模式
        tr_loss = 0    # 训练损失初始化
        
        # 步骤E.1.2.2：训练批次循环
        for Xb, yb in tr_loader:
            # 步骤E.1.2.2.1：梯度清零
            opt.zero_grad()
            # 步骤E.1.2.2.2：前向传播和损失计算
            loss = crit(model(Xb).squeeze(), yb)
            # 步骤E.1.2.2.3：反向传播
            loss.backward()
            # 步骤E.1.2.2.4：参数更新
            opt.step()
            # 步骤E.1.2.2.5：损失累积
            tr_loss += loss.item()*len(Xb)
        
        # 步骤E.1.2.3：训练损失平均化
        tr_loss /= len(tr_loader.dataset)

        # 步骤E.1.2.4：验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0  # 验证损失初始化
        
        # 步骤E.1.2.5：验证批次循环（无梯度计算）
        with torch.no_grad():
            for Xb, yb in val_loader:
                # 步骤E.1.2.5.1：验证损失累积
                val_loss += crit(model(Xb).squeeze(), yb).item()*len(Xb)
        
        # 步骤E.1.2.6：验证损失平均化和日志记录（并列操作）
        val_loss /= len(val_loader.dataset)                                      # 验证损失平均化
        logging.info(f"Epoch {ep:02d} | Train {tr_loss:.4f} | Val {val_loss:.4f}")  # 训练进度记录

# 步骤E.2：模型评估函数
def evaluate(model, X, y):
    """模型评估主函数"""
    # 步骤E.2.1：模型评估模式设置
    model.eval()
    
    # 步骤E.2.2：预测计算（无梯度计算）
    with torch.no_grad():
        # 步骤E.2.2.1：数据转换和预测
        preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    
    # 步骤E.2.3：评估指标计算和输出
    print_scores(y, preds)
    
    # 步骤E.2.4：预测结果返回
    return preds

# 步骤E.3：评估指标计算函数
def print_scores(y_true, y_pred):
    """计算并输出评估指标"""
    # 步骤E.3.1：评估指标计算（并列计算）
    r2  = r2_score(y_true, y_pred)           # R²决定系数计算
    mse = mean_squared_error(y_true, y_pred) # 均方误差计算
    mae = mean_absolute_error(y_true, y_pred)# 平均绝对误差计算
    
    # 步骤E.3.2：评估结果日志输出
    logging.info(f"Test R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

# ===== 步骤F：结果可视化函数 =====
# 步骤F.1：散点图绘制函数
def plot_scatter(y, y_hat, title="Actual vs Predicted"):
    """绘制真实值vs预测值散点图"""
    # 步骤F.1.1：图形画布初始化
    plt.figure(figsize=(5,5))
    
    # 步骤F.1.2：散点图绘制
    plt.scatter(y, y_hat, alpha=.7)
    
    # 步骤F.1.3：理想预测线绘制
    lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]  # 坐标轴范围计算
    plt.plot(lims, lims, 'r--')                                    # 理想预测线（y=x）
    
    # 步骤F.1.4：图形属性设置（并列设置）
    plt.xlabel("Actual")      # X轴标签设置
    plt.ylabel("Predicted")   # Y轴标签设置
    plt.title(title)          # 图形标题设置
    plt.grid(True)            # 网格线显示
    
    # 步骤F.1.5：图形输出（并列输出）
    plt.tight_layout()                # 布局优化
    plt.savefig("scatter_plot.png")   # 图形保存
    plt.show()                        # 图形显示

# ===== 步骤G：主程序执行流程 =====
# 步骤G.1：主程序函数
def main():
    """多模态抑郁症评分预测主程序"""
    # 步骤G.1.1：数据路径配置
    paths = {
        "train": {  # 训练集路径配置
            "video": {  # 视频数据路径（并列配置）
                "free":  os.path.join(DATA_BASE, "train", "Freeform"),   # 自由形式视频路径
                "north": os.path.join(DATA_BASE, "train", "Northwind")   # 北风任务视频路径
            },
            "audio": {  # 音频数据路径（并列配置）
                "free":  os.path.join(AUDIO_BASE, "train", "Freeform"),  # 自由形式音频路径
                "north": os.path.join(AUDIO_BASE, "train", "Northwind")  # 北风任务音频路径
            },
            "label": os.path.join(DATA_BASE, "label", LABEL_FILES["train"])  # 训练集标签路径
        },
        "dev": {    # 验证集路径配置
            "video": {  # 视频数据路径（并列配置）
                "free":  os.path.join(DATA_BASE, "dev", "Freeform"),     # 自由形式视频路径
                "north": os.path.join(DATA_BASE, "dev", "Northwind")     # 北风任务视频路径
            },
            "audio": {  # 音频数据路径（并列配置）
                "free":  os.path.join(AUDIO_BASE, "dev", "Freeform"),    # 自由形式音频路径
                "north": os.path.join(AUDIO_BASE, "dev", "Northwind")    # 北风任务音频路径
            },
            "label": os.path.join(DATA_BASE, "label", LABEL_FILES["dev"])    # 验证集标签路径
        },
        "test": {   # 测试集路径配置
            "video": {  # 视频数据路径（并列配置）
                "free":  os.path.join(DATA_BASE, "test", "Freeform"),    # 自由形式视频路径
                "north": os.path.join(DATA_BASE, "test", "Northwind")    # 北风任务视频路径
            },
            "audio": {  # 音频数据路径（并列配置）
                "free":  os.path.join(AUDIO_BASE, "test", "Freeform"),   # 自由形式音频路径
                "north": os.path.join(AUDIO_BASE, "test", "Northwind")   # 北风任务音频路径
            },
            "label": os.path.join(DATA_BASE, "label", LABEL_FILES["test"])   # 测试集标签路径
        }
    }

    # 步骤G.1.2：标签数据加载（并列加载）
    y_train = load_labels(paths["train"]["label"])  # 训练集标签加载
    y_dev   = load_labels(paths["dev"]["label"])    # 验证集标签加载
    y_test  = load_labels(paths["test"]["label"])   # 测试集标签加载

    # 步骤G.1.3：特征提取函数定义
    def build_features(split):
        """构建指定数据集的特征矩阵"""
        # 步骤G.1.3.1：文件列表获取（并列获取）
        vids_free  = sorted(os.listdir(paths[split]["video"]["free"]))   # 自由形式视频文件列表
        vids_north = sorted(os.listdir(paths[split]["video"]["north"]))  # 北风任务视频文件列表
        auds_free  = sorted(os.listdir(paths[split]["audio"]["free"]))   # 自由形式音频文件列表
        auds_north = sorted(os.listdir(paths[split]["audio"]["north"]))  # 北风任务音频文件列表
        
        # 步骤G.1.3.2：对应标签获取
        y_split = {"train":y_train,"dev":y_dev,"test":y_test}[split]
        
        # 步骤G.1.3.3：特征容器初始化
        feats = []
        
        # 步骤G.1.3.4：样本循环处理
        for i in range(len(y_split)):
            # 步骤G.1.3.4.1：文件路径构建（并列构建）
            v_free  = os.path.join(paths[split]["video"]["free"],  vids_free[i])   # 自由形式视频路径
            v_north = os.path.join(paths[split]["video"]["north"], vids_north[i])  # 北风任务视频路径
            a_free  = os.path.join(paths[split]["audio"]["free"],  auds_free[i])   # 自由形式音频路径
            a_north = os.path.join(paths[split]["audio"]["north"], auds_north[i])  # 北风任务音频路径
            
            # 步骤G.1.3.4.2：多模态特征提取和拼接（按模态并列提取）
            feat = np.concatenate([
                extract_audio_features(a_free),   # 自由形式音频特征提取
                extract_audio_features(a_north),  # 北风任务音频特征提取
                extract_video_features(v_free),   # 自由形式视频特征提取
                extract_video_features(v_north)   # 北风任务视频特征提取
            ])
            
            # 步骤G.1.3.4.3：特征数据类型转换和存储
            feats.append(feat.astype(np.float32))
        
        # 步骤G.1.3.5：特征矩阵构建和返回
        return np.vstack(feats)

    # 步骤G.1.4：数据集特征提取（并列提取）
    X_train = build_features("train")  # 训练集特征提取
    X_dev   = build_features("dev")    # 验证集特征提取
    X_test  = build_features("test")   # 测试集特征提取

    # 步骤G.1.5：数据标准化处理
    scaler = StandardScaler().fit(X_train)  # 基于训练集拟合标准化器
    # 步骤G.1.5.1：数据集标准化（并列标准化）
    X_train = scaler.transform(X_train)  # 训练集标准化
    X_dev   = scaler.transform(X_dev)    # 验证集标准化
    X_test  = scaler.transform(X_test)   # 测试集标准化

    # 步骤G.1.6：数据加载器构建（并列构建）
    train_loader = DataLoader(MultiModalDataset(X_train, y_train), batch_size=8, shuffle=True)   # 训练数据加载器
    dev_loader   = DataLoader(MultiModalDataset(X_dev,   y_dev),   batch_size=8, shuffle=False)  # 验证数据加载器

    # 步骤G.1.7：模型构建和训练
    # 步骤G.1.7.1：模型实例化和信息记录（并列操作）
    model = MLPModel(in_dim=X_train.shape[1])              # MLP模型实例化
    logging.info(f"Model input dim = {X_train.shape[1]}")  # 模型输入维度记录
    
    # 步骤G.1.7.2：模型训练执行
    train(model, train_loader, dev_loader, epochs=50, lr=1e-3)

    # 步骤G.1.8：模型评估执行
    y_pred = evaluate(model, X_test, y_test)

    # 步骤G.1.9：结果可视化
    plot_scatter(y_test, y_pred)

# ===== 步骤H：程序入口点 =====
if __name__ == "__main__":
    # 步骤H.1：主程序执行
    main()



