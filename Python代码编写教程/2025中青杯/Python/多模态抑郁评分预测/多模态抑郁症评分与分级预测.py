import os  # 操作系统相关
import copy  # 深拷贝最佳权重
import argparse  # 命令行参数解析
import random  # 随机数控制
import numpy as np  # 数值计算
import torch  # 深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import Dataset, DataLoader, random_split  # 数据工具
import torchvision.transforms as transforms  # 图像变换
import torchvision.models as models  # 视觉模型
import cv2  # 视频读取
import librosa  # 音频读取
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor  # 预训练音频模型
import logging  # 日志
from typing import List, Tuple  # 类型标注
from sklearn.metrics import confusion_matrix, classification_report  # 评估指标

# Device configuration
# 设备选择（优先GPU）
设备 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def 设置随机种子(种子: int = 42):
    # 统一设置随机性，保证可复现
    random.seed(种子)
    np.random.seed(种子)
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(种子)
    # 使 cuDNN 行为可复现（可能牺牲少量性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def 初始化日志器(输出目录: str) -> logging.Logger:
    # 初始化统一日志（文件+控制台）
    os.makedirs(输出目录, exist_ok=True)
    日志器 = logging.getLogger('train')
    日志器.setLevel(logging.INFO)
    日志器.handlers.clear()
    格式 = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    文件处理器 = logging.FileHandler(os.path.join(输出目录, 'train.log'), encoding='utf-8')
    文件处理器.setFormatter(格式)
    控制台处理器 = logging.StreamHandler()
    控制台处理器.setFormatter(格式)
    日志器.addHandler(文件处理器)
    日志器.addHandler(控制台处理器)
    return 日志器

# Define image transform for video frames (resize and normalize for ResNet)
# 图像预处理（尺寸、张量化、标准化）
图像变换 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load Wav2Vec2 feature extractor (for audio preprocessing)
# 音频特征提取器（用于批量padding与规范化）
音频特征器 = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')

# Dataset class for multimodal depression data
class DepressionDataset(Dataset):
    def __init__(self, data_list, num_frames=16, logger: logging.Logger = None):
        """
        data_list: list of tuples (video_path, audio_path, depression_score)
        num_frames: number of frames to sample from each video
        """
        # 过滤不存在或不可读的样本
        valid: List[Tuple[str, str, float]] = []
        for vp, ap, score in data_list:
            if isinstance(vp, str) and isinstance(ap, str) and os.path.isfile(vp) and os.path.isfile(ap):
                valid.append((vp, ap, score))
            elif logger is not None:
                logger.warning(f"无效样本，已跳过: video={vp}, audio={ap}")
        if logger is not None and len(valid) == 0:
            logger.error("数据集中无有效样本！")
        self.data_list = valid
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 先从索引取出样本三元组（视频、音频、分数），再按顺序处理视频与音频，最后返回张量与标签
        视频路径, 音频路径, 抑郁分数 = self.data_list[idx]

        # 先：读取视频并均匀抽帧；并列：不足帧数则重复最后一帧
        帧列表 = []
        try:
            视频捕获 = cv2.VideoCapture(视频路径)
        except Exception as e:
            raise RuntimeError(f"Error opening video file {视频路径}: {e}")
        总帧数 = int(视频捕获.get(cv2.CAP_PROP_FRAME_COUNT))
        if 总帧数 <= 0:
            抽帧索引 = None  # 无法获取帧数，退化为遍历读取
        else:
            采样帧数 = min(self.num_frames, 总帧数)
            步长 = 1 if 采样帧数 < self.num_frames else (总帧数 // 采样帧数)
            抽帧索引 = [i * 步长 for i in range(采样帧数)]
        当前帧序 = 0
        已采样 = 0
        while True:
            成功, 帧BGR = 视频捕获.read()
            if not 成功:
                break
            if 抽帧索引 is None or 当前帧序 in 抽帧索引:
                帧RGB = cv2.cvtColor(帧BGR, cv2.COLOR_BGR2RGB)
                帧PIL = transforms.functional.to_pil_image(帧RGB)
                帧张量 = 图像变换(帧PIL)  # 应用图像预处理（先Resize→再ToTensor→最后Normalize）
                帧列表.append(帧张量)
                已采样 += 1
                if 已采样 >= self.num_frames:
                    break
            当前帧序 += 1
        视频捕获.release()

        if len(帧列表) < self.num_frames:
            if len(帧列表) > 0:
                最后一帧 = 帧列表[-1]
                while len(帧列表) < self.num_frames:
                    帧列表.append(最后一帧)
            else:
                # 并列兜底：视频读取失败时，使用零张量填充
                帧列表 = [torch.zeros(3, 224, 224) for _ in range(self.num_frames)]
        帧序列张量 = torch.stack(帧列表)  # 最后：堆叠为[T,3,224,224]

        # 然后：读取音频到16kHz单声道；最后转float32
        波形, 采样率 = librosa.load(音频路径, sr=16000)
        音频数据 = 波形.astype(np.float32)

        # 最后：按阈值将抑郁分数划分为4个等级标签
        if 抑郁分数 < 5:
            类别标签 = 0
        elif 抑郁分数 < 10:
            类别标签 = 1
        elif 抑郁分数 < 15:
            类别标签 = 2
        else:
            类别标签 = 3

        return 帧序列张量, 音频数据, float(抑郁分数), 类别标签

# Custom collate function for DataLoader to handle variable-length audio
def collate_fn(batch):  # 批处理拼接：先堆叠帧→再对音频做padding→最后打包张量
    帧批次列表 = []
    音频批次列表 = []
    分数批次列表 = []
    标签批次列表 = []
    for 帧序列张量, 音频数据, 分数, 标签 in batch:
        帧批次列表.append(帧序列张量)
        音频批次列表.append(音频数据)
        分数批次列表.append(分数)
        标签批次列表.append(标签)
    # 先：堆叠所有帧张量为形状[B,T,3,224,224]
    帧批次 = torch.stack(帧批次列表)
    # 再：使用Wav2Vec2的特征器对变长音频批量padding并归一化
    编码结果 = 音频特征器(音频批次列表, sampling_rate=16000, return_tensors="pt", padding=True)
    音频输入 = 编码结果.input_values           # [B, max_len]
    音频注意力掩码 = 编码结果.attention_mask  # [B, max_len]
    # 最后：转换分数与标签为tensor
    分数批次 = torch.tensor(分数批次列表, dtype=torch.float32)
    标签批次 = torch.tensor(标签批次列表, dtype=torch.long)
    return 帧批次, 音频输入, 音频注意力掩码, 分数批次, 标签批次

def 从CSV加载数据(CSV路径: str):
    # 从CSV读取数据列表，并校验必要列
    import pandas as pd
    数据框 = pd.read_csv(CSV路径)
    必要列 = {'video_path', 'audio_path', 'depression_score'}
    if not 必要列.issubset(数据框.columns):
        缺失 = 必要列 - set(数据框.columns)
        raise ValueError(f"CSV 缺少必需列: {缺失}")
    数据列表 = list(zip(数据框['video_path'], 数据框['audio_path'], 数据框['depression_score']))
    return 数据列表


def main():  # 主程序入口
    parser = argparse.ArgumentParser(description="Multimodal depression score and level prediction")
    parser.add_argument('--csv', type=str, required=True, help='包含 video_path,audio_path,depression_score 的 CSV 路径')
    parser.add_argument('--num_frames', type=int, default=16, help='每个视频采样帧数')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--out_dir', type=str, default='outputs', help='输出目录，用于保存权重与指标')
    parser.add_argument('--save_best', action='store_true', help='保存验证最优权重')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练 (AMP)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 工作线程数')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='梯度裁剪阈值，0则禁用')
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine', 'step'], help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=10, help='StepLR 的步长')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR 的衰减因子')
    parser.add_argument('--unfreeze_resnet', action='store_true', help='微调ResNet参数')
    parser.add_argument('--unfreeze_wav2vec', action='store_true', help='微调Wav2Vec2参数')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径(.pth)')
    # 先：解析命令行参数→再设置随机种子→然后读取CSV数据
    参数 = parser.parse_args()

    设置随机种子(参数.seed)

    数据列表 = 从CSV加载数据(参数.csv)
    if len(数据列表) == 0:
        raise RuntimeError("CSV 中没有样本数据")

    # 同时：初始化日志器（文件+控制台）
    日志器 = 初始化日志器(参数.out_dir)

    # 先：构建数据集并进行样本过滤
    数据集 = DepressionDataset(数据列表, num_frames=参数.num_frames, logger=日志器)
    样本总数 = len(数据集)
    训练数 = int(0.8 * 测试总数) if (测试总数 := 样本总数) else 0  # 8:1:1 划分（避免除零）
    验证数 = int(0.1 * 样本总数)
    测试数 = 样本总数 - 训练数 - 验证数
    训练集, 验证集, 测试集 = random_split(数据集, [训练数, 验证数, 测试数])

    os.makedirs(参数.out_dir, exist_ok=True)  # 确保输出目录存在

    # 然后：创建三类数据加载器（训练/验证/测试）
    训练加载器 = DataLoader(训练集, batch_size=参数.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=参数.num_workers, pin_memory=True)
    验证加载器 = DataLoader(验证集, batch_size=参数.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=参数.num_workers, pin_memory=True)
    测试加载器 = DataLoader(测试集, batch_size=参数.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=参数.num_workers, pin_memory=True)

    # Initialize model, loss functions and optimizer
    # 同时：构建模型并放到设备（可选解冻主干参数）
    模型 = MultimodalDepressionModel(num_classes=4).to(设备)
    if 参数.unfreeze_resnet:
        for p in 模型.resnet.parameters():
            p.requires_grad = True
    if 参数.unfreeze_wav2vec:
        for p in 模型.wav2vec.parameters():
            p.requires_grad = True
    均方误差损失 = nn.MSELoss()  # 回归损失
    交叉熵损失 = nn.CrossEntropyLoss()  # 分类损失
    优化器 = optim.Adam(filter(lambda p: p.requires_grad, 模型.parameters()), lr=参数.lr, weight_decay=参数.weight_decay)
    混合精度缩放器 = torch.cuda.amp.GradScaler(enabled=参数.amp)

    # 学习率调度器
    # 然后：根据参数选择学习率调度策略
    if 参数.scheduler == 'cosine':
        学习率调度器 = optim.lr_scheduler.CosineAnnealingLR(优化器, T_max=参数.epochs)
    elif 参数.scheduler == 'step':
        学习率调度器 = optim.lr_scheduler.StepLR(优化器, step_size=参数.step_size, gamma=参数.gamma)
    else:
        学习率调度器 = None

    # 断点恢复
    # 若提供检查点，则先恢复训练状态
    起始轮次 = 1
    if 参数.resume and os.path.isfile(参数.resume):
        检查点 = torch.load(参数.resume, map_location=设备)
        模型.load_state_dict(检查点['model'])
        优化器.load_state_dict(检查点['optimizer'])
        if 'scaler' in 检查点 and 参数.amp:
            混合精度缩放器.load_state_dict(检查点['scaler'])
        起始轮次 = 检查点.get('epoch', 1)
        最佳验证损失 = 检查点.get('best_val_loss', float('inf'))
        日志器.info(f"从检查点恢复: {参数.resume}, 起始epoch={起始轮次}")

    # Training loop with Early Stopping
    总轮次 = 参数.epochs
    早停耐心 = 参数.patience
    最佳验证损失 = float('inf')
    最佳权重 = copy.deepcopy(模型.state_dict())
    无改进轮数 = 0

    # 训练主循环：先训练一个epoch→再验证→随后步进学习率→最后早停判断与保存检查点
    for 轮次 in range(起始轮次, 总轮次 + 1):
        模型.train()
        模型.resnet.eval()
        模型.wav2vec.eval()
        训练损失累计 = 0.0
        for 帧批次, 音频批次, 音频掩码, 分数批次, 标签批次 in 训练加载器:
            帧批次 = 帧批次.to(设备)
            音频批次 = 音频批次.to(设备)
            音频掩码 = 音频掩码.to(设备)
            分数批次 = 分数批次.to(设备)
            标签批次 = 标签批次.to(设备)
            with torch.cuda.amp.autocast(enabled=参数.amp):
                预测分数, 分类logits = 模型(帧批次, 音频批次, 音频掩码)
                回归损失 = 均方误差损失(预测分数, 分数批次)
                分类损失 = 交叉熵损失(分类logits, 标签批次)
                总损失 = 回归损失 + 分类损失
            优化器.zero_grad()
            混合精度缩放器.scale(总损失).backward()
            if 参数.clip_grad_norm and 参数.clip_grad_norm > 0:
                混合精度缩放器.unscale_(优化器)
                torch.nn.utils.clip_grad_norm_(模型.parameters(), 参数.clip_grad_norm)
            混合精度缩放器.step(优化器)
            混合精度缩放器.update()
            训练损失累计 += 总损失.item() * 帧批次.size(0)
        训练集平均损失 = 训练损失累计 / len(训练加载器.dataset)

        模型.eval()
        验证损失累计 = 0.0
        with torch.no_grad():
            for 帧批次, 音频批次, 音频掩码, 分数批次, 标签批次 in 验证加载器:
                帧批次 = 帧批次.to(设备)
                音频批次 = 音频批次.to(设备)
                音频掩码 = 音频掩码.to(设备)
                分数批次 = 分数批次.to(设备)
                标签批次 = 标签批次.to(设备)
                预测分数, 分类logits = 模型(帧批次, 音频批次, 音频掩码)
                回归损失 = 均方误差损失(预测分数, 分数批次)
                分类损失 = 交叉熵损失(分类logits, 标签批次)
                总损失 = 回归损失 + 分类损失
                验证损失累计 += 总损失.item() * 帧批次.size(0)
        验证集平均损失 = 验证损失累计 / len(验证加载器.dataset)

        日志器.info(f"Epoch {轮次}/{总轮次} - Train Loss: {训练集平均损失:.4f}, Val Loss: {验证集平均损失:.4f}")

        # 学习率步进
        if 学习率调度器 is not None:
            学习率调度器.step()

        if 验证集平均损失 < 最佳验证损失:
            最佳验证损失 = 验证集平均损失
            最佳权重 = copy.deepcopy(模型.state_dict())
            if 参数.save_best:
                最优路径 = os.path.join(参数.out_dir, 'best_model.pth')
                torch.save(最佳权重, 最优路径)
            无改进轮数 = 0
        else:
            无改进轮数 += 1
            if 无改进轮数 >= 早停耐心:
                print("Early stopping triggered.")
                break

        # 最后：保存最近检查点
        检查点 = {
            'epoch': 轮次 + 1,
            'model': 模型.state_dict(),
            'optimizer': 优化器.state_dict(),
            'best_val_loss': 最佳验证损失,
        }
        if 参数.amp:
            检查点['scaler'] = 混合精度缩放器.state_dict()
        torch.save(检查点, os.path.join(参数.out_dir, 'last.pth'))

    # Load best model weights
    模型.load_state_dict(最佳权重)  # 加载最佳权重

    # Test evaluation
    # 测试阶段：先切eval→逐批推理→并列收集回归与分类输出
    模型.eval()
    真实分数列表 = []
    预测分数列表 = []
    总样本 = 0
    正确数 = 0
    真实标签列表 = []
    预测标签列表 = []
    with torch.no_grad():
        for 帧批次, 音频批次, 音频掩码, 分数批次, 标签批次 in 测试加载器:
            帧批次 = 帧批次.to(设备)
            音频批次 = 音频批次.to(设备)
            音频掩码 = 音频掩码.to(设备)
            分数批次 = 分数批次.to(设备)
            标签批次 = 标签批次.to(设备)
            批预测分数, 分类logits = 模型(帧批次, 音频批次, 音频掩码)
            # Regression outputs
            真实分数列表.extend(分数批次.cpu().numpy().tolist())
            预测分数列表.extend(批预测分数.cpu().numpy().tolist())
            # Classification outputs
            预测类别 = torch.argmax(分类logits, dim=1)
            正确数 += (预测类别 == 标签批次.to(设备)).sum().item()
            总样本 += 标签批次.size(0)
            真实标签列表.extend(标签批次.cpu().numpy().tolist())
            预测标签列表.extend(预测类别.cpu().numpy().tolist())
    # Compute regression metrics
    # 先：计算回归指标（MSE/MAE/R²）
    真实分数数组 = np.array(真实分数列表)
    预测分数数组 = np.array(预测分数列表)
    均方误差 = np.mean((真实分数数组 - 预测分数数组) ** 2)
    平均绝对误差 = np.mean(np.abs(真实分数数组 - 预测分数数组))
    # Compute R^2 score
    if 真实分数数组.size > 0:
        残差平方和 = np.sum((真实分数数组 - 预测分数数组) ** 2)
        总离差平方和 = np.sum((真实分数数组 - np.mean(真实分数数组)) ** 2)
        决定系数R2 = 1 - 残差平方和 / 总离差平方和 if 总离差平方和 != 0 else 0.0
    else:
        决定系数R2 = 0.0
    # Compute classification accuracy
    准确率 = 正确数 / 总样本 if 总样本 > 0 else 0.0

    # 然后：打印所有指标到日志
    日志器.info(f"Test MSE: {均方误差:.4f}")
    日志器.info(f"Test MAE: {平均绝对误差:.4f}")
    日志器.info(f"Test R^2: {决定系数R2:.4f}")
    日志器.info(f"Test Classification Accuracy: {准确率*100:.2f}%")

    # 保存最终指标
    try:
        import json
        # 最后：保存指标到JSON
        指标 = {
            'mse': float(均方误差),
            'mae': float(平均绝对误差),
            'r2': float(决定系数R2),
            'accuracy': float(准确率)
        }
        with open(os.path.join(参数.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(指标, f, ensure_ascii=False, indent=2)
    except Exception as e:
        日志器.error(f"保存指标失败: {e}")

    # 导出测试集预测与分类报告
    try:
        import pandas as pd
        # 同时：导出逐样本回归预测到CSV
        预测CSV路径 = os.path.join(参数.out_dir, 'test_predictions.csv')
        数据框 = pd.DataFrame({
            'true_score': 真实分数数组,
            'pred_score': 预测分数数组
        })
        数据框.to_csv(预测CSV路径, index=False, encoding='utf-8')
        日志器.info(f"已导出预测: {预测CSV路径}")

        # 分类报告
        # 最后：导出分类报告与混淆矩阵
        混淆矩阵数组 = confusion_matrix(真实标签列表, 预测标签列表)
        分类报告文本 = classification_report(真实标签列表, 预测标签列表, digits=4)
        with open(os.path.join(参数.out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(分类报告文本)
        np.savetxt(os.path.join(参数.out_dir, 'confusion_matrix.csv'), 混淆矩阵数组, fmt='%d', delimiter=',')
        日志器.info("已导出分类报告与混淆矩阵")
    except Exception as e:
        日志器.error(f"导出预测失败: {e}")

# Define the multimodal model (ResNet + Wav2Vec2 + LSTM + fusion + regression & classification)
class MultimodalDepressionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalDepressionModel, self).__init__()
        # 视觉通道：先加载ResNet18→再替换最终全连接为Identity→然后冻结参数并设为eval
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        for 参数 in self.resnet.parameters():
            参数.requires_grad = False
        self.resnet.eval()

        # 音频通道：先加载Wav2Vec2→然后冻结参数并设为eval
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        for 参数 in self.wav2vec.parameters():
            参数.requires_grad = False
        self.wav2vec.eval()

        # 序列建模：随后用LSTM对Wav2Vec2的序列特征进行处理
        self.audio_lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1,
                                  batch_first=True, dropout=0.0, bidirectional=False)
        # 融合与输出：最后做特征拼接→Dropout→全连接→分支输出回归与分类
        self.dropout = nn.Dropout(p=0.5)
        self.fc_fusion = nn.Linear(512 + 128, 128)
        self.fc_regression = nn.Linear(128, 1)
        self.fc_classification = nn.Linear(128, num_classes)

    def forward(self, 帧批次, 音频输入, 音频注意力掩码):
        # 输入：帧批次[B,T,3,224,224]；音频输入[B, L]；注意力掩码[B, L]
        批大小, 采样帧数, _, _, _ = 帧批次.size()
        # 先：将(B,T,3,224,224)展平成(B*T,3,224,224)，以并列方式通过ResNet提取特征
        展平帧 = 帧批次.view(批大小 * 采样帧数, 3, 224, 224)
        with torch.no_grad():
            视觉展平特征 = self.resnet(展平帧)  # [B*T,512]
        # 再：恢复为[B,T,512]并对时间维做平均得到视频特征
        视觉序列特征 = 视觉展平特征.view(批大小, 采样帧数, -1)
        视频特征 = 视觉序列特征.mean(dim=1)  # [B,512]

        # 同时：用Wav2Vec2提取音频序列特征（不参与梯度）
        with torch.no_grad():
            音频输出 = self.wav2vec(音频输入, attention_mask=音频注意力掩码)
        音频序列特征 = 音频输出.last_hidden_state  # [B,S,768]
        # 然后：通过LSTM提取最后时刻的隐藏状态作为音频聚合特征
        _, (隐藏状态, _) = self.audio_lstm(音频序列特征)
        音频特征 = 隐藏状态[-1]  # [B,128]

        # 最后：拼接视频与音频特征→Dropout→全连接→再Dropout→输出回归与分类
        融合特征 = torch.cat([视频特征, 音频特征], dim=1)  # [B,640]
        中间 = self.dropout(融合特征)
        中间 = torch.relu(self.fc_fusion(中间))
        中间 = self.dropout(中间)
        回归输出 = self.fc_regression(中间).squeeze(1)
        分类logits = self.fc_classification(中间)
        return 回归输出, 分类logits

    # (training and evaluation moved into main())


if __name__ == "__main__":
    main()
