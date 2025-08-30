
import os
import glob
import re
import numpy as np
import scipy.io
import cv2
from skimage.feature import local_binary_pattern
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ================= 用户只需改这里 =================
LABEL_MAT_PATH   = r'C:\Users\asus\Desktop\2025数学建模中青杯\C\C题：忧郁症的双重防线：精准预测与有效治疗\附件：参考数据\label\train_label.mat'
VIDEO_ROOT_DIR   = r'C:\Users\asus\Desktop\2025数学建模中青杯\C\C题：忧郁症的双重防线：精准预测与有效治疗\附件：参考数据\train'
FACE_SIZE        = (128, 128)    # 裁剪后人脸尺寸
SAMPLE_INTERVAL  = 30            # 每隔 30 帧采样
TRAIN_RATIO      = 0.7           # 训练 70%  验证 15%  测试 15%
USE_DEEP         = False         # False=LBP+MFCC+SVR  True=ResNet18+MFCC+MLP
# =================================================
# ================ 1. LBP：改成 59-bin + 分块 =================
P, R = 8, 1
lbp = local_binary_pattern(face, P, R, method='uniform')
# === 分 4×4 =16 个 32×32 小块，拼接 16×59 = 944 维  (与 MATLAB CellSize 对齐)  ### FIX
block_feats = []
h, w = FACE_SIZE
for by in range(0, h, 32):
    for bx in range(0, w, 32):
        patch = lbp[by:by+32, bx:bx+32]
        hist, _ = np.histogram(patch.ravel(), bins=59,
                               range=(0, 58), density=True)
        block_feats.append(hist)
vvec = np.hstack(block_feats)           # 944 维

# ================ 2. 眼睛对齐 (OpenCV)  ======================
# 裁剪出 face 后：
eyes = cv2.CascadeClassifier(cv2.data.haarcascades +
                             'haarcascade_eye.xml').detectMultiScale(face)
if len(eyes) >= 2:
    # 取两只眼中心
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (x1,y1,w1,h1), (x2,y2,w2,h2) = eyes
    cx1, cy1 = x1+w1/2, y1+h1/2
    cx2, cy2 = x2+w2/2, y2+h2/2
    angle = np.degrees(np.arctan2(cy2-cy1, cx2-cx1))
    # 旋转对齐
    M = cv2.getRotationMatrix2D((FACE_SIZE[0]/2, FACE_SIZE[1]/2),
                                angle, 1)
    face = cv2.warpAffine(face, M, FACE_SIZE)

# ================ 3. 特征、标签同步检查  =====================
if len(vis_feats)==0:
    continue  # 并且同步删除 y_labels
    # Python:
    mask_valid.append(False)
else:
    mask_valid.append(True)
# 循环结束后
y_labels = y_labels[mask_valid]

# ================ 4. 归一化  ================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)

# ================ 5. SVR 超参数网格搜索  ====================
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[1,10,100], 'gamma':[1e-3,1e-2,1e-1]}
gscv = GridSearchCV(SVR(kernel='rbf', epsilon=0.1),
                    param_grid, cv=5, n_jobs=-1)
gscv.fit(np.vstack([X_tr,X_va]), np.hstack([y_tr,y_va]))
print("最佳参数:", gscv.best_params_)
y_pred = gscv.predict(X_te)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预加载 ResNet-18 特征提取器（如果用深度特征）
if USE_DEEP:
    _resnet = models.resnet18(pretrained=True)
    # 去掉最后的 fc 层，只保留 global avgpool 输出 512-d
    feature_extractor = torch.nn.Sequential(*list(_resnet.children())[:-1]).to(device).eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# 1. 读取标签（BDI-II 分数，一人一行）
mat = scipy.io.loadmat(LABEL_MAT_PATH)
# 排除 MATLAB 自带字段，取第一个非 "__" 开头的字段
key = next(k for k in mat if not k.startswith('__'))
bdi_scores = mat[key].flatten()
print(f"共读取 {len(bdi_scores)} 条标签")

# 2. 构建视频文件列表 (Freeform + Northwind)
ff = glob.glob(os.path.join(VIDEO_ROOT_DIR, 'Freeform', '*.mp4'))
nw = glob.glob(os.path.join(VIDEO_ROOT_DIR, 'Northwind', '*.mp4'))
video_files = ff + nw

# 按 “ID + 片段类型” 排序（Freeform 优先）
def extract_id(path):
    name = os.path.basename(path)
    return int(re.match(r'(\d+)_', name).group(1))
video_files.sort(key=lambda p: (extract_id(p), 'Freeform' not in p))

assert len(video_files) == 2*len(bdi_scores), \
    "视频数必须是标签行数的两倍（每人两段）"

# 3. 标签复制，与视频一一对齐
y_labels = np.repeat(bdi_scores, 2)

# 4. 预初始化
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
all_features = []
print("=== 提取视觉 + 音频特征 ===")

for idx, vpath in enumerate(video_files, 1):
    # —— 4.1 读取音频（librosa 能直接读 mp4 中音轨） ——
    try:
        audio, sr = librosa.load(vpath, sr=None, mono=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
            sr = 16000
    except:
        audio = None

    # —— 4.2 帧循环 & 人脸检测 ——
    cap = cv2.VideoCapture(vpath)
    vis_feats = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % SAMPLE_INTERVAL != 1:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            continue
        # 取最大人脸
        x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_SIZE)

        if USE_DEEP:
            # 深度特征：ResNet-18 pool 层输出 512-D
            rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
            inp = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = feature_extractor(inp)       # (1,512,1,1)
            vvec = out.squeeze().cpu().numpy()    # (512,)
        else:
            # LBP 特征：uniform, P=8,R=1 → global histogram
            lbp = local_binary_pattern(face, P=8, R=1, method='uniform')
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(),
                                   bins=n_bins,
                                   range=(0, n_bins-1),
                                   density=True)
            vvec = hist                           # e.g. ~10-D

        vis_feats.append(vvec)
    cap.release()

    if len(vis_feats) == 0:
        print(f"Warning: {vpath} 无有效帧，跳过")
        continue
    vfeat = np.mean(np.vstack(vis_feats), axis=0)

    # —— 4.3 音频特征：MFCC + Δ + ΔΔ → 39-D ——
    if audio is None:
        afeat = np.zeros(39)
    else:
        mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=13)
        d1   = librosa.feature.delta(mfcc)
        d2   = librosa.feature.delta(mfcc, order=2)
        feats = np.vstack([mfcc, d1, d2]).T   # (T,39)
        afeat = feats.mean(axis=0)            # (39,)

    all_features.append(np.hstack([vfeat, afeat]))
    print(f"已完成 {idx}/{len(video_files)}", end='\r')

X = np.vstack(all_features)   # (2N, Dv+39)
y = y_labels[:X.shape[0]]
print(f"\n特征矩阵维度: {X.shape}")

# 5. 划分训练 / 验证 / 测试
X_tr, X_temp, y_tr, y_temp = train_test_split(
    X, y, train_size=TRAIN_RATIO, random_state=0
)
X_va, X_te, y_va, y_te = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)

# 6. 训练回归模型
if USE_DEEP:
    # MLP 回归
    model = MLPRegressor(
        hidden_layer_sizes=(128,64),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.15
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    model_name = "MLPReg"
else:
    # 支持向量回归
    model = SVR(kernel='rbf', gamma='scale')
    # 将训练+验证一起用来拟合
    model.fit(np.vstack([X_tr, X_va]), np.hstack([y_tr, y_va]))
    y_pred = model.predict(X_te)
    model_name = "SVR"

# 7. 评估 & 可视化
mse = mean_squared_error(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
r2  = r2_score(y_te, y_pred)
print(f"{model_name} 结果:  MSE={mse:.2f}   MAE={mae:.2f}   R²={r2:.3f}")

plt.figure()
plt.scatter(y_te, y_pred, c='blue', s=20)
mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], '--r', linewidth=1.5)
plt.xlabel('真实 BDI-II')
plt.ylabel('预测 BDI-II')
plt.title(f'{model_name} 预测效果')
plt.grid(True)
plt.show()
