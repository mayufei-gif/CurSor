"""
可调参数区（含路径）
=========================
# 请根据实际路径调整以下目录
"""
import os
import copy
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# ========== 可调参数 ==========
BASE_DIR = r"C:\AAFujiancankao"
LABEL_DIR = os.path.join(BASE_DIR, "label")  # 存放 *.mat 标签的文件夹
PT_DIR = os.path.join(BASE_DIR, "pt")       # 存放预提特征的 .pt 文件
SPLITS = ['train', 'dev', 'test']
CONDS = ['Freeform', 'Northwind']

NUM_FRAMES    = 16
BATCH_SIZE    = 4
TRAIN_SPLIT   = 0.8
VAL_SPLIT     = 0.1
# TEST_SPLIT 自动由剩余计算
AUDIO_SR      = 16000
AUDIO_MODEL   = 'facebook/wav2vec2-base-960h'
RESNET_MODEL  = 'resnet18'
PRETRAINED    = True
NUM_CLASSES   = 4
NUM_EPOCHS    = 50
PATIENCE      = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
DROPOUT       = 0.5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理 & 特征提取器
image_transform    = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
feature_extractor  = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL)

# FFmpeg 提取音频
def extract_audio_from_video(path):
    cmd = ['ffmpeg','-i',path,'-f','wav','-acodec','pcm_s16le','-ac','1','-ar',str(AUDIO_SR),'-']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)/32768.0

# 加载标签.mat
labels = {}
for split in SPLITS:
    mat_filename = 'develop_label.mat' if split == 'dev' else f'{split}_label.mat'
    mat_path = os.path.join(LABEL_DIR, mat_filename)
    if not os.path.exists(mat_path):
        raise RuntimeError(f"标签文件不存在: {mat_path}")
    mat = sio.loadmat(mat_path)
    var_names = [k for k in mat.keys() if not k.startswith('__')]
    if not var_names:
        raise KeyError(f"矩阵文件中未找到有效变量: {mat_filename}")
    arr = mat[var_names[0]]
    try:
        arr = arr.flatten().astype(np.float32)
    except Exception:
        raise ValueError(f"无法解析变量 {var_names[0]} 的数据格式")
    labels[split] = arr

# 构建数据记录
records = []
for split in SPLITS:
    for cond in CONDS:
        vd = os.path.join(BASE_DIR, split, cond)
        vids = sorted([f for f in os.listdir(vd) if f.lower().endswith('.mp4')])
        assert len(vids)==len(labels[split]), f"{split}/{cond} 数量与标签不符"
        for i,fn in enumerate(vids):
            records.append((split, cond, i, os.path.join(vd,fn), float(labels[split][i])))

# 确保 PT_DIR 存在
os.makedirs(PT_DIR, exist_ok=True)
# 预提特征并保存 .pt
for split,cond,i,path,score in tqdm(records, desc='Preprocessing videos'):
    pt_name = f"{split}_{cond}_{i}.pt"
    pt_path = os.path.join(PT_DIR, pt_name)
    if not os.path.exists(pt_path):
        frames=[]
        cap=cv2.VideoCapture(path)
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count>0:
            step=count//NUM_FRAMES if count>=NUM_FRAMES else 1
            idxs=[j*step for j in range(min(NUM_FRAMES,count))]
        else:
            idxs=None
        fidx=0; sampled=0
        while True:
            ret,frm=cap.read()
            if not ret: break
            if idxs is None or fidx in idxs:
                from torchvision.transforms.functional import to_pil_image
                pil=to_pil_image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
                frames.append(image_transform(pil))
                sampled+=1
                if sampled>=NUM_FRAMES: break
            fidx+=1
        cap.release()
        if len(frames)<NUM_FRAMES:
            last=frames[-1] if frames else torch.zeros(3,224,224)
            frames+=[last]*(NUM_FRAMES-len(frames))
        frames_tensor=torch.stack(frames)
        audio_arr=extract_audio_from_video(path)
        cls=min(int(score//5),NUM_CLASSES-1)
        torch.save({'frames':frames_tensor,'audio':audio_arr,'score':score,'label':cls},pt_path)

def collate_fn(batch):
    frames_batch, audio_batch, score_batch, label_batch = zip(*batch)
    # 拼接视频帧
    frames = torch.stack(frames_batch)
    # 提取音频特征并返回 attention mask
    enc = feature_extractor(
        list(audio_batch),
        sampling_rate=AUDIO_SR,
        return_tensors='pt',
        padding=True,
        return_attention_mask=True
    )
    audio_inputs = enc['input_values']
    audio_mask = enc['attention_mask']
    scores = torch.tensor(score_batch, dtype=torch.float32)
    labels = torch.tensor(label_batch, dtype=torch.long)
    return frames, audio_inputs, audio_mask, scores, labels

# === 构建 Dataset & DataLoader === 与 DataLoader ===

# 定义基于 .pt 的 Dataset
class DepressionFeatureDataset(Dataset):
    def __init__(self, recs, pt_dir):
        self.recs = recs
        self.pt_dir = pt_dir
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, idx):
        split, cond, i, _, _ = self.recs[idx]
        pt_name = f"{split}_{cond}_{i}.pt"
        data = torch.load(os.path.join(self.pt_dir, pt_name))
        return data['frames'], data['audio'], data['score'], data['label']

# 按 split 构建 DataLoader
train_recs = [r for r in records if r[0] == 'train']
val_recs   = [r for r in records if r[0] == 'dev']
test_recs  = [r for r in records if r[0] == 'test']

train_dataset = DepressionFeatureDataset(train_recs, PT_DIR)
val_dataset   = DepressionFeatureDataset(val_recs, PT_DIR)
test_dataset  = DepressionFeatureDataset(test_recs, PT_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === 模型定义 ===
class MultimodalDepressionModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        self.resnet = getattr(models, RESNET_MODEL)(weights=weights)
        self.resnet.fc = nn.Identity()
        for p in self.resnet.parameters(): p.requires_grad = False
        self.wav2vec = Wav2Vec2Model.from_pretrained(AUDIO_MODEL)
        for p in self.wav2vec.parameters(): p.requires_grad = False
        self.audio_lstm = nn.LSTM(768,128,batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc_fusion = nn.Linear(512+128,128)
        self.fc_reg = nn.Linear(128,1)
        self.fc_cls = nn.Linear(128,num_classes)
    def forward(self, frames, audio_inputs, audio_mask):
        bs,nf,C,H,W=frames.size()
        x=frames.view(bs*nf,C,H,W)
        with torch.no_grad(): vis=self.resnet(x)
        vis_feat=vis.view(bs,nf,-1).mean(1)
        with torch.no_grad(): aud=self.wav2vec(audio_inputs,attention_mask=audio_mask)
        _,(h_n,_)=self.audio_lstm(aud.last_hidden_state)
        feat=torch.cat([vis_feat,h_n[-1]],1)
        feat=self.dropout(feat)
        feat=torch.relu(self.fc_fusion(feat))
        feat=self.dropout(feat)
        return self.fc_reg(feat).squeeze(1), self.fc_cls(feat)

# === 训练与验证 ===
model=MultimodalDepressionModel().to(DEVICE)
opt=optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
loss_mse,loss_ce=nn.MSELoss(),nn.CrossEntropyLoss()

best_loss,early=1e9,0
for epoch in tqdm(range(1,NUM_EPOCHS+1),desc='Epochs'):
    model.train()
    tloss=0
    for frames,audio,mask,scores,labels in train_loader:
        frames,audio,mask,scores,labels=[x.to(DEVICE) for x in (frames,audio,mask,scores,labels)]
        ps,pl=model(frames,audio,mask)
        loss=loss_mse(ps,scores)+loss_ce(pl,labels)
        opt.zero_grad();loss.backward();opt.step()
        tloss+=loss.item()*frames.size(0)
    tloss/=len(train_loader.dataset)

    model.eval()
    vloss=0
    with torch.no_grad():
        for frames,audio,mask,scores,labels in val_loader:
            frames,audio,mask,scores,labels=[x.to(DEVICE) for x in (frames,audio,mask,scores,labels)]
            ps,pl=model(frames,audio,mask)
            loss=loss_mse(ps,scores)+loss_ce(pl,labels)
            vloss+=loss.item()*frames.size(0)
    vloss/=len(val_loader.dataset)
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train Loss: {tloss:.4f} | Val Loss: {vloss:.4f}")
    if vloss<best_loss: best_loss,early=vloss,0;best_wts=copy.deepcopy(model.state_dict())
    else: early+=1
    if early>=PATIENCE:
        print("Early stopping triggered.");break

model.load_state_dict(best_wts)

# === 测试评估 & 输出美观结果 ===
model.eval()
all_t,all_p,correct,total=[],[],0,0
with torch.no_grad():
    for frames,audio,mask,scores,labels in test_loader:
        frames,audio,mask,scores,labels=[x.to(DEVICE) for x in (frames,audio,mask,scores,labels)]
        ps,pl=model(frames,audio,mask)
        all_t+=scores.cpu().tolist();all_p+=ps.cpu().tolist()
        preds=torch.argmax(pl,1)
        correct+=(preds==labels).sum().item();total+=labels.size(0)
mse=np.mean((np.array(all_t)-np.array(all_p))**2)
mae=np.mean(np.abs(np.array(all_t)-np.array(all_p)))
r2=1- np.sum((np.array(all_t)-np.array(all_p))**2)/np.sum((np.array(all_t)-np.mean(all_t))**2)
acc=correct/total
print("\n"+"="*40)
print(f"Test Results:\n MSE: {mse:.4f}\n MAE: {mae:.4f}\n R2: {r2:.4f}\n Accuracy: {acc*100:.2f}%")
print("="*40)

