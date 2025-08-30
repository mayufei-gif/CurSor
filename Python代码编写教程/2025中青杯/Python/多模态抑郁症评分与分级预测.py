import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define image transform for video frames (resize and normalize for ResNet)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load Wav2Vec2 feature extractor (for audio preprocessing)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')

# Dataset class for multimodal depression data
class DepressionDataset(Dataset):
    def __init__(self, data_list, num_frames=16):
        """
        data_list: list of tuples (video_path, audio_path, depression_score)
        num_frames: number of frames to sample from each video
        """
        self.data_list = data_list
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_path, audio_path, depression_score = self.data_list[idx]
        # Read video and sample frames
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            raise RuntimeError(f"Error opening video file {video_path}: {e}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            # If frame count not available, read until end
            frame_indices = None
        else:
            # Determine indices of frames to sample uniformly across the video
            sample_count = min(self.num_frames, frame_count)
            # If video has fewer frames than num_frames, we'll duplicate last frame later
            if sample_count < self.num_frames:
                step = 1
            else:
                step = frame_count // sample_count
            frame_indices = [i * step for i in range(sample_count)]
        frame_idx = 0
        sampled = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_indices is None or frame_idx in frame_indices:
                # Convert BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transform to get tensor
                frame_pil = transforms.functional.to_pil_image(frame_rgb)
                frame_tensor = image_transform(frame_pil)
                frames.append(frame_tensor)
                sampled += 1
                if sampled >= self.num_frames:
                    # Collected required frames
                    break
            frame_idx += 1
        cap.release()
        # If not enough frames were collected (video shorter than num_frames), pad by repeating last frame
        if len(frames) < self.num_frames:
            if len(frames) > 0:
                last_frame = frames[-1]
                while len(frames) < self.num_frames:
                    frames.append(last_frame)
            else:
                # In case video file read failed or no frames, create dummy frames of zeros
                frames = [torch.zeros(3, 224, 224) for _ in range(self.num_frames)]
        # Stack frames into a tensor of shape (num_frames, 3, 224, 224)
        frames_tensor = torch.stack(frames)

        # Read audio and resample to 16kHz (mono)
        # Use librosa to load audio
        y, sr = librosa.load(audio_path, sr=16000)
        # librosa.load with sr=16000 will resample if needed and default mix to mono
        audio_data = y.astype(np.float32)

        # Define depression level label based on score (example thresholds)
        # Here we use 4 levels: 0-4 (no depression), 5-9 (mild), 10-14 (moderate), >=15 (severe)
        if depression_score < 5:
            class_label = 0
        elif depression_score < 10:
            class_label = 1
        elif depression_score < 15:
            class_label = 2
        else:
            class_label = 3

        return frames_tensor, audio_data, float(depression_score), class_label

# Custom collate function for DataLoader to handle variable-length audio
def collate_fn(batch):
    frames_batch = []
    audio_batch = []
    score_batch = []
    label_batch = []
    for frames_tensor, audio_data, score, label in batch:
        frames_batch.append(frames_tensor)
        audio_batch.append(audio_data)
        score_batch.append(score)
        label_batch.append(label)
    # Stack all frame tensors into shape (batch, num_frames, 3, 224, 224)
    frames_batch = torch.stack(frames_batch)
    # Use Wav2Vec2FeatureExtractor to pad and normalize audio data
    encoded_inputs = feature_extractor(audio_batch, sampling_rate=16000, return_tensors="pt", padding=True)
    audio_inputs = encoded_inputs.input_values  # shape (batch, max_len)
    audio_attention_mask = encoded_inputs.attention_mask  # shape (batch, max_len)
    # Convert score and label lists to tensors
    score_batch = torch.tensor(score_batch, dtype=torch.float32)
    label_batch = torch.tensor(label_batch, dtype=torch.long)
    return frames_batch, audio_inputs, audio_attention_mask, score_batch, label_batch

# Prepare data (example usage; replace with actual data loading as needed)
# Here data_list should be a list of (video_path, audio_path, score) tuples.
# For example, using a CSV or predefined lists of file paths and scores:
# video_paths = ["data/video1.mp4", "data/video2.mp4", ...]
# audio_paths = ["data/audio1.wav", "data/audio2.wav", ...]
# scores = [score1, score2, ...]
# data_list = list(zip(video_paths, audio_paths, scores))
data_list = []  # TODO: populate with actual (video_path, audio_path, score) data
# If using a CSV file:
# import pandas as pd
# df = pd.read_csv('data_info.csv')
# data_list = list(zip(df['video_path'], df['audio_path'], df['depression_score']))

# Split data into train/val/test
if len(data_list) == 0:
    raise RuntimeError("data_list is empty. Please populate data_list with actual dataset information.")
dataset = DepressionDataset(data_list)
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 4  # batch size tuned for small dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the multimodal model (ResNet + Wav2Vec2 + LSTM + fusion + regression & classification)
class MultimodalDepressionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalDepressionModel, self).__init__()
        # Visual CNN (ResNet) for face frames feature extraction
        self.resnet = models.resnet18(pretrained=True)
        # Replace final FC with identity to get feature vector
        self.resnet.fc = nn.Identity()
        # Freeze ResNet parameters (we will use it as fixed feature extractor)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()  # set ResNet to evaluation mode (freeze batchnorm, etc.)

        # Audio feature extractor (Wav2Vec2)
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        # Freeze Wav2Vec2 parameters as well
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.wav2vec.eval()

        # LSTM to process sequence of audio features from Wav2Vec2
        self.audio_lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1,
                                  batch_first=True, dropout=0.0, bidirectional=False)
        # Fusion and output layers
        self.dropout = nn.Dropout(p=0.5)
        self.fc_fusion = nn.Linear(512 + 128, 128)    # 512-d from ResNet + 128-d from LSTM
        self.fc_regression = nn.Linear(128, 1)        # Regression output (depression score)
        self.fc_classification = nn.Linear(128, num_classes)  # Classification output (depression level)

    def forward(self, frames, audio_inputs, audio_mask):
        # frames: tensor of shape (batch, num_frames, 3, 224, 224)
        # audio_inputs: tensor of shape (batch, seq_len) for audio waveform
        # audio_mask: attention mask for audio_inputs (batch, seq_len)
        batch_size, num_frames, C, H, W = frames.size()
        # Merge batch and frames dimensions to pass through ResNet in one go
        frames_flat = frames.view(batch_size * num_frames, C, H, W)
        # Extract visual features with ResNet (no grad)
        with torch.no_grad():
            vis_feat_flat = self.resnet(frames_flat)  # shape: (batch*num_frames, 512)
        # Reshape back to (batch, num_frames, feat_dim) and average pool over frames
        vis_feat_seq = vis_feat_flat.view(batch_size, num_frames, -1)
        video_feat = vis_feat_seq.mean(dim=1)  # shape: (batch, 512)

        # Extract audio features with Wav2Vec2 (no grad)
        with torch.no_grad():
            audio_outputs = self.wav2vec(audio_inputs, attention_mask=audio_mask)
        audio_feat_seq = audio_outputs.last_hidden_state  # shape: (batch, seq_len_feat, 768)
        # Process audio sequence with LSTM
        lstm_out, (h_n, c_n) = self.audio_lstm(audio_feat_seq)  # h_n shape: (1, batch, 128)
        audio_feat = h_n[-1]  # last LSTM hidden state for each batch element, shape: (batch, 128)

        # Fuse visual and audio features
        fused_feat = torch.cat([video_feat, audio_feat], dim=1)  # shape: (batch, 640)
        # Apply dropout and fully connected layers for prediction
        x = self.dropout(fused_feat)
        x = torch.relu(self.fc_fusion(x))
        x = self.dropout(x)
        # Outputs
        score_pred = self.fc_regression(x).squeeze(1)     # regression output, shape: (batch,)
        class_logits = self.fc_classification(x)          # classification output (logits), shape: (batch, num_classes)
        return score_pred, class_logits

# Initialize model, loss functions and optimizer
model = MultimodalDepressionModel(num_classes=4).to(device)
mse_loss_fn = nn.MSELoss()
ce_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training loop with Early Stopping
num_epochs = 50
patience = 5
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

for epoch in range(1, num_epochs + 1):
    # Training
    model.train()
    # (Keep ResNet and Wav2Vec2 in eval mode even during training)
    model.resnet.eval()
    model.wav2vec.eval()
    running_loss = 0.0
    for frames, audio_inputs, audio_mask, scores, labels in train_loader:
        # Move data to device
        frames = frames.to(device)
        audio_inputs = audio_inputs.to(device)
        audio_mask = audio_mask.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        # Forward pass
        pred_scores, class_logits = model(frames, audio_inputs, audio_mask)
        # Compute losses
        loss_reg = mse_loss_fn(pred_scores, scores)
        loss_cls = ce_loss_fn(class_logits, labels)
        loss = loss_reg + loss_cls
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * frames.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, audio_inputs, audio_mask, scores, labels in val_loader:
            frames = frames.to(device)
            audio_inputs = audio_inputs.to(device)
            audio_mask = audio_mask.to(device)
            scores = scores.to(device)
            labels = labels.to(device)
            pred_scores, class_logits = model(frames, audio_inputs, audio_mask)
            loss_reg = mse_loss_fn(pred_scores, scores)
            loss_cls = ce_loss_fn(class_logits, labels)
            loss = loss_reg + loss_cls
            val_loss += loss.item() * frames.size(0)
    epoch_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Check for improvement for early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Load best model weights
model.load_state_dict(best_model_wts)

# Test evaluation
model.eval()
true_scores = []
pred_scores = []
total = 0
correct = 0
with torch.no_grad():
    for frames, audio_inputs, audio_mask, scores, labels in test_loader:
        frames = frames.to(device)
        audio_inputs = audio_inputs.to(device)
        audio_mask = audio_mask.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        pred_scores_batch, class_logits = model(frames, audio_inputs, audio_mask)
        # Regression outputs
        true_scores.extend(scores.cpu().numpy().tolist())
        pred_scores.extend(pred_scores_batch.cpu().numpy().tolist())
        # Classification outputs
        pred_classes = torch.argmax(class_logits, dim=1)
        correct += (pred_classes == labels.to(device)).sum().item()
        total += labels.size(0)
# Compute regression metrics
true_scores_arr = np.array(true_scores)
pred_scores_arr = np.array(pred_scores)
mse = np.mean((true_scores_arr - pred_scores_arr) ** 2)
mae = np.mean(np.abs(true_scores_arr - pred_scores_arr))
# Compute R^2 score
if true_scores_arr.size > 0:
    ss_res = np.sum((true_scores_arr - pred_scores_arr) ** 2)
    ss_tot = np.sum((true_scores_arr - np.mean(true_scores_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
else:
    r2 = 0.0
# Compute classification accuracy
accuracy = correct / total if total > 0 else 0.0

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R^2: {r2:.4f}")
print(f"Test Classification Accuracy: {accuracy*100:.2f}%")
