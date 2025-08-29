import sys
print("Python exe:", sys.executable)
print("sys.path:", sys.path)
import os
import cv2
import subprocess
import numpy as np
import soundfile as sf
import librosa
import torch
from moviepy.editor import VideoFileClip

# 配置参数
TARGET_SR = 16000        # 目标采样率 (Hz)
N_MFCC = 13              # 提取MFCC的个数
USE_MOVIEPY = True       # 是否优先使用 MoviePy 提取音频

def robust_load_audio(video_path, sr=TARGET_SR):
    """
    尝试以多种方式加载视频文件的音频部分，返回音频波形数据和采样率。
    首先使用 moviepy，其次使用 librosa (内部包含 soundfile 和 ffmpeg 支持)。
    如视频无音频或所有方法失败，返回 (None, None)。
    """
    audio_data = None
    audio_sr = None

    # 方法1：使用 MoviePy 提取音频
    if USE_MOVIEPY:
        try:
            clip = VideoFileClip(video_path)
            audioclip = clip.audio
            if audioclip is None:
                raise ValueError("No audio track found in video.")
            # 将音频转换为 numpy 数组 (浮点型，在-1~1范围)，并重采样到目标采样率
            audio_data = audioclip.to_soundarray(fps=sr)
            audio_sr = sr
        except Exception as e:
            print(f"[Warning] MoviePy audio extraction failed for {video_path}: {e}")
            audio_data, audio_sr = None, None
        finally:
            # 确保资源释放
            try:
                clip.close()
            except Exception:
                pass

    # 方法2：使用 librosa 加载音频
    if audio_data is None:
        try:
            # librosa.load 若 soundfile 无法读取压缩音频，将调用 ffmpeg 解码:contentReference[oaicite:4]{index=4}
            y, audio_sr = librosa.load(video_path, sr=sr, mono=True)
            if y.size == 0:
                # 若读取到空数组，视为无音频轨道
                raise ValueError("No audio data could be loaded (possible missing audio track).")
            audio_data = y  # audio_data 为 1维 numpy 数组 (float32)
        except Exception as e:
            print(f"[Warning] librosa audio load failed for {video_path}: {e}")
            audio_data, audio_sr = None, None

    # 方法3：使用 ffmpeg 提取到临时文件再读取
    if audio_data is None:
        tmp_wav = "temp_audio.wav"
        try:
            # 调用 ffmpeg 提取音频到临时 WAV 文件
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", str(sr),
                tmp_wav
            ]
            # 将 stderr 重定向，避免控制台大量输出；check=True 确保出错时抛异常
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # 读取提取的 WAV 文件
            audio_data, audio_sr = sf.read(tmp_wav, dtype='float32')
        except Exception as e:
            print(f"[Error] ffmpeg extraction failed for {video_path}: {e}")
            audio_data, audio_sr = None, None
        finally:
            # 删除临时文件
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    return audio_data, audio_sr

def compute_LBP_hist(gray_image):
    """
    计算单帧灰度图的 LBP 特征直方图 (256维)。
    使用半径1、8邻域基本LBP模式。忽略图像边界像素（不计算LBP）。
    """
    img = gray_image.astype(np.uint8)
    # 定义相对于中心像素的8个邻居坐标（按顺时针方向依次为：左上, 上, 右上, 右, 右下, 下, 左下, 左）
    center = img[1:-1, 1:-1]       # 中心区域 (排除边界)
    # 8个邻居切片
    top_left     = img[0:-2, 0:-2]
    top          = img[0:-2, 1:-1]
    top_right    = img[0:-2, 2:  ]
    right        = img[1:-1, 2:  ]
    bottom_right = img[2:  , 2:  ]
    bottom       = img[2:  , 1:-1]
    bottom_left  = img[2:  , 0:-2]
    left         = img[1:-1, 0:-2]
    # 逐位比较生成 LBP 码
    lbp_code = np.zeros_like(center, dtype=np.uint8)
    lbp_code |= (top_left     >= center).astype(np.uint8) << 7
    lbp_code |= (top          >= center).astype(np.uint8) << 6
    lbp_code |= (top_right    >= center).astype(np.uint8) << 5
    lbp_code |= (right        >= center).astype(np.uint8) << 4
    lbp_code |= (bottom_right >= center).astype(np.uint8) << 3
    lbp_code |= (bottom       >= center).astype(np.uint8) << 2
    lbp_code |= (bottom_left  >= center).astype(np.uint8) << 1
    lbp_code |= (left         >= center).astype(np.uint8) << 0
    # 计算 0-255 LBP模式出现频数
    hist = np.bincount(lbp_code.ravel(), minlength=256)
    return hist

def process_video(video_path):
    """
    处理单个视频文件，提取 LBP 和 MFCC 特征。
    返回结果字典，包含帧级LBP直方图序列、视频级LBP直方图、MFCC特征矩阵等。
    """
    print(f"\nProcessing video: {video_path}")
    # 1. 读取视频帧并计算每帧 LBP 直方图
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frame_lbp_hists = []  # 保存每帧的LBP直方图
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # 转灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算该帧的 LBP 特征直方图
        hist = compute_LBP_hist(gray)
        frame_lbp_hists.append(hist)
    cap.release()
    frame_lbp_hists = np.array(frame_lbp_hists, dtype=np.int64)
    # 聚合整个视频的 LBP 直方图（可以取平均或求和，这里采用求和表示整体分布）
    if frame_lbp_hists.size > 0:
        video_lbp_hist = frame_lbp_hists.sum(axis=0)
    else:
        video_lbp_hist = np.zeros(256, dtype=np.int64)
    # 2. 提取音频特征 MFCC
    audio_wave, audio_sr = robust_load_audio(video_path, sr=TARGET_SR)
    if audio_wave is None or audio_sr is None:
        print("No audio features extracted (audio missing or unreadable).")
        mfcc_features = None
    else:
        # 若 audio_wave 是多通道，转换为单通道
        if audio_wave.ndim > 1:
            audio_wave = audio_wave.mean(axis=1)  # 多声道转为单声道
        # 计算 MFCC 特征: 返回形状 (N_MFCC, 时间帧数)
        mfcc_features = librosa.feature.mfcc(y=audio_wave, sr=audio_sr, n_mfcc=N_MFCC)
    # 打印一些信息
    print(f"Total frames: {frame_count}, LBP hist per frame shape: {frame_lbp_hists.shape}")
    if mfcc_features is not None:
        print(f"MFCC feature shape (coeff x frames): {mfcc_features.shape}")
    else:
        print("MFCC feature shape: None (no audio)")

    # 转为 PyTorch Tensor（如果需要进一步深度学习处理）
    frame_lbp_tensor = torch.from_numpy(frame_lbp_hists)            # 每帧 LBP 特征序列
    video_lbp_tensor = torch.from_numpy(video_lbp_hist.astype(np.float32))  # 视频级 LBP特征
    mfcc_tensor = torch.from_numpy(mfcc_features.astype(np.float32)) if mfcc_features is not None else None

    return {
        "frame_lbp_hists": frame_lbp_hists,
        "video_lbp_hist": video_lbp_hist,
        "mfcc": mfcc_features,
        "frame_lbp_tensor": frame_lbp_tensor,
        "video_lbp_tensor": video_lbp_tensor,
        "mfcc_tensor": mfcc_tensor
    }

# ===== 主程序示例 =====
if __name__ == "__main__":
    # 将 video_path 设置为待处理的视频文件路径（支持中文、空格等特殊字符）
    video_path = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\C题：忧郁症的双重防线：精准预测与有效治疗\附件：参考数据\train\Freeform\203_1_Freeform_video.mp4"
    features = process_video(video_path)
    # 如需处理整个目录下的多个视频，可以如下批量调用:
    # dir_path = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\附件：参考数据\train\Freeform"
    # for fname in os.listdir(dir_path):
    #     if fname.endswith(".mp4"):
    #         fpath = os.path.join(dir_path, fname)
    #         process_video(fpath)

