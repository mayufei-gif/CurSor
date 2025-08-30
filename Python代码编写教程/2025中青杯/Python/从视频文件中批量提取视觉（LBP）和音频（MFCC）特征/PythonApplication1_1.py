# ===== 步骤A：系统环境检查和库导入 =====
# 步骤A.1：系统环境诊断（调试用）
import sys
print("Python exe:", sys.executable)  # 输出Python解释器路径
print("sys.path:", sys.path)          # 输出Python模块搜索路径

# 步骤A.2：核心功能库导入（按功能分组并列导入）
# 文件系统操作库
import os                            # 文件路径和目录操作
# 视频处理库
import cv2                           # OpenCV视频读取和图像处理
from moviepy.editor import VideoFileClip  # MoviePy视频音频提取
# 系统调用库
import subprocess                    # 外部命令执行（ffmpeg调用）
# 数值计算库
import numpy as np                   # 数组计算和数值处理
# 音频处理库（并列导入）
import soundfile as sf               # 音频文件读写
import librosa                       # 音频特征提取
# 深度学习库
import torch                         # PyTorch张量操作

# ===== 步骤B：全局配置参数设置 =====
# 步骤B.1：音频处理参数配置（并列设置）
TARGET_SR = 16000        # 目标采样率 (Hz) - 音频重采样标准
N_MFCC = 13              # 提取MFCC的个数 - 音频特征维度
# 步骤B.2：处理策略配置
USE_MOVIEPY = True       # 是否优先使用 MoviePy 提取音频 - 音频提取方法选择

# ===== 步骤C：鲁棒音频加载函数定义 =====
def robust_load_audio(video_path, sr=TARGET_SR):
    """
    尝试以多种方式加载视频文件的音频部分，返回音频波形数据和采样率。
    首先使用 moviepy，其次使用 librosa (内部包含 soundfile 和 ffmpeg 支持)。
    如视频无音频或所有方法失败，返回 (None, None)。
    """
    # 步骤C.1：初始化音频数据容器（并列初始化）
    audio_data = None            # 音频波形数据容器
    audio_sr = None              # 音频采样率容器

    # 步骤C.2：方法1 - 使用MoviePy提取音频（优先方法）
    if USE_MOVIEPY:  # 条件分支：检查是否启用MoviePy方法
        try:
            # 步骤C.2.1：视频文件加载
            clip = VideoFileClip(video_path)        # 加载视频文件对象
            # 步骤C.2.2：音频轨道提取
            audioclip = clip.audio                  # 提取音频轨道对象
            # 步骤C.2.3：音频轨道有效性检查
            if audioclip is None:
                raise ValueError("No audio track found in video.")  # 无音频轨道异常
            # 步骤C.2.4：音频数据转换和重采样（并列操作）
            audio_data = audioclip.to_soundarray(fps=sr)  # 转换为numpy数组并重采样
            audio_sr = sr                           # 设置采样率为目标值
        except Exception as e:
            # 步骤C.2.5：异常处理 - MoviePy方法失败
            print(f"[Warning] MoviePy audio extraction failed for {video_path}: {e}")
            audio_data, audio_sr = None, None       # 重置音频数据（并列重置）
        finally:
            # 步骤C.2.6：资源清理 - 确保视频对象释放
            try:
                clip.close()                       # 关闭视频文件对象
            except Exception:
                pass                                # 忽略关闭时的异常

    # 步骤C.3：方法2 - 使用librosa加载音频（备用方法）
    if audio_data is None:  # 条件分支：检查MoviePy方法是否失败
        try:
            # 步骤C.3.1：librosa音频加载（内置ffmpeg支持）
            y, audio_sr = librosa.load(video_path, sr=sr, mono=True)  # 加载并转单声道
            # 步骤C.3.2：音频数据有效性检查
            if y.size == 0:
                # 若读取到空数组，视为无音频轨道
                raise ValueError("No audio data could be loaded (possible missing audio track).")
            # 步骤C.3.3：音频数据赋值
            audio_data = y  # audio_data 为 1维 numpy 数组 (float32)
        except Exception as e:
            # 步骤C.3.4：异常处理 - librosa方法失败
            print(f"[Warning] librosa audio load failed for {video_path}: {e}")
            audio_data, audio_sr = None, None       # 重置音频数据（并列重置）

    # 步骤C.4：方法3 - 使用ffmpeg提取音频（最后备用方法）
    if audio_data is None:  # 条件分支：检查前两种方法是否都失败
        # 步骤C.4.1：临时文件路径设置
        tmp_wav = "temp_audio.wav"          # 临时WAV文件名
        try:
            # 步骤C.4.2：ffmpeg命令构建
            cmd = [
                "ffmpeg", "-y", "-i", video_path,  # 输入视频文件（覆盖模式）
                "-ac", "1", "-ar", str(sr),        # 音频参数：单声道，指定采样率
                tmp_wav                             # 输出临时WAV文件
            ]
            # 步骤C.4.3：ffmpeg命令执行
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # 步骤C.4.4：临时音频文件读取
            audio_data, audio_sr = sf.read(tmp_wav, dtype='float32')  # 读取为float32格式
        except Exception as e:
            # 步骤C.4.5：异常处理 - ffmpeg方法失败
            print(f"[Error] ffmpeg extraction failed for {video_path}: {e}")
            audio_data, audio_sr = None, None   # 最终重置音频数据（并列重置）
        finally:
            # 步骤C.4.6：临时文件清理
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)              # 删除临时WAV文件

    # 步骤C.5：函数结果返回
    return audio_data, audio_sr             # 返回音频数据和采样率（并列返回）

# ===== 步骤D：LBP特征计算函数定义 =====
def compute_LBP_hist(gray_image):
    """
    计算单帧灰度图的 LBP 特征直方图 (256维)。
    使用半径1、8邻域基本LBP模式。忽略图像边界像素（不计算LBP）。
    """
    # 步骤D.1：图像数据类型转换
    img = gray_image.astype(np.uint8)       # 确保为uint8格式
    
    # 步骤D.2：中心像素区域定义
    center = img[1:-1, 1:-1]                # 中心区域 (排除边界像素)
    
    # 步骤D.3：8邻域像素切片定义（按顺时针方向并列定义）
    # 上排邻居（并列定义）
    top_left     = img[0:-2, 0:-2]          # 左上邻居
    top          = img[0:-2, 1:-1]          # 正上邻居
    top_right    = img[0:-2, 2:  ]          # 右上邻居
    # 中排邻居（并列定义）
    right        = img[1:-1, 2:  ]          # 正右邻居
    left         = img[1:-1, 0:-2]          # 正左邻居
    # 下排邻居（并列定义）
    bottom_right = img[2:  , 2:  ]          # 右下邻居
    bottom       = img[2:  , 1:-1]          # 正下邻居
    bottom_left  = img[2:  , 0:-2]          # 左下邻居
    # 步骤D.4：LBP码计算（位运算编码）
    # 步骤D.4.1：LBP码容器初始化
    lbp_code = np.zeros_like(center, dtype=np.uint8)  # 初始化LBP码矩阵
    # 步骤D.4.2：8邻域比较和位运算编码（按位权重并列操作）
    lbp_code |= (top_left     >= center).astype(np.uint8) << 7  # 位7：左上邻居
    lbp_code |= (top          >= center).astype(np.uint8) << 6  # 位6：正上邻居
    lbp_code |= (top_right    >= center).astype(np.uint8) << 5  # 位5：右上邻居
    lbp_code |= (right        >= center).astype(np.uint8) << 4  # 位4：正右邻居
    lbp_code |= (bottom_right >= center).astype(np.uint8) << 3  # 位3：右下邻居
    lbp_code |= (bottom       >= center).astype(np.uint8) << 2  # 位2：正下邻居
    lbp_code |= (bottom_left  >= center).astype(np.uint8) << 1  # 位1：左下邻居
    lbp_code |= (left         >= center).astype(np.uint8) << 0  # 位0：正左邻居
    
    # 步骤D.5：LBP直方图计算
    # 步骤D.5.1：LBP码展平
    lbp_flat = lbp_code.ravel()             # 将2D LBP码矩阵展平为1D
    # 步骤D.5.2：直方图统计
    hist = np.bincount(lbp_flat, minlength=256)  # 计算0-255模式出现频数
    
    # 步骤D.6：函数结果返回
    return hist                             # 返回256维LBP直方图

# ===== 步骤E：视频处理主函数定义 =====
def process_video(video_path):
    """
    处理单个视频文件，提取 LBP 和 MFCC 特征。
    返回结果字典，包含帧级LBP直方图序列、视频级LBP直方图、MFCC特征矩阵等。
    """
    # 步骤E.1：处理开始提示
    print(f"\nProcessing video: {video_path}")
    
    # 步骤E.2：视频文件打开和验证
    cap = cv2.VideoCapture(video_path)      # 创建视频捕获对象
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")  # 文件打开失败异常
    
    # 步骤E.3：视频帧处理初始化（并列初始化）
    frame_lbp_hists = []                    # 保存每帧LBP直方图的列表
    frame_count = 0                         # 帧计数器
    
    # 步骤E.4：视频帧循环处理
    while True:  # 主处理循环
        # 步骤E.4.1：帧读取
        ret, frame = cap.read()             # 读取下一帧
        if not ret:                         # 检查是否读取成功
            break                           # 视频结束，退出循环
        
        # 步骤E.4.2：帧计数更新
        frame_count += 1                    # 增加帧计数
        
        # 步骤E.4.3：帧预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        
        # 步骤E.4.4：LBP特征提取
        hist = compute_LBP_hist(gray)       # 计算当前帧的LBP直方图
        
        # 步骤E.4.5：特征存储
        frame_lbp_hists.append(hist)        # 将直方图添加到列表
    
    # 步骤E.5：视频资源释放
    cap.release()                           # 释放视频捕获对象
    # 步骤E.6：视频LBP特征数据整理
    # 步骤E.6.1：帧级LBP特征转换
    frame_lbp_hists = np.array(frame_lbp_hists, dtype=np.int64)  # 转换为numpy数组
    
    # 步骤E.6.2：视频级LBP特征聚合
    if frame_lbp_hists.size > 0:            # 条件分支：检查是否有有效帧
        video_lbp_hist = frame_lbp_hists.sum(axis=0)  # 按帧求和聚合
    else:
        video_lbp_hist = np.zeros(256, dtype=np.int64)  # 空视频时的零向量
    
    # 步骤E.7：音频特征提取处理
    # 步骤E.7.1：音频数据加载
    audio_wave, audio_sr = robust_load_audio(video_path, sr=TARGET_SR)  # 调用音频加载函数
    
    # 步骤E.7.2：音频有效性检查和MFCC提取
    if audio_wave is None or audio_sr is None:  # 条件分支：检查音频是否成功加载
        # 步骤E.7.2.1：无音频情况处理
        print("No audio features extracted (audio missing or unreadable).")
        mfcc_features = None                # 设置MFCC为空
    else:
        # 步骤E.7.2.2：音频预处理
        if audio_wave.ndim > 1:             # 条件分支：检查是否为多声道
            audio_wave = audio_wave.mean(axis=1)  # 多声道转单声道
        
        # 步骤E.7.2.3：MFCC特征计算
        mfcc_features = librosa.feature.mfcc(y=audio_wave, sr=audio_sr, n_mfcc=N_MFCC)  # 提取MFCC特征
    # 步骤E.8：处理结果信息输出
    # 步骤E.8.1：视频特征信息输出
    print(f"Total frames: {frame_count}, LBP hist per frame shape: {frame_lbp_hists.shape}")
    # 步骤E.8.2：音频特征信息输出（条件输出）
    if mfcc_features is not None:
        print(f"MFCC feature shape (coeff x frames): {mfcc_features.shape}")
    else:
        print("MFCC feature shape: None (no audio)")

    # 步骤E.9：PyTorch张量转换（深度学习兼容性处理）
    # 步骤E.9.1：LBP特征张量转换（并列转换）
    frame_lbp_tensor = torch.from_numpy(frame_lbp_hists)            # 帧级LBP特征序列
    video_lbp_tensor = torch.from_numpy(video_lbp_hist.astype(np.float32))  # 视频级LBP特征
    # 步骤E.9.2：MFCC特征张量转换（条件转换）
    mfcc_tensor = torch.from_numpy(mfcc_features.astype(np.float32)) if mfcc_features is not None else None

    # 步骤E.10：结果字典构建和返回
    return {
        "frame_lbp_hists": frame_lbp_hists,    # numpy格式帧级LBP直方图
        "video_lbp_hist": video_lbp_hist,      # numpy格式视频级LBP直方图
        "mfcc": mfcc_features,                 # numpy格式MFCC特征
        "frame_lbp_tensor": frame_lbp_tensor,  # 张量格式帧级LBP特征
        "video_lbp_tensor": video_lbp_tensor,  # 张量格式视频级LBP特征
        "mfcc_tensor": mfcc_tensor             # 张量格式MFCC特征
    }

# ===== 步骤F：主程序执行部分 =====
if __name__ == "__main__":
    # 步骤F.1：视频文件路径配置
    # 设置待处理的视频文件路径（支持中文、空格等特殊字符）
    video_path = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\C题：忧郁症的双重防线：精准预测与有效治疗\附件：参考数据\train\Freeform\203_1_Freeform_video.mp4"
    
    # 步骤F.2：单个视频处理执行
    features = process_video(video_path)    # 调用视频处理函数
    
    # 步骤F.3：批量处理示例（注释状态）
    # 如需处理整个目录下的多个视频，可以如下批量调用:
    # 步骤F.3.1：目录路径设置
    # dir_path = r"C:\Users\asus\Desktop\2025数学建模中青杯\C\附件：参考数据\train\Freeform"
    # 步骤F.3.2：目录遍历和批量处理
    # for fname in os.listdir(dir_path):     # 遍历目录中的文件
    #     if fname.endswith(".mp4"):         # 条件分支：检查是否为MP4文件
    #         fpath = os.path.join(dir_path, fname)  # 构建完整文件路径
    #         process_video(fpath)            # 处理单个视频文件