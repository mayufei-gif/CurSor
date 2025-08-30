# batch_translate_recursive.py
# 功能：递归地把 ROOT/{dev,test,train}/**/*.wav 翻译成中文，
#       同名 .txt 存到 ROOT/txt/{dev,test,train}/
# 依赖：pip install openai-whisper tqdm

import os
import glob
import whisper
from tqdm import tqdm

# ---------- 1. 配置区 ----------
ROOT = "C:/Users/asus/Desktop/2025数学建模中青杯/C/audio"  # <= 你的主路径
SUBSETS = ["dev", "test", "train"]
OUT_ROOT = os.path.join(ROOT, "txt")
MODEL_SIZE = "base"          # tiny / base / small / medium / large
# --------------------------------

model = whisper.load_model(MODEL_SIZE)
print(f"[INFO] Whisper-{MODEL_SIZE} 模型已加载")

for subset in SUBSETS:
    in_root = os.path.join(ROOT, subset)
    out_root = os.path.join(OUT_ROOT, subset)
    os.makedirs(out_root, exist_ok=True)

    # 递归收集 wav 文件（大小写都抓）
    wav_list = glob.glob(os.path.join(in_root, "**", "*.wav"), recursive=True)
    wav_list += glob.glob(os.path.join(in_root, "**", "*.WAV"), recursive=True)

    if not wav_list:
        print(f"[WARN] 在 {in_root} 没找到 .wav，跳过")
        continue

    for wav_path in tqdm(wav_list, desc=f"Translating {subset}", unit="file"):
        txt_name = os.path.splitext(os.path.basename(wav_path))[0] + ".txt"
        txt_path = os.path.join(out_root, txt_name)

        if os.path.exists(txt_path):   # 已翻译的跳过
            continue

        # 识别+翻译
        result = model.transcribe(wav_path, task="translate", language="zh")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"].strip())

    print(f"[DONE] {subset} 处理完毕 —— 输出目录: {out_root}")
