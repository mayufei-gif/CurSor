import os
import subprocess

# ——— 配置区 ———
SOURCE_DIRS = [
    r"C:\AAFujiancankao\dev\Freeform",
    r"C:\AAFujiancankao\dev\Northwind",
    r"C:\AAFujiancankao\test\Freeform",
    r"C:\AAFujiancankao\test\Northwind",
    r"C:\AAFujiancankao\train\Freeform",
    r"C:\AAFujiancankao\train\Northwind",
]

OUTPUT_ROOT = r"C:\Users\asus\Desktop\2025数学建模中青杯\C"
TARGET_SR = 16000
# —————————

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def convert_video_to_wav(in_path, out_path, sr=16000):
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vn",
        "-ar", str(sr),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        out_path
    ]
    subprocess.run(cmd, check=True)

def main():
    # 先确保根目录存在
    ensure_dir(OUTPUT_ROOT)

    for src in SOURCE_DIRS:
        # 取出二级目录名：dev/test/train
        parent = os.path.basename(os.path.dirname(src.rstrip("\\/")))
        # 取出三级目录名：Freeform 或 Northwind
        folder = os.path.basename(src.rstrip("\\/"))

        # 组合成 C:\... \C\dev\Freeform 这种结构
        dest_folder = os.path.join(OUTPUT_ROOT, parent, folder)
        ensure_dir(dest_folder)

        for fname in os.listdir(src):
            if fname.lower().endswith(".mp4"):
                in_fp = os.path.join(src, fname)
                base, _ = os.path.splitext(fname)
                out_fp = os.path.join(dest_folder, base + ".wav")

                print(f"Converting:\n  {in_fp}\n→ {out_fp}")
                try:
                    convert_video_to_wav(in_fp, out_fp, sr=TARGET_SR)
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {in_fp}: {e}")

    print("全部转换完成！")

if __name__ == "__main__":
    main()

