from errno import EEXIST
from math import e
import pandas as pd
import os
import time

# 禁用FutureWarning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 数据处理函数
def load_and_process_data(file_path, output_path=None, output_filename=None, drop_rows=None, header_replace=None):
    # 数据输入
    print(f"正在读取文件: {file_path}")
    
    # 读取数据
    df = pd.read_excel(file_path, sheet_name=0)
    
    if isinstance(df, dict):
        df = df[list(df.keys())[0]]
    # 删除指定行
    if drop_rows:
        df = df.drop(drop_rows)

    # 替换列头
    if header_replace:
        df.columns = header_replace

    df = df.reset_index(drop=True)
    
    # 填充层空值并处理特殊值 - 修复FutureWarning
    df['层'] = df['层'].ffill()
    df['层'] = df['层'].replace('塔尖', 14)
    df['层'] = pd.to_numeric(df['层'], errors='coerce').astype('Int64')

    # 转换数据类型
    df['点'] = pd.to_numeric(df['点'], errors='coerce').astype('Int64')
    df['x/m'] = pd.to_numeric(df['x/m'], errors='coerce').astype(float)
    df['y/m'] = pd.to_numeric(df['y/m'], errors='coerce').astype(float)
    df['z/m'] = pd.to_numeric(df['z/m'], errors='coerce').astype(float)

    print("Processed DataFrame:")
    print(df)  # 调试输出处理后的DataFrame

    # 保存结果
    if output_path and output_filename:
        os.makedirs(output_path, exist_ok=True)
        
        # 处理权限问题：使用临时文件名
        temp_filename = f"temp_{int(time.time())}_{output_filename}"
        final_path = os.path.join(output_path, output_filename)
        temp_path = os.path.join(output_path, temp_filename)
        
        try:
            df.to_excel(final_path, index=False)
            print(f"已保存: {output_filename}")
        except PermissionError:
            print(f"权限错误：无法直接保存到 {final_path}")
            print("尝试保存到临时文件...")
            df.to_excel(temp_path, index=False)
            print(f"已保存到临时文件: {temp_filename}")
            print("请手动重命名临时文件为目标文件名")
    
    return df

# 主函数
def main():
    # 调用数据处理函数
    input_file = r"D:\OneDrive\桌面\2009.xlsx"  # 替换为实际输入文件路径
    output_dir = r"D:\OneDrive\桌面"
    output_file = "更新数据2009.xlsx"
    
    # 处理数据并获取结果
    processed_data = load_and_process_data(
        file_path=input_file,
        output_path=output_dir,
        output_filename=output_file
    )
    
    # 输出处理结果信息
    print(f"处理完成，共{len(processed_data)}条数据")

if __name__ == "__main__":
    main()







