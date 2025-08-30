import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 设置字体为 SimHei 支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 读取Excel文件中的数据
file_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1986.xlsx'  # 请将路径替换为你的实际文件路径
df = pd.read_excel(file_path)

# 查找表头所在的行（假设“层”这个词汇在表头）
header_row = df[df.iloc[:, 0] == '层'].index[0] + 1

# 提取表头后的数据并设置列名
df_1986 = df.iloc[header_row:, 0:5]
df_1986.columns = ['层', '点', 'x/m', 'y/m', 'z/m']

# 去除空白行
df_1986 = df_1986.dropna().reset_index(drop=True)

# 过滤只包含数值数据的行（排除其他年份的数据）
df_1986 = df_1986[pd.to_numeric(df_1986['层'], errors='coerce').notnull()]

# 强制转换 x/m 和 y/m 列为数值类型，处理可能的非数值数据
df_1986['x/m'] = pd.to_numeric(df_1986['x/m'], errors='coerce')
df_1986['y/m'] = pd.to_numeric(df_1986['y/m'], errors='coerce')

# 计算每一层与塔底的水平距离差异 (倾斜分析)
df_1986['倾斜距离差'] = np.sqrt((df_1986['x/m'] - df_1986['x/m'].iloc[0])**2 + (df_1986['y/m'] - df_1986['y/m'].iloc[0])**2)

# 计算相邻层的水平偏移量 (弯曲分析)
df_1986['相邻层偏移'] = df_1986['倾斜距离差'].diff().abs()

# 计算塔顶与塔底的水平角度差异 (扭曲分析)
df_1986['角度'] = np.degrees(np.arctan2(df_1986['y/m'] - df_1986['y/m'].iloc[0], df_1986['x/m'] - df_1986['x/m'].iloc[0]))

# 输出分析结果
print("倾斜分析：")
print(df_1986[['层', '倾斜距离差']])

print("\n弯曲分析：")
print(df_1986[['层', '相邻层偏移']])

print("\n扭曲分析：")
print(df_1986[['层', '角度']])


# 可视化分析结果
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# 倾斜分析
ax[0].plot(df_1986['层'], df_1986['倾斜距离差'], marker='o')
ax[0].set_title('倾斜分析')
ax[0].set_xlabel('层')
ax[0].set_ylabel('倾斜距离差 (m)')
ax[0].grid(True)

# 弯曲分析
ax[1].plot(df_1986['层'], df_1986['相邻层偏移'], marker='o', color='orange')
ax[1].set_title('弯曲分析')
ax[1].set_xlabel('层')
ax[1].set_ylabel('相邻层偏移 (m)')
ax[1].grid(True)

# 扭曲分析
ax[2].plot(df_1986['层'], df_1986['角度'], marker='o', color='green')
ax[2].set_title('扭曲分析')
ax[2].set_xlabel('层')
ax[2].set_ylabel('角度 (度)')
ax[2].grid(True)

# 调整布局并显示图形
plt.tight_layout()
plt.show()
