import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 设置字体为 SimHei 支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取Excel文件
file_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1996.xlsx'  # 请将路径替换为你的实际文件路径

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
    raise
except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    raise

# 查找表头所在的行（假设“层”这个词汇在表头）
header_row = df[df.iloc[:, 0] == '层'].index[0] + 1

# 提取表头后的数据并设置列名
df_1986 = df.iloc[header_row:, 0:5]
df_1986.columns = ['层', '点', 'x/m', 'y/m', 'z/m']

# 去除空白行
df_1986 = df_1986.dropna().reset_index(drop=True)

# 过滤只包含数值数据的行（排除其他年份的数据）
df_1986 = df_1986[pd.to_numeric(df_1986['层'], errors='coerce').notnull()]

# 计算每层的中心坐标
centers_1986 = df_1986.groupby('层').agg({'x/m': 'mean', 'y/m': 'mean', 'z/m': 'mean'}).reset_index()

# 输出各层中心坐标结果
print("1986年各层中心坐标：")
print(centers_1986)

# 绘制 x/m 和 y/m 的散点图来表示各层中心的平面位置
plt.figure(figsize=(10, 6))

# 绘制散点图并标注每层层号
plt.scatter(centers_1986['x/m'], centers_1986['y/m'], c=centers_1986['层'], cmap='viridis', s=100)

# 为每个点标注层号
for i, row in centers_1986.iterrows():
    plt.text(row['x/m'], row['y/m'], f"层 {int(row['层'])}", fontsize=12, ha='right')

# 设置图表标题和标签
plt.title('1986年古塔各层中心坐标平面图', fontsize=16)
plt.xlabel('x/m', fontsize=14)
plt.ylabel('y/m', fontsize=14)
plt.colorbar(label='层号')
plt.grid(True)

# 保存2D散点图为图片
plt.savefig(r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1996_xy_scatter.png')  # 请将路径替换为你希望保存的位置
plt.show()

# 绘制数据表格并保存为图片
fig, ax = plt.subplots(figsize=(12, 8))  # 设置图像大小
ax.axis('tight')
ax.axis('off')

# 创建表格
table = ax.table(cellText=df_1986.values, colLabels=df_1986.columns, cellLoc='center', loc='center')

# 设置字体大小
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# 保存表格为图片
plt.savefig(r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1996_table.png')  # 请将路径替换为你希望保存的位置
plt.show()

# 建立三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
sc = ax.scatter(centers_1986['x/m'], centers_1986['y/m'], centers_1986['z/m'], c=centers_1986['层'], cmap='viridis', s=100)

# 标注每层的层号
for i, row in centers_1986.iterrows():
    ax.text(row['x/m'], row['y/m'], row['z/m'], f"层 {int(row['层'])}", fontsize=10, ha='right')

# 设置图表标题和轴标签
ax.set_title('1986年古塔各层中心坐标三维图', fontsize=16)
ax.set_xlabel('x/m', fontsize=12)
ax.set_ylabel('y/m', fontsize=12)
ax.set_zlabel('z/m', fontsize=12)

# 添加颜色条
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('层号')

# 保存三维图为图片
plt.savefig(r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1996_3d_plot.png')  # 保存路径

# 显示图表
plt.show()

# 显示图表并保持窗口打开，直到手动关闭
plt.show(block=True)
