import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import rcParams

# 设置字体为 SimHei 支持中文显示
def setup_chinese_font():
    rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取并预处理Excel文件
def load_and_process_data(file_path, sheet_name=0, drop_rows=None, header_replace=None):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 检查df是否是字典类型，如果是，则获取第一个DataFrame
    if isinstance(df, dict):
        df = df[list(df.keys())[0]]

    # 删除指定行
    if drop_rows:
        df = df.drop(drop_rows)

    # 替换列头
    if header_replace:
        df.columns = header_replace

    df = df.reset_index(drop=True)

    # 填充层空值并处理特殊值
    df['层'] = df['层'].ffill()  # 使用 ffill() 填充空值
    df['层'] = df['层'].replace('塔尖', 14).astype(int)  # 替换“塔尖”为14，并转换为整数

    # 转换数据类型
    df['点'] = df['点'].astype(int)
    df['x/m'] = df['x/m'].astype(float)
    df['y/m'] = df['y/m'].astype(float)
    df['z/m'] = df['z/m'].astype(float)

    print("Processed DataFrame:")
    print(df)  # 调试输出处理后的DataFrame

    return df

# 计算每层的中心点
def calculate_layer_centers(df):
    centers = df.groupby('层').agg({
        'x/m': 'mean',
        'y/m': 'mean',
        'z/m': 'mean'
    }).reset_index()
    return centers

# 绘制三维散点图、连线和填充
def plot_tower(df, colors, line_width=2, fill_alpha=0.3):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 计算并获取每层的中心点
    centers = calculate_layer_centers(df)

    # 按层绘制散点图、连线和填充
    for i, layer in enumerate(df['层'].unique()):
        layer_data = df[df['层'] == layer]

        # 绘制散点图
        ax.scatter(layer_data['x/m'], layer_data['y/m'], layer_data['z/m'],
                   color=colors[i], label=f'层 {layer}', s=60, alpha=0.8)

        # 绘制点与点之间的连线，并设置线条粗细
        ax.plot(np.append(layer_data['x/m'], layer_data['x/m'].iloc[0]),
                np.append(layer_data['y/m'], layer_data['y/m'].iloc[0]),
                np.append(layer_data['z/m'], layer_data['z/m'].iloc[0]),
                color=colors[i], linewidth=line_width)

        # 绘制三角面填充
        vertices = [list(zip(layer_data['x/m'], layer_data['y/m'], layer_data['z/m']))]
        poly = Poly3DCollection(vertices, color=colors[i], alpha=fill_alpha)
        ax.add_collection3d(poly)

    # 绘制每层的中心点
    ax.scatter(centers['x/m'], centers['y/m'], centers['z/m'], color='red', s=100, label='中心点')

    # 设置图表标题和轴标签
    ax.set_title('1996年古塔各点三维散点图与层面连线与填充', fontsize=16)
    ax.set_xlabel('x/m', fontsize=12)
    ax.set_ylabel('y/m', fontsize=12)
    ax.set_zlabel('z/m', fontsize=12)

    # 添加图例
    ax.legend(title='层号')

    # 添加网格以增强视觉效果
    ax.grid(True)

    # 调整背景颜色
    ax.set_facecolor('whitesmoke')

    # 显示图表
    plt.show()

# 主函数，整合所有功能
def main():
    setup_chinese_font()

    # 读取并处理数据
    file_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1996.xlsx'  # 替换为你的实际文件路径
    drop_rows = [0, 1]  # 需要删除的行（从0开始计数）
    header_replace = ['层', '点', 'x/m', 'y/m', 'z/m']  # 替换的列标题

    df = load_and_process_data(file_path, drop_rows=drop_rows, header_replace=header_replace)

    # 自定义颜色映射：为每一层分配不同的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['层'].unique())))

    # 绘制三维图
    plot_tower(df, colors, line_width=5, fill_alpha=0.09)

# 执行主函数
if __name__ == "__main__":
    main()
