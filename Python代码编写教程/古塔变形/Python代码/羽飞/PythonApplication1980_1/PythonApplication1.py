# 第一步：导入必要的库（按功能分类并列导入）
# 1.1 数据处理库
import pandas as pd  # 用于Excel文件读取和数据处理
# 1.2 绘图相关库（并列导入三个绘图组件）
import matplotlib.pyplot as plt  # 基础绘图库
from mpl_toolkits.mplot3d import Axes3D  # 三维绘图支持
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 三维多边形集合
# 1.3 数值计算库
import numpy as np  # 数值计算和数组操作
# 1.4 字体配置库
from matplotlib import rcParams  # matplotlib参数配置

# 第二步：中文字体配置函数（解决中文显示问题）
def setup_chinese_font():
    """配置matplotlib支持中文显示的字体设置"""
    # 步骤1：设置中文字体（并列配置两个字体参数）
    # 1.1 设置无衬线字体为黑体（支持中文字符显示）
    rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    # 1.2 设置负号正常显示（避免中文字体导致的负号显示问题）
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 第三步：数据加载和预处理函数（完整的Excel数据处理流程）
def load_and_process_data(file_path, sheet_name=0, drop_rows=None, header_replace=None):
    """读取并预处理Excel文件，返回清洗后的DataFrame"""
    # 步骤1：数据读取
    # 1.1 从Excel文件读取指定工作表的数据
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 步骤2：数据格式检查和转换（条件分支处理）
    # 2.1 检查读取结果是否为字典类型（多工作表情况）
    if isinstance(df, dict):
        # 2.1.1 分支1：字典类型时获取第一个DataFrame
        df = df[list(df.keys())[0]]

    # 步骤3：数据清理（条件性删除和替换操作）
    # 3.1 删除指定的无用行（如果提供了删除行参数）
    if drop_rows:
        # 3.1.1 执行行删除操作
        df = df.drop(drop_rows)

    # 3.2 替换列标题（如果提供了新的列标题）
    if header_replace:
        # 3.2.1 使用新的列标题替换原有标题
        df.columns = header_replace

    # 3.3 重置索引（确保索引连续性）
    df = df.reset_index(drop=True)

    # 步骤4：特殊数据处理（按顺序处理层数据和类型转换）
    # 4.1 层数据处理（先后执行两个层数据操作）
    # 4.1.1 填充层列的空值（使用前向填充方法）
    df['层'] = df['层'].ffill()  # 使用 ffill() 填充空值
    # 4.1.2 处理特殊值并转换数据类型（将"塔尖"替换为14并转为整数）
    df['层'] = df['层'].replace('塔尖', 14).astype(int)  # 替换"塔尖"为14，并转换为整数

    # 4.2 数据类型转换（并列转换四个列的数据类型）
    # 4.2.1 转换点列为整数类型
    df['点'] = df['点'].astype(int)
    # 4.2.2 转换x坐标为浮点类型
    df['x/m'] = df['x/m'].astype(float)
    # 4.2.3 转换y坐标为浮点类型
    df['y/m'] = df['y/m'].astype(float)
    # 4.2.4 转换z坐标为浮点类型
    df['z/m'] = df['z/m'].astype(float)

    # 步骤5：调试输出（并列输出调试信息）
    # 5.1 输出处理完成提示
    print("Processed DataFrame:")
    # 5.2 输出处理后的DataFrame内容
    print(df)  # 调试输出处理后的DataFrame

    # 步骤6：返回处理结果
    return df

# 第四步：中心点计算函数（计算每层的几何中心）
def calculate_layer_centers(df):
    """计算每层点的几何中心坐标"""
    # 步骤1：按层分组并计算中心点坐标（并列计算三个坐标的均值）
    # 1.1 按层分组并聚合计算x、y、z坐标的平均值
    centers = df.groupby('层').agg({
        'x/m': 'mean',  # 计算x坐标平均值
        'y/m': 'mean',  # 计算y坐标平均值
        'z/m': 'mean'   # 计算z坐标平均值
    }).reset_index()  # 重置索引使层号成为普通列
    
    # 步骤2：返回中心点数据
    return centers

# 第五步：三维可视化函数（完整的古塔三维绘图流程）
def plot_tower(df, colors, line_width=2, fill_alpha=0.3):
    """绘制古塔的三维散点图、连线和填充效果"""
    # 步骤1：图形初始化（创建三维绘图环境）
    # 1.1 创建图形对象（设置图形大小）
    fig = plt.figure(figsize=(12, 10))
    # 1.2 添加三维子图（设置三维投影）
    ax = fig.add_subplot(111, projection='3d')

    # 步骤2：中心点计算
    # 2.1 计算并获取每层的几何中心点
    centers = calculate_layer_centers(df)

    # 步骤3：分层绘制循环（按层遍历并绘制各种图形元素）
    for i, layer in enumerate(df['层'].unique()):
        # 3.1 数据筛选：获取当前层的所有点数据
        layer_data = df[df['层'] == layer]

        # 3.2 散点图绘制：绘制当前层的所有点
        ax.scatter(layer_data['x/m'], layer_data['y/m'], layer_data['z/m'],
                   color=colors[i], label=f'层 {layer}', s=60, alpha=0.8)

        # 3.3 连线绘制：连接当前层的所有点形成封闭图形
        # 3.3.1 构建封闭路径（将第一个点添加到末尾形成闭合）
        ax.plot(np.append(layer_data['x/m'], layer_data['x/m'].iloc[0]),
                np.append(layer_data['y/m'], layer_data['y/m'].iloc[0]),
                np.append(layer_data['z/m'], layer_data['z/m'].iloc[0]),
                color=colors[i], linewidth=line_width)

        # 3.4 面填充绘制：为当前层创建三角面填充效果
        # 3.4.1 构建顶点列表（用于创建多边形面）
        vertices = [list(zip(layer_data['x/m'], layer_data['y/m'], layer_data['z/m']))]
        # 3.4.2 创建三维多边形集合并添加到图形中
        poly = Poly3DCollection(vertices, color=colors[i], alpha=fill_alpha)
        ax.add_collection3d(poly)

    # 步骤4：中心点绘制
    # 4.1 绘制每层的几何中心点（用红色突出显示）
    ax.scatter(centers['x/m'], centers['y/m'], centers['z/m'], color='red', s=100, label='中心点')

    # 步骤5：图形美化设置（并列设置多个图形属性）
    # 5.1 设置图表标题
    ax.set_title('1986年古塔各点三维散点图与层面连线与填充', fontsize=16)
    # 5.2 设置坐标轴标签（并列设置三个轴的标签）
    # 5.2.1 设置x轴标签
    ax.set_xlabel('x/m', fontsize=12)
    # 5.2.2 设置y轴标签
    ax.set_ylabel('y/m', fontsize=12)
    # 5.2.3 设置z轴标签
    ax.set_zlabel('z/m', fontsize=12)

    # 5.3 添加图例
    ax.legend(title='层号')

    # 5.4 添加网格以增强视觉效果
    ax.grid(True)

    # 5.5 调整背景颜色
    ax.set_facecolor('whitesmoke')

    # 步骤6：显示图表
    plt.show()

# 第六步：主函数（整合所有功能的程序入口）
def main():
    """主函数，整合所有功能模块"""
    # 步骤1：环境配置
    # 1.1 设置中文字体支持
    setup_chinese_font()

    # 步骤2：数据处理配置（并列设置三个数据处理参数）
    # 2.1 设置Excel文件路径
    file_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\1996.xlsx'  # 替换为你的实际文件路径
    # 2.2 设置需要删除的行（从0开始计数）
    drop_rows = [0, 1]  # 需要删除的行（从0开始计数）
    # 2.3 设置替换的列标题
    header_replace = ['层', '点', 'x/m', 'y/m', 'z/m']  # 替换的列标题

    # 步骤3：数据加载和处理
    # 3.1 读取并处理Excel数据
    df = load_and_process_data(file_path, drop_rows=drop_rows, header_replace=header_replace)

    # 步骤4：可视化配置
    # 4.1 自定义颜色映射：为每一层分配不同的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['层'].unique())))

    # 步骤5：三维图形绘制
    # 5.1 绘制古塔三维可视化图形（设置线宽和透明度参数）
    plot_tower(df, colors, line_width=5, fill_alpha=0.09)

# 第七步：程序执行入口（条件性执行主函数）
if __name__ == "__main__":
    # 当脚本直接运行时执行主函数
    main()