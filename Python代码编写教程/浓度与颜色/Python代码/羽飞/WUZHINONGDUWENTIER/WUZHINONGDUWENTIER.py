# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import numpy as np  # 数值计算和数组操作
import pandas as pd  # 数据处理和分析
import matplotlib.pyplot as plt  # 数据可视化和图表绘制
import seaborn as sns  # 高级统计图表绘制
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 模型评估指标

# ========== 第二步：定义二氧化硫实验数据 ==========
# 以下数据定义为并列关系，构成完整的实验数据集

# 2.1 定义物质类别数组（二氧化硫检测实验）
substance = np.array([
    6, 6, 6, 6, 6,  # 二氧化硫 - 水（对照组）
    6, 6, 6,  # 二氧化硫 - 20 ppm
    6, 6, 6, 6,  # 二氧化硫 - 30 ppm
    6, 6, 6,  # 二氧化硫 - 50 ppm
    6, 6, 6,  # 二氧化硫 - 80 ppm
    6, 6, 6,  # 二氧化硫 - 100 ppm
    6, 6, 6, 6  # 二氧化硫 - 150 ppm
])

# 2.2 定义浓度数组（ppm单位）
concentration = np.array([
    0, 0, 0, 0, 0,  # 水（对照组）
    20, 20, 20,  # 20 ppm浓度组
    30, 30, 30, 30,  # 30 ppm浓度组
    50, 50, 50,  # 50 ppm浓度组
    80, 80, 80,  # 80 ppm浓度组
    100, 100, 100,  # 100 ppm浓度组
    150, 150, 150, 150  # 150 ppm浓度组
])

# 2.3 定义颜色数据矩阵（红、绿、蓝）
color_data = np.array([
    [153, 148, 157],  # 水（对照组）样本1
    [153, 147, 157],  # 水（对照组）样本2
    [153, 146, 158],  # 水（对照组）样本3
    [153, 146, 158],  # 水（对照组）样本4
    [154, 145, 157],  # 水（对照组）样本5
    [144, 115, 170],  # 20 ppm样本1
    [144, 115, 169],  # 20 ppm样本2
    [145, 115, 172],  # 20 ppm样本3
    [145, 114, 174],  # 30 ppm样本1
    [145, 114, 176],  # 30 ppm样本2
    [145, 114, 175],  # 30 ppm样本3
    [146, 114, 175],  # 30 ppm样本4
    [142, 99, 175],   # 50 ppm样本1
    [141, 99, 174],   # 50 ppm样本2
    [142, 99, 176],   # 50 ppm样本3
    [141, 96, 181],   # 80 ppm样本1
    [141, 96, 182],   # 80 ppm样本2
    [140, 96, 182],   # 80 ppm样本3
    [139, 96, 175],   # 100 ppm样本1
    [139, 96, 174],   # 100 ppm样本2
    [139, 96, 176],   # 100 ppm样本3
    [139, 86, 178],   # 150 ppm样本1
    [139, 87, 177],   # 150 ppm样本2
    [138, 86, 177],   # 150 ppm样本3
    [139, 86, 178]    # 150 ppm样本4
])  # 对应的颜色数据：红、绿、蓝

# ========== 第三步：数据预处理和结构化 ==========
# 3.1 创建DataFrame数据结构
data = pd.DataFrame(color_data, columns=['R', 'G', 'B'])  # 创建颜色数据框
data['Concentration'] = concentration  # 添加浓度列

# ========== 第四步：误差分析函数定义 ==========
def error_analysis(data, poly_degree=2):
    """
    对RGB三个颜色通道进行多项式拟合并计算误差指标
    
    参数:
    data: 包含RGB和浓度数据的DataFrame
    poly_degree: 多项式拟合的阶数
    
    返回:
    errors: 包含各通道误差指标的字典
    """
    errors = {}  # 初始化误差结果字典
    
    # 4.1 对每个颜色通道进行分析（并列关系）
    for color in ['R', 'G', 'B']:
        # 4.2 提取当前颜色通道数据
        x = data['Concentration'].values  # 浓度作为自变量
        y = data[color].values  # 当前颜色通道值作为因变量
        
        # 4.3 多项式拟合
        coeffs = np.polyfit(x, y, poly_degree)  # 计算多项式系数
        poly_func = np.poly1d(coeffs)  # 创建多项式函数
        y_pred = poly_func(x)  # 计算预测值
        
        # 4.4 计算误差指标（并列关系）
        mse = mean_squared_error(y, y_pred)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差
        mae = mean_absolute_error(y, y_pred)  # 平均绝对误差
        r2 = r2_score(y, y_pred)  # 决定系数
        
        # 4.5 存储当前通道的误差指标
        errors[color] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    return errors

# ========== 第五步：可视化函数定义 ==========
def plot_rgb_vs_concentration(data):
    """
    绘制RGB三个颜色通道与浓度的关系图
    
    参数:
    data: 包含RGB和浓度数据的DataFrame
    """
    # 5.1 设置图表布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图
    colors = ['R', 'G', 'B']  # 颜色通道列表
    color_names = ['红色', '绿色', '蓝色']  # 中文颜色名称
    plot_colors = ['red', 'green', 'blue']  # 绘图颜色
    
    # 5.2 为每个颜色通道绘制散点图（并列关系）
    for i, (color, color_name, plot_color) in enumerate(zip(colors, color_names, plot_colors)):
        # 5.3 绘制散点图
        axes[i].scatter(data['Concentration'], data[color], 
                       color=plot_color, alpha=0.7, s=50)  # 绘制数据点
        
        # 5.4 添加趋势线
        x = data['Concentration'].values  # 浓度数据
        y = data[color].values  # 当前颜色通道数据
        
        # 5.5 多项式拟合和趋势线绘制
        coeffs = np.polyfit(x, y, 2)  # 二次多项式拟合
        poly_func = np.poly1d(coeffs)  # 创建多项式函数
        x_smooth = np.linspace(x.min(), x.max(), 100)  # 生成平滑曲线的x值
        y_smooth = poly_func(x_smooth)  # 计算对应的y值
        axes[i].plot(x_smooth, y_smooth, color='black', linestyle='--', linewidth=2)  # 绘制趋势线
        
        # 5.6 设置图表属性
        axes[i].set_title(f'浓度与{color_name}通道值的关系', fontsize=14)  # 设置标题
        axes[i].set_xlabel('浓度 (ppm)', fontsize=12)  # 设置x轴标签
        axes[i].set_ylabel(f'{color_name}通道值', fontsize=12)  # 设置y轴标签
        axes[i].grid(True, alpha=0.3)  # 显示网格
        axes[i].set_xlim(-5, 155)  # 设置x轴范围
    
    # 5.7 调整布局并显示
    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示图表

# ========== 第六步：执行可视化分析 ==========
plot_rgb_vs_concentration(data)

# ========== 第七步：执行误差分析 ==========
# 7.1 进行误差分析（可调整多项式阶数）
errors = error_analysis(data, poly_degree=3)  # 使用三次多项式进行拟合

# 7.2 输出误差分析结果
for color, metrics in errors.items():
    print(f"\n{color} 通道的误差分析结果:")  # 输出通道名称
    print(f"均方误差 (MSE): {metrics['MSE']:.4f}")  # 输出均方误差
    print(f"均方根误差 (RMSE): {metrics['RMSE']:.4f}")  # 输出均方根误差
    print(f"平均绝对误差 (MAE): {metrics['MAE']:.4f}")  # 输出平均绝对误差
    print(f"决定系数 (R²): {metrics['R2']:.4f}")  # 输出决定系数

# ========== 第八步：总结分析 ==========
print("\n=== 分析总结 ===")
print("1. 通过多项式拟合分析了RGB三个颜色通道与二氧化硫浓度的关系")
print("2. 计算了各通道的误差指标，包括MSE、RMSE、MAE和R²")
print("3. 可视化展示了各颜色通道随浓度变化的趋势")
print("4. R²值越接近1表示拟合效果越好，MSE、RMSE、MAE值越小表示误差越小")


