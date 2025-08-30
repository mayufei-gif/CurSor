# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import numpy as np  # 数值计算和数组操作
import matplotlib.pyplot as plt  # 数据可视化和图表绘制
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 多项式特征转换
from sklearn.preprocessing import StandardScaler  # 数据标准化
from matplotlib import rcParams  # matplotlib配置参数

# ========== 第二步：设置中文字体显示 ==========
# 以下两行为并列关系，共同解决中文显示问题
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文显示
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ========== 第三步：定义二氧化硫实验数据 ==========
# 以下数据定义为并列关系，构成完整的实验数据集

# 3.1 定义物质类别数组（二氧化硫检测实验）
substance = np.array([
    6, 6, 6, 6, 6,  # 二氧化硫 - 水（对照组）
    6, 6, 6,  # 二氧化硫 - 20 ppm
    6, 6, 6, 6,  # 二氧化硫 - 30 ppm
    6, 6, 6,  # 二氧化硫 - 50 ppm
    6, 6, 6,  # 二氧化硫 - 80 ppm
    6, 6, 6,  # 二氧化硫 - 100 ppm
    6, 6, 6, 6  # 二氧化硫 - 150 ppm
])

# 3.2 定义浓度数组（ppm单位）
concentration = np.array([
    0, 0, 0, 0, 0,  # 水（对照组）
    20, 20, 20,  # 20 ppm浓度组
    30, 30, 30, 30,  # 30 ppm浓度组
    50, 50, 50,  # 50 ppm浓度组
    80, 80, 80,  # 80 ppm浓度组
    100, 100, 100,  # 100 ppm浓度组
    150, 150, 150, 150  # 150 ppm浓度组
])

# 3.3 定义每个浓度组的样本数量
sample_count = np.array([5,3,4,3,3,4])  # 每个浓度组的样本数量

# 3.4 定义颜色数据矩阵（红、绿、蓝、饱和度、色调）
color_data = np.array([
    [153, 148, 157, 138, 14],  # 水（对照组）样本1
    [153, 147, 157, 138, 16],  # 水（对照组）样本2
    [153, 146, 158, 137, 20],  # 水（对照组）样本3
    [153, 146, 158, 137, 20],  # 水（对照组）样本4
    [154, 145, 157, 141, 19],  # 水（对照组）样本5
    [144, 115, 170, 135, 82],  # 20 ppm样本1
    [144, 115, 169, 136, 81],  # 20 ppm样本2
    [145, 115, 172, 135, 83],  # 20 ppm样本3
    [145, 114, 174, 135, 87],  # 30 ppm样本1
    [145, 114, 176, 135, 89],  # 30 ppm样本2
    [145, 114, 175, 135, 89],  # 30 ppm样本3
    [146, 114, 175, 135, 88],  # 30 ppm样本4
    [142, 99, 175, 137, 110],  # 50 ppm样本1
    [141, 99, 174, 137, 109],  # 50 ppm样本2
    [142, 99, 176, 136, 110],  # 50 ppm样本3
    [141, 96, 181, 135, 119],  # 80 ppm样本1
    [141, 96, 182, 135, 119],  # 80 ppm样本2
    [140, 96, 182, 135, 120],  # 80 ppm样本3
    [139, 96, 175, 136, 115],  # 100 ppm样本1
    [139, 96, 174, 136, 114],  # 100 ppm样本2
    [139, 96, 176, 136, 116],  # 100 ppm样本3
    [139, 86, 178, 136, 131],  # 150 ppm样本1
    [139, 87, 177, 137, 129],  # 150 ppm样本2
    [138, 86, 177, 137, 130],  # 150 ppm样本3
    [139, 86, 178, 137, 131]   # 150 ppm样本4
])  # 对应的颜色数据：红、绿、蓝、饱和度、色调

# ========== 第四步：设置权重参数 ==========
# 以下权重设置为并列关系，用于不同特征的加权计算
weights_rgb = np.array([0.3, 0.3, 0.3])  # RGB每个通道的权重
weights_hsv = np.array([0.05, 0.05])  # 色调和饱和度的权重

# ========== 第五步：颜色特征组合和标准化 ==========
# 5.1 计算RGB加权组合特征
color_combined = np.sum(color_data[:, 0:3] * weights_rgb, axis=1)

# 5.2 对组合特征进行标准化处理
scaler = StandardScaler()
color_combined = scaler.fit_transform(color_combined.reshape(-1, 1))

# ========== 第六步：初始化分析参数 ==========
# 以下初始化为并列关系，为后续分析做准备
unique_substances = np.unique(substance)  # 获取唯一物质类别

# 初始化结果存储列表
results = []

# 设置多项式回归的阶数
degree = 3  # 设置多项式的阶数，这里使用三次多项式

# ========== 第七步：对二氧化硫进行回归分析 ==========
for i, sub in enumerate(unique_substances):
    # 7.1 数据提取和准备
    sub_idx = substance == sub  # 获取当前物质类别的索引
    X = color_combined[sub_idx].reshape(-1, 1)  # 提取该类别的颜色数据并调整形状
    y = concentration[sub_idx]  # 提取该类别的浓度值

    # 7.2 多项式特征转换和模型训练
    poly = PolynomialFeatures(degree)  # 创建多项式特征转换器
    X_poly = poly.fit_transform(X)  # 转换为多项式特征
    model = LinearRegression()  # 创建线性回归模型
    model.fit(X_poly, y)  # 训练模型

    # 7.3 提取回归方程参数
    coefficients = model.coef_  # 获取回归系数
    intercept = model.intercept_  # 获取截距

    # 7.4 构建和输出回归方程
    equation_terms = [f'{coeff:.3f}*x^{i}' for i, coeff in enumerate(coefficients)]
    equation = ' + '.join(equation_terms) + f' + {intercept:.3f}'
    print(f"物质 {sub} 的回归方程: y = {equation}")

    # 7.5 生成预测数据用于绘图
    X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # 生成更密集的点以显示曲线
    y_pred = model.predict(poly.transform(X_fit))  # 预测对应的浓度值

    # 7.6 计算模型评估指标
    mse = np.mean((model.predict(X_poly) - y) ** 2)  # 计算均方误差
    r2 = model.score(X_poly, y)  # 计算决定系数

    # 7.7 计算样本分布特征
    sample_distribution = np.std(y)  # 计算浓度值的标准差

    # 7.8 计算综合评分
    weights = np.array([0.64, -0.27, 0.18])  # MSE为负权重，因为越小越好，样本数量的权重为0.18
    scores = np.array([r2, mse, sample_count[i]])  # 评估指标数组
    composite_score = np.sum(weights * scores)  # 计算综合评分

    # 7.9 存储分析结果
    results.append({
        'wuzhi': sub,
        'MSE': mse,
        'R2': r2,
        'yangbenfengbu': sample_distribution,
        'zhonghepinfeng': composite_score,
        'samples': sample_count[i]
    })

    # 7.10 输出分析结果
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'样本分布（标准差）: {sample_distribution:.4f}')
    print(f'样本数量: {sample_count[i]}')
    print(f'综合评分: {composite_score:.4f}')

    # 7.11 绘制散点图和拟合曲线
    plt.figure()
    plt.scatter(X, y, color='blue', label='实际值')  # 绘制实际数据点
    plt.plot(X_fit, y_pred, color='red', label='拟合曲线')  # 绘制拟合曲线
    plt.title(f'物质 {sub}: 实际值与预测值')  # 设置图表标题
    plt.xlabel('颜色特征')  # 设置x轴标签
    plt.ylabel('浓度')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图表
