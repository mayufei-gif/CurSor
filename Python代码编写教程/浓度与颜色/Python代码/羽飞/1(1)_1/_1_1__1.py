import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

# 设置字体为 SimHei 支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 手动输入数据：物质类别，浓度，蓝、绿、红、色调、饱和度，样本数量
substance = np.array([
    6, 6, 6, 6, 6,  # 二氧化硫 - 水
    6, 6, 6,  # 二氧化硫 - 20 ppm
    6, 6, 6, 6,  # 二氧化硫 - 30 ppm
    6, 6, 6,  # 二氧化硫 - 50 ppm
    6, 6, 6,  # 二氧化硫 - 80 ppm
    6, 6, 6,  # 二氧化硫 - 100 ppm
    6, 6, 6, 6  # 二氧化硫 - 150 ppm
])
concentration = np.array([
    0, 0, 0, 0, 0,  # 水
    20, 20, 20,  # 20 ppm
    30, 30, 30, 30,  # 30 ppm
    50, 50, 50,  # 50 ppm
    80, 80, 80,  # 80 ppm
    100, 100, 100,  # 100 ppm
    150, 150, 150, 150  # 150 ppm
])
sample_count = np.array([5,3,4,3,3,4])  # 每个物质类别的样本数量
color_data = np.array([
    [153, 148, 157, 138, 14],  # 水
    [153, 147, 157, 138, 16],
    [153, 146, 158, 137, 20],
    [153, 146, 158, 137, 20],
    [154, 145, 157, 141, 19],
    [144, 115, 170, 135, 82],  # 20 ppm
    [144, 115, 169, 136, 81],
    [145, 115, 172, 135, 83],
    [145, 114, 174, 135, 87],  # 30 ppm
    [145, 114, 176, 135, 89],
    [145, 114, 175, 135, 89],
    [146, 114, 175, 135, 88],
    [142, 99, 175, 137, 110],  # 50 ppm
    [141, 99, 174, 137, 109],
    [142, 99, 176, 136, 110],
    [141, 96, 181, 135, 119],  # 80 ppm
    [141, 96, 182, 135, 119],
    [140, 96, 182, 135, 120],
    [139, 96, 175, 136, 115],  # 100 ppm
    [139, 96, 174, 136, 114],
    [139, 96, 176, 136, 116],
    [139, 86, 178, 136, 131],  # 150 ppm
    [139, 87, 177, 137, 129],
    [138, 86, 177, 137, 130],
    [139, 86, 178, 137, 131]
])  # 对应的颜色数据：红、绿、蓝、饱和度、色调

# 定义每个颜色分量的权重，可以根据需要进行调整
weights_rgb = np.array([0.3, 0.3, 0.3])  # RGB每个通道的权重
weights_hsv = np.array([0.05, 0.05])  # 色调和饱和度的权重

# 计算颜色数据的加权和，得到一个一维数组表示颜色特征
color_combined = np.sum(color_data[:, 0:3] * weights_rgb, axis=1)

# 对颜色数据进行标准化处理
scaler = StandardScaler()
color_combined = scaler.fit_transform(color_combined.reshape(-1, 1))

# 获取所有物质的唯一类别
unique_substances = np.unique(substance)

# 初始化结构体，用于存储每个物质类别的回归结果
results = []

# 对每个物质进行建模，使用多项式回归
degree = 3 # 设置多项式的阶数，这里使用三次多项式

for i, sub in enumerate(unique_substances):
    sub_idx = substance == sub  # 获取当前物质类别的索引
    X = color_combined[sub_idx].reshape(-1, 1)  # 提取该类别的颜色数据并调整形状
    y = concentration[sub_idx]  # 提取该类别的浓度值

    # 使用多项式回归进行拟合
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # 提取回归系数和截距
    coefficients = model.coef_
    intercept = model.intercept_

    # 输出回归方程
    equation_terms = [f'{coeff:.3f}*x^{i}' for i, coeff in enumerate(coefficients)]
    equation = ' + '.join(equation_terms) + f' + {intercept:.3f}'
    print(f"物质 {sub} 的回归方程: y = {equation}")

    # 生成预测值
    X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # 生成更密集的点以显示曲线
    y_pred = model.predict(poly.transform(X_fit))

    # 计算MSE和R²
    mse = np.mean((model.predict(X_poly) - y) ** 2)
    r2 = model.score(X_poly, y)

    # 计算样本分布情况（标准差）
    sample_distribution = np.std(y)

    # 层次分析法计算综合评分（假设权重）
    weights = np.array([0.64, -0.27, 0.18])  # MSE为负权重，因为越小越好，样本数量的权重为0.1
    scores = np.array([r2, mse, sample_count[i]])
    composite_score = np.sum(weights * scores)  # 计算综合评分

    # 保存结果到结构体
    results.append({
        'wuzhi': sub,
        'MSE': mse,
        'R2': r2,
        'yangbenfengbu': sample_distribution,
        'zhonghepinfeng': composite_score,
        'samples': sample_count[i]
    })

    # 显示结果
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'样本分布（标准差）: {sample_distribution:.4f}')
    print(f'样本数量: {sample_count[i]}')
    print(f'综合评分: {composite_score:.4f}')

    # 可视化实际值与预测值
    plt.figure()
    plt.scatter(X, y, color='blue', label='实际值')
    plt.plot(X_fit, y_pred, color='red', label='拟合曲线')
    plt.title(f'物质 {sub}: 实际值与预测值')
    plt.xlabel('颜色特征')
    plt.ylabel('浓度')
    plt.grid(True)
    plt.legend()
    plt.show()
