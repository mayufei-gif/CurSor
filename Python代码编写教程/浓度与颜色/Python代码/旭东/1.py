import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from matplotlib import rcParams

# 设置字体为 SimHei 支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 手动输入数据：物质类别，浓度，蓝、绿、红、色调、饱和度，样本数量
substance = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 组胺
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # 溴酸钾
                      3, 3, 3, 3, 3, 3, 3,            # 工业碱
                      4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                      4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                      4, 4, 4, 4, 4, 4,4,4,4,4,4,4,4,4,4,4,  # 硫酸铝钾
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])  # 物质类别，每个类别包含多个样本
concentration = np.array([0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5,  # 组胺
                          0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5,  # 溴酸钾
                          7.34, 8.14, 8.74, 9.19, 10.18, 11.8, 0,      # 工业碱
                          0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # 硫酸铝钾
                          1, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                          2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5,         # 硫酸铝钾
                          0, 500, 1000, 1500, 2000, 0, 5, 500, 1500, 2000,  # 奶中尿素
                          0, 500, 1000, 1500, 2000])  # 对应的浓度值
sample_count = np.array([10, 10, 7, 36, 15])  # 每个物质类别的样本数量
color_data = np.array([
    [68, 110, 121, 23, 111],  # 组胺
    [37, 66, 110, 12, 169],
    [46, 87, 117, 16, 155],
    [62, 99, 120, 19, 122],
    [66, 102, 118, 20, 112],
    [65, 110, 120, 24, 115],
    [35, 64, 109, 11, 172],
    [46, 87, 118, 16, 153],
    [60, 99, 120, 19, 126],
    [64, 101, 118, 20, 115],
    [129, 141, 145, 22, 27],  # 溴酸钾
    [7, 133, 145, 27, 241],
    [60, 133, 141, 27, 145],
    [69, 136, 145, 26, 133],
    [85, 139, 145, 26, 106],
    [128, 141, 144, 23, 28],
    [7, 133, 145, 27, 242],
    [57, 133, 141, 27, 151],
    [70, 137, 146, 26, 132],
    [87, 138, 146, 26, 102],
    [153, 140, 132, 108, 35],  # 工业碱
    [151, 142, 133, 104, 29],
    [158, 126, 127, 120, 52],
    [161, 85, 118, 132, 120],
    [127, 21, 119, 147, 211],
    [94, 6, 91, 148, 237],
    [152, 142, 132, 105, 32],
    [116, 126, 104, 76, 44],  # 硫酸铝钾
    [114, 126, 104, 74, 45],
    [118, 125, 105, 78, 40],
    [113, 124, 103, 73, 42],
    [114, 124, 104, 75, 39],
    [113, 126, 104, 72, 45],
    [148, 112, 47, 100, 174],
    [150, 111, 44, 100, 178],
    [138, 118, 71, 98, 123],
    [136, 118, 70, 98, 122],
    [136, 117, 64, 98, 134],
    [136, 118, 64, 97, 135],
    [149, 116, 48, 99, 172],
    [150, 115, 49, 100, 171],
    [147, 119, 55, 99, 159],
    [149, 119, 64, 100, 145],
    [140, 113, 54, 99, 156],
    [137, 111, 51, 99, 160],
    [153, 113, 44, 101, 180],
    [153, 113, 42, 100, 184],
    [153, 115, 50, 101, 171],
    [153, 115, 47, 100, 176],
    [152, 116, 52, 100, 167],
    [153, 116, 49, 100, 171],
    [156, 106, 34, 102, 199],
    [162, 107, 37, 103, 196],
    [161, 110, 40, 102, 190],
    [163, 111, 38, 102, 194],
    [159, 104, 35, 103, 198],
    [158, 105, 35, 103, 198],
    [155, 107, 34, 101, 198],
    [156, 108, 34, 101, 198],
    [152, 116, 48, 100, 174],
    [151, 115, 51, 100, 168],
    [154, 105, 33, 102, 199],
    [156, 105, 35, 102, 197],
    [118, 136, 139, 25, 37],  # 奶中尿素
    [117, 137, 139, 27, 41],
    [108, 136, 138, 28, 54],
    [110, 136, 139, 26, 52],
    [108, 140, 142, 28, 60],
    [120, 136, 138, 26, 33],
    [119, 140, 142, 26, 40],
    [111, 139, 142, 27, 55],
    [107, 136, 139, 26, 58],
    [105, 136, 137, 28, 58],
    [125, 135, 140, 20, 27],
    [114, 134, 138, 25, 44],
    [112, 132, 134, 27, 42],
    [105, 134, 138, 26, 60],
    [107, 135, 138, 26, 57]
])  # 对应的颜色数据：蓝、绿、红、色调、饱和度

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

    # 将 MSE, R2, 样本数量归一化
    scaler_minmax = MinMaxScaler()
    normalized_values = scaler_minmax.fit_transform(np.array([r2, mse, sample_count[i]]).reshape(-1, 1)).flatten()

    # 层次分析法计算综合评分（假设权重）
    weights = np.array([0.64, -0.27, 0.18])  # MSE为负权重，因为越小越好，样本数量的权重为0.1
    composite_score = np.sum(weights * normalized_values)  # 计算综合评分

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

    # 可视化实际值与预测值，交换坐标轴
    plt.figure()
    plt.scatter(y, X, color='blue', label='实际值')
    plt.plot(y_pred, X_fit, color='red', label='拟合曲线')
    plt.title(f'物质 {sub}: 实际值与预测值')
    plt.xlabel('浓度')
    plt.ylabel('颜色特征')
    plt.grid(True)
    plt.legend()
    plt.show()