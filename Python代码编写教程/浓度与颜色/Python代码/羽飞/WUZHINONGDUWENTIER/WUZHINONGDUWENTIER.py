import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 手动输入数据：物质类别，浓度，红、绿、蓝，样本数量
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

color_data = np.array([
    [153, 148, 157],  # 水
    [153, 147, 157],
    [153, 146, 158],
    [153, 146, 158],
    [154, 145, 157],
    [144, 115, 170],  # 20 ppm
    [144, 115, 169],
    [145, 115, 172],
    [145, 114, 174],  # 30 ppm
    [145, 114, 176],
    [145, 114, 175],
    [146, 114, 175],
    [142, 99, 175],  # 50 ppm
    [141, 99, 174],
    [142, 99, 176],
    [141, 96, 181],  # 80 ppm
    [141, 96, 182],
    [140, 96, 182],
    [139, 96, 175],  # 100 ppm
    [139, 96, 174],
    [139, 96, 176],
    [139, 86, 178],  # 150 ppm
    [139, 87, 177],
    [138, 86, 177],
    [139, 86, 178]
])  # 对应的颜色数据：红、绿、蓝

# 将颜色数据与浓度组合成一个DataFrame
data = pd.DataFrame(color_data, columns=['R', 'G', 'B'])
data['Concentration'] = concentration

# 定义误差分析函数
def error_analysis(data, poly_degree=2):
    errors = {}
    
    for color in ['R', 'G', 'B']:
        # 多项式拟合
        coeffs = np.polyfit(data['Concentration'], data[color], poly_degree)
        fitted_vals = np.polyval(coeffs, data['Concentration'])
        
        # 计算误差
        mse = mean_squared_error(data[color], fitted_vals)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(data[color], fitted_vals)
        r2 = r2_score(data[color], fitted_vals)
        
        # 存储误差结果
        errors[color] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    return errors

# 定义绘图和多项式拟合函数
def plot_rgb_vs_concentration(data):
    plt.figure(figsize=(12, 8))
    
    # 多项式阶数（可以在这里修改多项式阶数）
    poly_degree = 2  # 例如，使用2阶多项式进行拟合

    # 绘制红色值与浓度的关系
    plt.subplot(3, 1, 1)
    sns.scatterplot(x=data['Concentration'], y=data['R'], hue=data['Concentration'], palette='coolwarm', s=100)
    plt.title('浓度与红色(R)值的关系')
    plt.xlabel('浓度 (ppm)')
    plt.ylabel('红色(R)值')
    
    # 多项式拟合
    poly_coeffs_R = np.polyfit(data['Concentration'], data['R'], poly_degree)
    poly_vals_R = np.polyval(poly_coeffs_R, sorted(data['Concentration']))
    plt.plot(sorted(data['Concentration']), poly_vals_R, color='black', linestyle='--')  # 绘制拟合曲线

    # 绘制绿色值与浓度的关系
    plt.subplot(3, 1, 2)
    sns.scatterplot(x=data['Concentration'], y=data['G'], hue=data['Concentration'], palette='coolwarm', s=100)
    plt.title('浓度与绿色(G)值的关系')
    plt.xlabel('浓度 (ppm)')
    plt.ylabel('绿色(G)值')
    
    # 多项式拟合
    poly_coeffs_G = np.polyfit(data['Concentration'], data['G'], poly_degree)
    poly_vals_G = np.polyval(poly_coeffs_G, sorted(data['Concentration']))
    plt.plot(sorted(data['Concentration']), poly_vals_G, color='black', linestyle='--')  # 绘制拟合曲线

    # 绘制蓝色值与浓度的关系
    plt.subplot(3, 1, 3)
    sns.scatterplot(x=data['Concentration'], y=data['B'], hue=data['Concentration'], palette='coolwarm', s=100)
    plt.title('浓度与蓝色(B)值的关系')
    plt.xlabel('浓度 (ppm)')
    plt.ylabel('蓝色(B)值')
    
    # 多项式拟合
    poly_coeffs_B = np.polyfit(data['Concentration'], data['B'], poly_degree)
    poly_vals_B = np.polyval(poly_coeffs_B, sorted(data['Concentration']))
    plt.plot(sorted(data['Concentration']), poly_vals_B, color='black', linestyle='--')  # 绘制拟合曲线

    plt.tight_layout()
    plt.show()

# 为二氧化硫绘制RGB值与浓度的散点图并进行多项式拟合
plot_rgb_vs_concentration(data)

# 计算误差并打印结果
errors = error_analysis(data, poly_degree=3)  # 可以在这里修改多项式阶数

for color, metrics in errors.items():
    print(f"\n{color} 通道的误差分析结果:")
    print(f"均方误差 (MSE): {metrics['MSE']:.4f}")
    print(f"均方根误差 (RMSE): {metrics['RMSE']:.4f}")
    print(f"平均绝对误差 (MAE): {metrics['MAE']:.4f}")
    print(f"决定系数 (R²): {metrics['R2']:.4f}")


