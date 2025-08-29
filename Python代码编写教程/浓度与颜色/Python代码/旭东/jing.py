import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei (黑体) 以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题
# 定义 Michaelis-Menten 模型的方程
def michaelis_menten(x, Vmax, Km):
    return (Vmax * x) / (Km + x)

# 组胺数据
concentration = [0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5]
blue = [68, 37, 46, 62, 66, 65, 35, 46, 60, 64]
green = [110, 66, 87, 99, 102, 110, 64, 87, 99, 101]
red = [121, 110, 117, 120, 118, 120, 109, 118, 120, 118]

# 绘制二维散点图
plt.figure()
plt.scatter(concentration, blue, s=100, c='b', marker='o', label='蓝色 (圆圈)')
plt.scatter(concentration, green, s=100, c='g', marker='s', label='绿色 (方块)')
plt.scatter(concentration, red, s=100, c='r', marker='^', label='红色 (三角形)')
plt.xlabel('组胺浓度')
plt.ylabel('颜色读数')
plt.title('组胺浓度与颜色读数之间的关系')
plt.legend()
plt.show()

# RGB转灰度值
gray_values = 0.2989 * np.array(red) + 0.5870 * np.array(green) + 0.1140 * np.array(blue)

# 绘制灰度值与组胺浓度的散点图
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')
plt.xlabel('组胺浓度')
plt.ylabel('灰度值')
plt.title('组胺浓度与灰度值之间的关系')
plt.legend()
plt.show()

# 溴酸钾数据
concentration = [0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5]
blue = [129, 7, 60, 69, 85, 128, 7, 57, 70, 87]
green = [141, 133, 133, 136, 139, 141, 133, 133, 137, 138]
red = [145, 145, 141, 145, 145, 144, 145, 141, 146, 146]

# 绘制二维散点图
plt.figure()
plt.scatter(concentration, blue, s=100, c='b', marker='o', label='蓝色 (圆圈)')
plt.scatter(concentration, green, s=100, c='g', marker='s', label='绿色 (方块)')
plt.scatter(concentration, red, s=100, c='r', marker='^', label='红色 (三角形)')
plt.xlabel('溴酸钾浓度')
plt.ylabel('颜色读数')
plt.title('溴酸钾浓度与颜色读数之间的关系')
plt.legend()
plt.show()

# RGB转灰度值
gray_values = 0.2989 * np.array(red) + 0.5870 * np.array(green) + 0.1140 * np.array(blue)

# 绘制灰度值与溴酸钾浓度的散点图
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')
plt.xlabel('溴酸钾浓度')
plt.ylabel('灰度值')
plt.title('溴酸钾浓度与灰度值之间的关系')
plt.legend()
plt.show()

# 进行线性回归
p = np.polyfit(concentration, gray_values, 1)
yfit = np.polyval(p, concentration)

# 绘制回归线
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')
plt.plot(concentration, yfit, 'r-', label='线性回归')
plt.xlabel('溴酸钾浓度')
plt.ylabel('灰度值')
plt.title('溴酸钾浓度与灰度值的线性回归')
plt.legend()
plt.show()

# 二次多项式拟合
p2 = np.polyfit(concentration, gray_values, 2)
yfit2 = np.polyval(p2, concentration)

# 绘制二次函数拟合曲线
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')
plt.plot(concentration, yfit2, 'b-', label='二次拟合曲线')
plt.xlabel('溴酸钾浓度')
plt.ylabel('灰度值')
plt.title('溴酸钾浓度与灰度值的二次函数拟合')
plt.legend()
plt.show()

# 计算R²值
y_mean = np.mean(gray_values)
SS_tot = np.sum((gray_values - y_mean) ** 2)
SS_res = np.sum((gray_values - yfit2) ** 2)
R2 = 1 - (SS_res / SS_tot)
print(f'R²值: {R2}')

# 工业碱数据
concentration = [7.34, 8.14, 8.74, 9.19, 10.18, 11.8, 0]
blue = [153, 151, 158, 161, 127, 94, 152]
green = [140, 142, 126, 85, 21, 6, 142]
red = [132, 133, 127, 118, 119, 91, 132]

# 绘制二维散点图
plt.figure()
plt.scatter(concentration, blue, s=100, c='b', marker='o', label='蓝色 (圆圈)')
plt.scatter(concentration, green, s=100, c='g', marker='s', label='绿色 (方块)')
plt.scatter(concentration, red, s=100, c='r', marker='^', label='红色 (三角形)')
plt.xlabel('工业碱浓度')
plt.ylabel('颜色读数')
plt.title('工业碱浓度与颜色读数之间的关系')
plt.legend()
plt.show()

# RGB转灰度值
gray_values = 0.2989 * np.array(red) + 0.5870 * np.array(green) + 0.1140 * np.array(blue)

# 绘制灰度值与工业碱浓度的散点图
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')
plt.xlabel('工业碱浓度')
plt.ylabel('灰度值')
plt.title('工业碱浓度与灰度值之间的关系')
plt.legend()
plt.show()

# 硫酸铝钾数据
concentration = [0, 0.5, 1, 1.5, 2, 5]
concentration_repeated = np.repeat(concentration, 6)

blue = [116, 114, 118, 113, 114, 113, 148, 150, 138, 136, 136, 136, 138, 149, 150, 147, 149, 140, 137, 153, 153, 153, 153, 152, 153, 156, 162, 161, 163, 159, 155, 156, 152, 151, 154, 156]
green = [126, 126, 125, 124, 124, 126, 112, 111, 118, 117, 118, 118, 111, 116, 115, 119, 119, 113, 111, 113, 113, 115, 115, 116, 116, 106, 107, 110, 111, 104, 107, 108, 116, 115, 105, 105]
red = [104, 104, 105, 103, 104, 104, 47, 44, 71, 70, 64, 64, 50, 48, 49, 55, 64, 54, 51, 44, 42, 50, 47, 52, 49, 34, 37, 40, 38, 35, 35, 34, 48, 51, 33, 35]

# 计算灰度值
gray = 0.299 * np.array(red) + 0.587 * np.array(green) + 0.114 * np.array(blue)

# 绘制颜色读数的散点图
plt.figure()
plt.scatter(concentration_repeated, blue, c='b', marker='o', label='蓝色')
plt.scatter(concentration_repeated, green, c='g', marker='s', label='绿色')
plt.scatter(concentration_repeated, red, c='r', marker='^', label='红色')
plt.xlabel('浓度 (mol/L)')
plt.ylabel('颜色读数')
plt.title('颜色读数与硫酸铝钾浓度的关系')
plt.legend()
plt.show()

# 绘制灰度与浓度的散点图
plt.figure()
plt.scatter(concentration_repeated, gray, c='k', marker='d', label='灰度值')

# 添加灰度与浓度的回归模型
p = np.polyfit(concentration_repeated, gray, 1)
yfit = np.polyval(p, concentration_repeated)
plt.plot(concentration_repeated, yfit, 'r-', label='线性拟合')
plt.xlabel('浓度 (mol/L)')
plt.ylabel('灰度值')
plt.title('灰度值与硫酸铝钾浓度的关系')
plt.legend()
plt.show()

# 色调和饱和度数据
hue = [76, 74, 78, 73, 75, 72, 100, 100, 98, 98, 98, 97, 99, 99, 100, 99, 100, 99, 99, 101, 101, 101, 100, 100, 100, 102, 103, 102, 102, 102, 103, 103, 100, 100, 102, 102]
saturation = [44, 45, 40, 42, 39, 45, 174, 178, 123, 122, 134, 135, 161, 172, 171, 159, 145, 156, 160, 180, 184, 171, 176, 167, 171, 199, 196, 190, 194, 198, 198, 198, 174, 168, 199, 197]

# 二次多项式拟合 - 色调
p_hue = np.polyfit(concentration_repeated, hue, 2)
yfit_hue = np.polyval(p_hue, concentration_repeated)

# 二次多项式拟合 - 饱和度
p_saturation = np.polyfit(concentration_repeated, saturation, 2)
yfit_saturation = np.polyval(p_saturation, concentration_repeated)

# 显示拟合结果 - 色调
plt.figure()
plt.scatter(concentration_repeated, hue, c='b', marker='o', label='色调')
plt.plot(concentration_repeated, yfit_hue, 'r-', label='二次拟合曲线')
plt.xlabel('浓度 (mol/L)')
plt.ylabel('色调')
plt.title('色调 - 二次多项式拟合')
plt.legend()
plt.show()

# 显示拟合结果 - 饱和度
plt.figure()
plt.scatter(concentration_repeated, saturation, c='r', marker='s', label='饱和度')
plt.plot(concentration_repeated, yfit_saturation, 'b-', label='二次拟合曲线')
plt.xlabel('浓度 (mol/L)')
plt.ylabel('饱和度')
plt.title('饱和度 - 二次多项式拟合')
plt.legend()
plt.show()

# Michaelis-Menten拟合
initial_guess = [max(hue), 1]
hue_fit_params, _ = curve_fit(michaelis_menten, concentration_repeated, hue, p0=initial_guess)
fitted_hue = michaelis_menten(concentration_repeated, *hue_fit_params)

# 绘制色调拟合结果
plt.figure()
plt.scatter(concentration_repeated, hue, c='b', marker='o', label='原始数据')
plt.plot(concentration_repeated, fitted_hue, 'r-', label='Michaelis-Menten拟合')
plt.xlabel('浓度 (mol/L)')
plt.ylabel('色调')
plt.title('色调与硫酸铝钾浓度的Michaelis-Menten拟合')
plt.legend()
plt.show()

# 对饱和度进行相同处理
initial_guess = [max(saturation), 1]
saturation_fit_params, _ = curve_fit(michaelis_menten, concentration_repeated, saturation, p0=initial_guess)
fitted_saturation = michaelis_menten(concentration_repeated, *saturation_fit_params)

# 绘制饱和度拟合结果
plt.figure()
plt.scatter(concentration_repeated, saturation, c='r', marker='s', label='原始数据')
plt.plot(concentration_repeated, fitted_saturation, 'b-', label='Michaelis-Menten拟合')
plt.xlabel('浓度 (mol/L)')
plt.ylabel('饱和度')
plt.title('饱和度与硫酸铝钾浓度的Michaelis-Menten拟合')
plt.legend()
plt.show()

# 比较拟合效果
rss_hue = np.sum((hue - fitted_hue) ** 2)
rss_saturation = np.sum((saturation - fitted_saturation) ** 2)

# 选择拟合效果较好的模型并绘图
plt.figure()
if rss_hue < rss_saturation:
    better_fit = '色调'
    better_fit_params = hue_fit_params
    plt.scatter(concentration_repeated, hue, c='b', marker='o', label='原始数据')
    plt.plot(concentration_repeated, fitted_hue, 'r-', label='Michaelis-Menten拟合')
else:
    better_fit = '饱和度'
    better_fit_params = saturation_fit_params
    plt.scatter(concentration_repeated, saturation, c='r', marker='s', label='原始数据')
    plt.plot(concentration_repeated, fitted_saturation, 'b-', label='Michaelis-Menten拟合')

plt.xlabel('浓度 (mol/L)')
plt.ylabel(better_fit)
plt.title(f'{better_fit}与硫酸铝钾浓度的Michaelis-Menten拟合')
plt.legend()
plt.show()

# 输出拟合效果较好的模型方程
print(f'{better_fit}的拟合效果较好：')
print(f'拟合方程: y = ({better_fit_params[0]} * [S]) / ({better_fit_params[1]} + [S])')
