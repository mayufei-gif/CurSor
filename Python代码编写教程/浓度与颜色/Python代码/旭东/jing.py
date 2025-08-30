# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import numpy as np  # 数值计算和数组操作
import matplotlib.pyplot as plt  # 数据可视化和图表绘制
from scipy.optimize import curve_fit  # 非线性曲线拟合
import matplotlib.pyplot as plt  # 数据可视化（重复导入，可优化）
from matplotlib import rcParams  # matplotlib配置参数

# ========== 第二步：设置中文字体显示 ==========
# 以下两行为并列关系，共同解决中文显示问题
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文显示
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ========== 第三步：定义Michaelis-Menten模型函数 ==========
def michaelis_menten(x, Vmax, Km):
    """Michaelis-Menten动力学模型：用于描述酶反应动力学或类似的饱和现象"""
    return (Vmax * x) / (Km + x)

# ========== 第四步：组胺数据分析 ==========
# 4.1 定义组胺实验数据
concentration = [0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5]  # 组胺浓度数据
blue = [68, 37, 46, 62, 66, 65, 35, 46, 60, 64]  # 蓝色通道读数
green = [110, 66, 87, 99, 102, 110, 64, 87, 99, 101]  # 绿色通道读数
red = [121, 110, 117, 120, 118, 120, 109, 118, 120, 118]  # 红色通道读数

# 4.2 绘制组胺RGB三通道散点图
plt.figure()
plt.scatter(concentration, blue, s=100, c='b', marker='o', label='蓝色 (圆圈)')  # 蓝色通道散点
plt.scatter(concentration, green, s=100, c='g', marker='s', label='绿色 (方块)')  # 绿色通道散点
plt.scatter(concentration, red, s=100, c='r', marker='^', label='红色 (三角形)')  # 红色通道散点
plt.xlabel('组胺浓度')  # 设置x轴标签
plt.ylabel('颜色读数')  # 设置y轴标签
plt.title('组胺浓度与颜色读数之间的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 4.3 计算组胺数据的灰度值
gray_values = 0.2989 * np.array(red) + 0.5870 * np.array(green) + 0.1140 * np.array(blue)  # RGB转灰度公式

# 4.4 绘制组胺灰度值散点图
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')  # 灰度值散点
plt.xlabel('组胺浓度')  # 设置x轴标签
plt.ylabel('灰度值')  # 设置y轴标签
plt.title('组胺浓度与灰度值之间的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# ========== 第五步：溴酸钾数据分析 ==========
# 5.1 定义溴酸钾实验数据
concentration = [0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5]  # 溴酸钾浓度数据
blue = [129, 7, 60, 69, 85, 128, 7, 57, 70, 87]  # 蓝色通道读数
green = [141, 133, 133, 136, 139, 141, 133, 133, 137, 138]  # 绿色通道读数
red = [145, 145, 141, 145, 145, 144, 145, 141, 146, 146]  # 红色通道读数

# 5.2 绘制溴酸钾RGB三通道散点图
plt.figure()
plt.scatter(concentration, blue, s=100, c='b', marker='o', label='蓝色 (圆圈)')  # 蓝色通道散点
plt.scatter(concentration, green, s=100, c='g', marker='s', label='绿色 (方块)')  # 绿色通道散点
plt.scatter(concentration, red, s=100, c='r', marker='^', label='红色 (三角形)')  # 红色通道散点
plt.xlabel('溴酸钾浓度')  # 设置x轴标签
plt.ylabel('颜色读数')  # 设置y轴标签
plt.title('溴酸钾浓度与颜色读数之间的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 5.3 计算溴酸钾数据的灰度值
gray_values = 0.2989 * np.array(red) + 0.5870 * np.array(green) + 0.1140 * np.array(blue)  # RGB转灰度公式

# 5.4 绘制溴酸钾灰度值散点图
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')  # 灰度值散点
plt.xlabel('溴酸钾浓度')  # 设置x轴标签
plt.ylabel('灰度值')  # 设置y轴标签
plt.title('溴酸钾浓度与灰度值之间的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 5.5 进行线性回归拟合
p = np.polyfit(concentration, gray_values, 1)  # 一次多项式拟合（线性回归）
yfit = np.polyval(p, concentration)  # 计算拟合值

# 5.6 绘制线性回归结果
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')  # 原始数据散点
plt.plot(concentration, yfit, 'r-', label='线性回归')  # 线性拟合曲线
plt.xlabel('溴酸钾浓度')  # 设置x轴标签
plt.ylabel('灰度值')  # 设置y轴标签
plt.title('溴酸钾浓度与灰度值的线性回归')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 5.7 进行二次多项式拟合
p2 = np.polyfit(concentration, gray_values, 2)  # 二次多项式拟合
yfit2 = np.polyval(p2, concentration)  # 计算二次拟合值

# 5.8 绘制二次拟合结果
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')  # 原始数据散点
plt.plot(concentration, yfit2, 'b-', label='二次拟合曲线')  # 二次拟合曲线
plt.xlabel('溴酸钾浓度')  # 设置x轴标签
plt.ylabel('灰度值')  # 设置y轴标签
plt.title('溴酸钾浓度与灰度值的二次函数拟合')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 5.9 计算二次拟合的R²值
y_mean = np.mean(gray_values)  # 计算灰度值均值
SS_tot = np.sum((gray_values - y_mean) ** 2)  # 总平方和
SS_res = np.sum((gray_values - yfit2) ** 2)  # 残差平方和
R2 = 1 - (SS_res / SS_tot)  # 计算决定系数R²
print(f'R²值: {R2}')  # 输出R²值

# ========== 第六步：工业碱数据分析 ==========
# 6.1 定义工业碱实验数据
concentration = [7.34, 8.14, 8.74, 9.19, 10.18, 11.8, 0]  # 工业碱浓度数据
blue = [153, 151, 158, 161, 127, 94, 152]  # 蓝色通道读数
green = [140, 142, 126, 85, 21, 6, 142]  # 绿色通道读数
red = [132, 133, 127, 118, 119, 91, 132]  # 红色通道读数

# 6.2 绘制工业碱RGB三通道散点图
plt.figure()
plt.scatter(concentration, blue, s=100, c='b', marker='o', label='蓝色 (圆圈)')  # 蓝色通道散点
plt.scatter(concentration, green, s=100, c='g', marker='s', label='绿色 (方块)')  # 绿色通道散点
plt.scatter(concentration, red, s=100, c='r', marker='^', label='红色 (三角形)')  # 红色通道散点
plt.xlabel('工业碱浓度')  # 设置x轴标签
plt.ylabel('颜色读数')  # 设置y轴标签
plt.title('工业碱浓度与颜色读数之间的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 6.3 计算工业碱数据的灰度值
gray_values = 0.2989 * np.array(red) + 0.5870 * np.array(green) + 0.1140 * np.array(blue)  # RGB转灰度公式

# 6.4 绘制工业碱灰度值散点图
plt.figure()
plt.scatter(concentration, gray_values, s=100, c='k', marker='d', label='灰度值')  # 灰度值散点
plt.xlabel('工业碱浓度')  # 设置x轴标签
plt.ylabel('灰度值')  # 设置y轴标签
plt.title('工业碱浓度与灰度值之间的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# ========== 第七步：硫酸铝钾数据分析 ==========
# 7.1 定义硫酸铝钾实验数据
concentration = [0, 0.5, 1, 1.5, 2, 5]  # 基础浓度数组
concentration_repeated = np.repeat(concentration, 6)  # 每个浓度重复6次，对应6个样本

# 7.2 定义RGB颜色数据（36个样本）
blue = [116, 114, 118, 113, 114, 113, 148, 150, 138, 136, 136, 136, 138, 149, 150, 147, 149, 140, 137, 153, 153, 153, 153, 152, 153, 156, 162, 161, 163, 159, 155, 156, 152, 151, 154, 156]
green = [126, 126, 125, 124, 124, 126, 112, 111, 118, 117, 118, 118, 111, 116, 115, 119, 119, 113, 111, 113, 113, 115, 115, 116, 116, 106, 107, 110, 111, 104, 107, 108, 116, 115, 105, 105]
red = [104, 104, 105, 103, 104, 104, 47, 44, 71, 70, 64, 64, 50, 48, 49, 55, 64, 54, 51, 44, 42, 50, 47, 52, 49, 34, 37, 40, 38, 35, 35, 34, 48, 51, 33, 35]

# 7.3 计算硫酸铝钾数据的灰度值
gray = 0.299 * np.array(red) + 0.587 * np.array(green) + 0.114 * np.array(blue)  # RGB转灰度公式

# 7.4 绘制硫酸铝钾RGB三通道散点图
plt.figure()
plt.scatter(concentration_repeated, blue, c='b', marker='o', label='蓝色')  # 蓝色通道散点
plt.scatter(concentration_repeated, green, c='g', marker='s', label='绿色')  # 绿色通道散点
plt.scatter(concentration_repeated, red, c='r', marker='^', label='红色')  # 红色通道散点
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel('颜色读数')  # 设置y轴标签
plt.title('颜色读数与硫酸铝钾浓度的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 7.5 绘制硫酸铝钾灰度值与线性拟合
plt.figure()
plt.scatter(concentration_repeated, gray, c='k', marker='d', label='灰度值')  # 灰度值散点

# 7.6 进行线性拟合
p = np.polyfit(concentration_repeated, gray, 1)  # 一次多项式拟合
yfit = np.polyval(p, concentration_repeated)  # 计算拟合值
plt.plot(concentration_repeated, yfit, 'r-', label='线性拟合')  # 线性拟合曲线
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel('灰度值')  # 设置y轴标签
plt.title('灰度值与硫酸铝钾浓度的关系')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# ========== 第八步：硫酸铝钾色调和饱和度分析 ==========
# 8.1 定义色调和饱和度数据
hue = [76, 74, 78, 73, 75, 72, 100, 100, 98, 98, 98, 97, 99, 99, 100, 99, 100, 99, 99, 101, 101, 101, 100, 100, 100, 102, 103, 102, 102, 102, 103, 103, 100, 100, 102, 102]
saturation = [44, 45, 40, 42, 39, 45, 174, 178, 123, 122, 134, 135, 161, 172, 171, 159, 145, 156, 160, 180, 184, 171, 176, 167, 171, 199, 196, 190, 194, 198, 198, 198, 174, 168, 199, 197]

# 8.2 对色调进行二次多项式拟合
p_hue = np.polyfit(concentration_repeated, hue, 2)  # 色调二次拟合
yfit_hue = np.polyval(p_hue, concentration_repeated)  # 计算色调拟合值

# 8.3 对饱和度进行二次多项式拟合
p_saturation = np.polyfit(concentration_repeated, saturation, 2)  # 饱和度二次拟合
yfit_saturation = np.polyval(p_saturation, concentration_repeated)  # 计算饱和度拟合值

# 8.4 绘制色调二次拟合结果
plt.figure()
plt.scatter(concentration_repeated, hue, c='b', marker='o', label='色调')  # 色调散点
plt.plot(concentration_repeated, yfit_hue, 'r-', label='二次拟合曲线')  # 色调拟合曲线
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel('色调')  # 设置y轴标签
plt.title('色调 - 二次多项式拟合')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 8.5 绘制饱和度二次拟合结果
plt.figure()
plt.scatter(concentration_repeated, saturation, c='r', marker='s', label='饱和度')  # 饱和度散点
plt.plot(concentration_repeated, yfit_saturation, 'b-', label='二次拟合曲线')  # 饱和度拟合曲线
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel('饱和度')  # 设置y轴标签
plt.title('饱和度 - 二次多项式拟合')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# ========== 第九步：Michaelis-Menten模型拟合 ==========
# 9.1 对色调进行Michaelis-Menten拟合
initial_guess = [max(hue), 1]  # 初始参数猜测：最大值和Km
hue_fit_params, _ = curve_fit(michaelis_menten, concentration_repeated, hue, p0=initial_guess)  # 拟合参数
fitted_hue = michaelis_menten(concentration_repeated, *hue_fit_params)  # 计算拟合值

# 9.2 绘制色调Michaelis-Menten拟合结果
plt.figure()
plt.scatter(concentration_repeated, hue, c='b', marker='o', label='原始数据')  # 原始色调数据
plt.plot(concentration_repeated, fitted_hue, 'r-', label='Michaelis-Menten拟合')  # 拟合曲线
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel('色调')  # 设置y轴标签
plt.title('色调与硫酸铝钾浓度的Michaelis-Menten拟合')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 9.3 对饱和度进行Michaelis-Menten拟合
initial_guess = [max(saturation), 1]  # 初始参数猜测：最大值和Km
saturation_fit_params, _ = curve_fit(michaelis_menten, concentration_repeated, saturation, p0=initial_guess)  # 拟合参数
fitted_saturation = michaelis_menten(concentration_repeated, *saturation_fit_params)  # 计算拟合值

# 9.4 绘制饱和度Michaelis-Menten拟合结果
plt.figure()
plt.scatter(concentration_repeated, saturation, c='r', marker='s', label='原始数据')  # 原始饱和度数据
plt.plot(concentration_repeated, fitted_saturation, 'b-', label='Michaelis-Menten拟合')  # 拟合曲线
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel('饱和度')  # 设置y轴标签
plt.title('饱和度与硫酸铝钾浓度的Michaelis-Menten拟合')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# ========== 第十步：拟合效果比较和最优模型选择 ==========
# 10.1 计算残差平方和进行模型比较
rss_hue = np.sum((hue - fitted_hue) ** 2)  # 色调拟合的残差平方和
rss_saturation = np.sum((saturation - fitted_saturation) ** 2)  # 饱和度拟合的残差平方和

# 10.2 选择拟合效果更好的模型并绘制
plt.figure()
if rss_hue < rss_saturation:  # 比较残差平方和，选择更好的拟合
    better_fit = '色调'  # 色调拟合更好
    better_fit_params = hue_fit_params  # 使用色调拟合参数
    plt.scatter(concentration_repeated, hue, c='b', marker='o', label='原始数据')  # 绘制色调数据
    plt.plot(concentration_repeated, fitted_hue, 'r-', label='Michaelis-Menten拟合')  # 绘制色调拟合曲线
else:
    better_fit = '饱和度'  # 饱和度拟合更好
    better_fit_params = saturation_fit_params  # 使用饱和度拟合参数
    plt.scatter(concentration_repeated, saturation, c='r', marker='s', label='原始数据')  # 绘制饱和度数据
    plt.plot(concentration_repeated, fitted_saturation, 'b-', label='Michaelis-Menten拟合')  # 绘制饱和度拟合曲线

# 10.3 设置最优拟合图表属性
plt.xlabel('浓度 (mol/L)')  # 设置x轴标签
plt.ylabel(better_fit)  # 设置y轴标签为更好的拟合类型
plt.title(f'{better_fit}与硫酸铝钾浓度的Michaelis-Menten拟合')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表

# 10.4 输出最优拟合结果
print(f'{better_fit}的拟合效果较好：')  # 输出拟合效果较好的类型
print(f'拟合方程: y = ({better_fit_params[0]} * [S]) / ({better_fit_params[1]} + [S])')  # 输出拟合方程
