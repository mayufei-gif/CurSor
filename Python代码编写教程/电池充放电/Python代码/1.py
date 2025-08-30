# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import pandas as pd  # 数据处理和分析
import matplotlib.pyplot as plt  # 数据可视化
import numpy as np  # 数值计算
from scipy.optimize import curve_fit  # 曲线拟合

# ========== 第二步：配置中文显示环境 ==========
# 2.1 设置字体以支持中文显示（并列关系）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 第三步：数据读取和预处理 ==========
# 3.1 读取Excel文件中的电池放电数据
data = pd.read_excel(r'D:\山西机电\数学建模\2024培训内容\电池充放电\简单处理后的表格1.xlsx', skiprows=0)

# 3.2 数据信息输出
print("使用的数据有", len(data.columns), "列")

# 3.3 重新定义列名以便于理解
data.columns = ['time', '20A', '30A', '40A', '50A', '60A', '70A', '80A', '90A', '100A'] + list(data.columns[10:])

# ========== 第四步：数据可视化 ==========
# 4.1 创建图形窗口
plt.figure(figsize=(10, 6))

# 4.2 绘制不同电流强度下的放电曲线
for column in data.columns[1:]:  # 跳过'time'列，只处理电压列
    plt.plot(data['time'], data[column], label=column)  # 绘制每条放电曲线

# 4.3 设置图形属性（并列关系）
plt.xlabel('时间 (s)')  # 设置x轴标签
plt.ylabel('电压 (V)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.title('不同电流强度下的放电曲线')  # 设置图形标题
plt.show()  # 显示图形

# ========== 第五步：定义拟合函数 ==========
# 5.1 定义多项式函数作为拟合函数
def poly_func(t, a1, b1, c1):
    """
    二次多项式拟合函数
    
    参数:
    t: 时间变量
    a1, b1, c1: 多项式系数
    
    返回:
    拟合的电压值
    """
    return a1 * t ** 2 + b1 * t + c1

# ========== 第六步：设置分析参数 ==========
# 6.1 预测目标电压
V_target = 9.8  # 目标电压值（V）

# ========== 第七步：逐个电流强度进行分析 ==========
# 7.1 遍历每个电流强度，分别计算MRE并输出
for column in data.columns[1:]:
    print(f"\n=== 正在分析 {column} 电流强度 ===")
    
    # 7.2 提取当前电流强度的数据
    time = data['time']  # 时间数据
    voltage = data[column]  # 对应电流强度的电压数据

    # 7.3 数据清洗：移除含有 NaN 或 inf 的数据点
    valid_mask = np.isfinite(time) & np.isfinite(voltage)  # 创建有效数据掩码
    time = time[valid_mask]  # 筛选有效时间数据
    voltage = voltage[valid_mask]  # 筛选有效电压数据

    # 7.4 对当前电流强度进行曲线拟合
    params, _ = curve_fit(poly_func, time, voltage)  # 使用最小二乘法拟合
    a1, b1, c1 = params  # 提取拟合参数
    
    print(f"拟合参数: a1={a1:.6f}, b1={b1:.6f}, c1={c1:.6f}")

    # 7.5 计算预测时间和平均相对误差
    # 使用插值方法计算预测时间
    predicted_time = np.interp(voltage[::-1], poly_func(time, a1, b1, c1)[::-1], time[::-1])[::-1]

    # 7.6 计算平均相对误差（MRE）
    MRE = np.mean(np.abs(predicted_time - time) / time)
    print(f'对于{column}电流强度，平均相对误差 MRE = {MRE:.4f}')

    # 7.7 预测目标电压对应的剩余放电时间
    # 使用插值方法计算目标电压对应的时间
    remaining_time = np.interp(V_target, poly_func(time, a1, b1, c1)[::-1], time[::-1])
    print(f'对于{column}电流强度，电压为{V_target}V时的剩余放电时间: {remaining_time:.2f} s')

print("\n=== 电池放电分析完成 ===")
