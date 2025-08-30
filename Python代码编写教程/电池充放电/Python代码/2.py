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
# 3.1 读取Excel文件中的电池放电数据（跳过前20行）
data = pd.read_excel(r'D:\山西机电\数学建模\2024培训内容\电池充放电\简单处理后的表格1.xlsx', skiprows=20)

# 3.2 重新定义列名以便于理解
data.columns = ['time', '20A', '30A', '40A', '50A', '60A', '70A', '80A', '90A', '100A'] + list(data.columns[10:])

# ========== 第四步：定义拟合函数 ==========
# 4.1 定义多项式函数作为拟合函数
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

# ========== 第五步：批量拟合各电流强度的放电曲线 ==========
# 5.1 初始化存储拟合参数的字典
params_dict = {}  # 存储每个电流强度的拟合参数
time = data['time']  # 提取时间数据

# 5.2 对每个电流强度的放电曲线进行拟合
for column in data.columns[1:]:
    print(f"正在拟合 {column} 电流强度的放电曲线...")
    
    # 5.3 提取当前电流强度的电压数据
    voltage = data[column]
    
    # 5.4 数据清洗：移除无效数据点
    valid_mask = np.isfinite(time) & np.isfinite(voltage)  # 创建有效数据掩码
    time_clean = time[valid_mask]  # 筛选有效时间数据
    voltage_clean = voltage[valid_mask]  # 筛选有效电压数据

    # 5.5 执行曲线拟合
    params, _ = curve_fit(poly_func, time_clean, voltage_clean)  # 使用最小二乘法拟合
    params_dict[column] = params  # 存储拟合参数
    
    print(f"{column} 拟合参数: a1={params[0]:.6f}, b1={params[1]:.6f}, c1={params[2]:.6f}")

# ========== 第六步：建立任意电流强度的放电曲线模型 ==========
# 6.1 定义插值模型函数
def discharge_curve_model(current, time_points):
    """
    基于插值方法建立任意电流强度的放电曲线模型
    
    参数:
    current: 目标电流强度
    time_points: 时间点数组
    
    返回:
    predicted_voltage: 预测的电压值数组
    """
    # 6.2 提取已知电流强度和对应的拟合参数
    currents = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])  # 已知电流强度
    a1_values = np.array([params_dict[f'{c}A'][0] for c in currents])  # 提取a1系数
    b1_values = np.array([params_dict[f'{c}A'][1] for c in currents])  # 提取b1系数
    c1_values = np.array([params_dict[f'{c}A'][2] for c in currents])  # 提取c1系数

    # 6.3 使用线性插值得到任意电流强度下的多项式系数
    a1_interp = np.interp(current, currents, a1_values)  # 插值得到a1系数
    b1_interp = np.interp(current, currents, b1_values)  # 插值得到b1系数
    c1_interp = np.interp(current, currents, c1_values)  # 插值得到c1系数

    # 6.4 输出插值得到的拟合参数
    print(f"{current}A 插值参数: a1={a1_interp:.6f}, b1={b1_interp:.6f}, c1={c1_interp:.6f}")

    # 6.5 使用插值参数计算预测电压
    predicted_voltage = poly_func(time_points, a1_interp, b1_interp, c1_interp)
    
    return predicted_voltage

# ========== 第七步：模型验证和误差分析 ==========
# 7.1 设置目标电流强度进行预测
target_current = 55  # 目标电流强度（A）
time_points = np.linspace(time_clean.min(), time_clean.max(), 500)  # 生成时间点
predicted_voltage = discharge_curve_model(target_current, time_points)  # 预测55A的放电曲线

# 7.2 验证模型准确性：使用30A数据进行验证
print("\n=== 模型验证：30A电流强度 ===")
actual_voltage_30A = poly_func(time_clean, *params_dict['30A'])  # 30A实际拟合曲线
predicted_voltage_30A = discharge_curve_model(30, time_clean)  # 30A插值预测曲线
MRE_30A = np.mean(np.abs(predicted_voltage_30A - actual_voltage_30A) / actual_voltage_30A)  # 计算平均相对误差
print(f'30A 电流强度的 MRE = {MRE_30A:.4f}')

# ========== 第八步：结果可视化 ==========
# 8.1 绘制30A电流强度下的实际与插值放电曲线对比
plt.figure(figsize=(10, 6))
plt.plot(time_clean, actual_voltage_30A, 'b-', label='实际曲线 (30A)', linewidth=2)  # 实际曲线
plt.plot(time_clean, predicted_voltage_30A, 'r--', label='插值曲线 (30A)', linewidth=2)  # 插值曲线
plt.xlabel('时间 (s)')  # x轴标签
plt.ylabel('电压 (V)')  # y轴标签
plt.legend()  # 显示图例
plt.title('30A电流强度下的实际与插值放电曲线对比')  # 图形标题
plt.grid(True, alpha=0.3)  # 添加网格
plt.show()  # 显示图形

# 8.2 绘制55A电流强度的插值放电曲线
plt.figure(figsize=(10, 6))
plt.plot(time_points, predicted_voltage, 'g-', label=f'{target_current}A 插值放电曲线', linewidth=2)  # 插值曲线
plt.xlabel('时间 (s)')  # x轴标签
plt.ylabel('电压 (V)')  # y轴标签
plt.legend()  # 显示图例
plt.title(f'电流强度为 {target_current}A 时的插值放电曲线')  # 图形标题
plt.grid(True, alpha=0.3)  # 添加网格
plt.show()  # 显示图形

print(f"\n=== 任意电流强度放电曲线建模完成 ===")
print(f"已成功建立 {target_current}A 电流强度的放电曲线模型")
