import pandas as pd  # 表格数据
import matplotlib.pyplot as plt  # 绘图
import numpy as np  # 数值计算
from scipy.optimize import curve_fit  # 拟合

# 设置字体以支持中文显示
# 先：中文字体→再：坐标轴负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel文件中的数据
数据表 = pd.read_excel(r'D:\山西机电\数学建模\2024培训内容\电池充放电\简单处理后的表格1.xlsx', skiprows=20)

# 将数据的列名转换为电流强度
数据表.columns = ['时间', '20A', '30A', '40A', '50A', '60A', '70A', '80A', '90A', '100A'] + list(数据表.columns[10:])


# 定义多项式函数作为拟合函数
def 多项式函数(时间序列, 系数a, 系数b, 系数c):
    return 系数a * 时间序列 ** 2 + 系数b * 时间序列 + 系数c


# 存储拟合参数的字典
参数字典 = {}
时间 = 数据表['时间']

# 对每个电流强度的放电曲线进行拟合
for 列名 in 数据表.columns[1:]:
    电压 = 数据表[列名]
    合法掩码 = np.isfinite(时间) & np.isfinite(电压)
    时间清洗 = 时间[合法掩码]
    电压清洗 = 电压[合法掩码]

    # 拟合曲线
    参数, _ = curve_fit(多项式函数, 时间清洗, 电压清洗)
    参数字典[列名] = 参数


# 建立以任意电流强度的放电曲线
def 放电曲线模型(电流, 时间点):
    电流列表 = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])
    a值 = np.array([参数字典[f'{c}A'][0] for c in 电流列表])
    b值 = np.array([参数字典[f'{c}A'][1] for c in 电流列表])
    c值 = np.array([参数字典[f'{c}A'][2] for c in 电流列表])

    # 插值得到任意电流强度下的多项式系数
    a插值 = np.interp(电流, 电流列表, a值)
    b插值 = np.interp(电流, 电流列表, b值)
    c插值 = np.interp(电流, 电流列表, c值)

    # 打印55A时的放电曲线方程
    if 电流 == 55:
        print(f"55A时的放电曲线方程: V(t) = {a插值:.4e} * t^2 + {b插值:.4e} * t + {c插值:.4e}")

    # 计算放电曲线
    return 多项式函数(时间点, a插值, b插值, c插值)


# 选择一个特定电流强度 (例如 55A) 来绘制放电曲线
目标电流 = 55
时间点 = np.linspace(时间清洗.min(), 时间清洗.max(), 500)
预测电压 = 放电曲线模型(目标电流, 时间点)

# 计算MRE评估模型精度
# 这里我们选择使用某一个已知电流强度的数据进行评估，假设是30A
实际电压_30A = 多项式函数(时间清洗, *参数字典['30A'])
预测电压_30A = 放电曲线模型(30, 时间清洗)
MRE_30A = np.mean(np.abs(预测电压_30A - 实际电压_30A) / 实际电压_30A)
print(f'30A 电流强度的 MRE = {MRE_30A:.4f}')

# 可视化实际曲线与插值曲线
plt.figure(figsize=(10, 6))
plt.plot(时间清洗, 实际电压_30A, 'b-', label='实际曲线 (30A)')
plt.plot(时间清洗, 预测电压_30A, 'r--', label='插值曲线 (30A)')
plt.xlabel('时间 (s)')
plt.ylabel('电压 (V)')
plt.legend()
plt.title('30A电流强度下的实际与插值放电曲线对比')
plt.show()

# 可视化 55A 的放电曲线
plt.figure(figsize=(10, 6))
plt.plot(时间点, 预测电压, 'g-', label=f'{目标电流}A 插值放电曲线')
plt.xlabel('时间 (s)')
plt.ylabel('电压 (V)')
plt.legend()
plt.title(f'电流强度为 {target_current}A 时的插值放电曲线')
plt.show()
