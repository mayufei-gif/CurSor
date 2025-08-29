import pandas as pd  # 表格数据处理
import matplotlib.pyplot as plt  # 绘图
import numpy as np  # 数值计算
from scipy.optimize import curve_fit  # 拟合

# 先：设置中文字体；同时：允许坐标轴显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 然后：读取Excel数据（路径需按需替换）
数据表 = pd.read_excel(r'D:\山西机电\数学建模\2024培训内容\电池充放电\简单处理后的表格1.xlsx', skiprows=0)

# 打印列数
print("使用的数据有", len(数据表.columns), "列")

# 先：统一列名（时间 + 各电流电压列）
数据表.columns = ['时间', '20A', '30A', '40A', '50A', '60A', '70A', '80A', '90A', '100A'] + list(数据表.columns[10:])

# 数据可视化：先创建画布→再并列绘制各电流曲线
plt.figure(figsize=(10, 6))
for 列名 in 数据表.columns[1:]:  # 跳过“时间”列
    plt.plot(数据表['时间'], 数据表[列名], label=列名)

plt.xlabel('时间 (s)')
plt.ylabel('电压 (V)')
plt.legend()
plt.title('不同电流强度下的放电曲线')
plt.show()

# 定义：二次多项式拟合函数
def 多项式函数(时间序列, 系数a, 系数b, 系数c):
    return 系数a * 时间序列 ** 2 + 系数b * 时间序列 + 系数c

# 目标电压（用于估算剩余时间）
目标电压 = 9.8

// 遍历每个电流强度（先清洗→后拟合→再评估与估算）
for 列名 in 数据表.columns[1:]:
    时间 = 数据表['时间']
    电压 = 数据表[列名]

    # 移除含有 NaN 或 inf 的数据点
    合法掩码 = np.isfinite(时间) & np.isfinite(电压)
    时间 = 时间[合法掩码]
    电压 = 电压[合法掩码]

    # 对每个电流强度进行拟合
    参数, _ = curve_fit(多项式函数, 时间, 电压)
    系数a, 系数b, 系数c = 参数

    # 使用拟合模型预测每个电压点的时间
    预测时间 = np.interp(电压[::-1], 多项式函数(时间, 系数a, 系数b, 系数c)[::-1], 时间[::-1])[::-1]

    # 计算平均相对误差MRE
    平均相对误差 = np.mean(np.abs(预测时间 - 时间) / 时间)
    print(f'对于{列名}电流强度，平均相对误差 MRE = {平均相对误差:.4f}')

    # 使用拟合模型预测剩余放电时间
    剩余时间 = np.interp(目标电压, 多项式函数(时间, 系数a, 系数b, 系数c)[::-1], 时间[::-1])
    print(f'对于{列名}电流强度，电压为{目标电压}V时的剩余放电时间: {剩余时间:.2f} s')
