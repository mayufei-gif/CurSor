import numpy as np  # 数值计算
import pandas as pd  # 表格数据
from scipy.linalg import eig  # （未使用，可删除）

# 先：读取赛程数据（列：比赛日/客队编号/主队编号）
数据表 = pd.read_excel(r'F:\NBA数据.xlsx')
比赛日 = 数据表.iloc[:, 0].values
客队 = 数据表.iloc[:, 1].values
主队 = 数据表.iloc[:, 2].values

# 然后：构建客场出场矩阵 a（行：队伍；列：天）
a = np.zeros((客队.max(), 比赛日.max()))
for i in range(len(客队)):
    if 客队[i] <= 客队.max() and 比赛日[i] <= 比赛日.max():
        a[客队[i]-1, 比赛日[i]-1] = 1

# 同时：构建主场出场矩阵 b
b = np.zeros((主队.max(), 比赛日.max()))
for i in range(len(主队)):
    if 主队[i] <= 主队.max() and 比赛日[i] <= 比赛日.max():
        b[主队[i]-1, 比赛日[i]-1] = 1

总出场 = a + b  # 并列：主/客合并

# 然后：构建对手日程表（客队当天的对手=主队编号）
对手表 = np.zeros((客队.max(), 比赛日.max()))
for i in range(len(客队)):
    if 客队[i] <= 客队.max() and 比赛日[i] <= 比赛日.max():
        对手表[客队[i]-1, 比赛日[i]-1] = 主队[i]

# 统计：主客场背靠背（先合并矩阵→再逐队逐日检测）
主客背靠背 = np.zeros(30)
for i in range(30):
    for j in range(总出场.shape[1] - 1):
        if 总出场[i, j] == 1 and 总出场[i, j+1] == 1:
            主客背靠背[i] += 1

# 统计：客场背靠背
客场背靠背 = np.zeros(30)
for i in range(30):
    for j in range(a.shape[1] - 1):
        if a[i, j] == 1 and a[i, j+1] == 1:
            客场背靠背[i] += 1

# 统计：主场背靠背
主场背靠背 = np.zeros(30)
for i in range(30):
    for j in range(b.shape[1] - 1):
        if b[i, j] == 1 and b[i, j+1] == 1:
            主场背靠背[i] += 1

# 统计：连续对强队（1-15）
强队编号 = list(range(1, 16))
是否强队 = np.isin(对手表, 强队编号)
连续强队次数 = np.zeros(30)
for i in range(30):
    for j in range(是否强队.shape[1] - 1):
        if 是否强队[i, j] and 是否强队[i, j+1]:
            连续强队次数[i] += 1

# 统计：连续对弱队（16-30）
弱队编号 = list(range(16, 31))
是否弱队 = np.isin(对手表, 弱队编号)
连续弱队次数 = np.zeros(30)
for i in range(30):
    for j in range(是否弱队.shape[1] - 1):
        if 是否弱队[i, j] and 是否弱队[i, j+1]:
            连续弱队次数[i] += 1

# 统计：间隔天数标准差（先找出赛程天序→再求相邻差→最后求std）
间隔标准差 = np.zeros(30)
for i in range(30):
    天序 = np.where(总出场[i, :] == 1)[0]
    间隔 = np.diff(天序)
    if len(间隔) > 0:
        间隔标准差[i] = np.std(间隔)

# 均衡度（实现同上，等价于间隔标准差；可按需替换定义）
均衡度 = np.zeros(30)
for j in range(30):
    天序 = np.where(总出场[j, :] == 1)[0]
    间隔 = np.diff(天序)
    if len(间隔) > 0:
        均衡度[j] = np.std(间隔)

# 最后：汇总并保存结果
结果表 = pd.DataFrame({
    '队伍编号': np.arange(1, 31),
    '主客场背靠背次数': 主客背靠背,
    '客场背靠背次数': 客场背靠背,
    '主场背靠背次数': 主场背靠背,
    '连续对强队比赛次数': 连续强队次数,
    '连续对弱队比赛次数': 连续弱队次数,
    '间隔天数标准差': 间隔标准差,
    '均衡度': 均衡度
})

# 保存到新的Excel文件，更新路径到实际存储路径
输出路径 = r'F:\NBA处理结果.xlsx'
结果表.to_excel(输出路径, index=False)
