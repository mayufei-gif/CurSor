# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import numpy as np  # 数值计算和数组操作
import pandas as pd  # 数据处理和分析
from scipy.linalg import eig  # 线性代数计算（虽然本代码中未使用）

# ========== 第二步：数据读取和预处理 ==========
# 2.1 读取NBA赛程数据
data = pd.read_excel(r'F:\NBA数据.xlsx')  # 读取Excel文件

# 2.2 提取关键数据列（并列关系）
day = data.iloc[:, 0].values  # 比赛日期列
ke = data.iloc[:, 1].values   # 客队编号列
zhu = data.iloc[:, 2].values  # 主队编号列

# ========== 第三步：构建赛程矩阵 ==========
# 3.1 构建客队赛程矩阵
a = np.zeros((ke.max(), day.max()))  # 初始化客队矩阵
for i in range(len(ke)):
    if ke[i] <= ke.max() and day[i] <= day.max():
        a[ke[i]-1, day[i]-1] = 1  # 标记客队比赛日

# 3.2 构建主队赛程矩阵
b = np.zeros((zhu.max(), day.max()))  # 初始化主队矩阵
for i in range(len(zhu)):
    if zhu[i] <= zhu.max() and day[i] <= day.max():
        b[zhu[i]-1, day[i]-1] = 1  # 标记主队比赛日

# 3.3 构建总赛程矩阵
c = a + b  # 合并主客场赛程，1表示有比赛

# ========== 第四步：构建对手信息矩阵 ==========
# 4.1 构建对手日程表
ah = np.zeros((ke.max(), day.max()))  # 初始化对手矩阵
for i in range(len(ke)):
    if ke[i] <= ke.max() and day[i] <= day.max():
        ah[ke[i]-1, day[i]-1] = zhu[i]  # 记录客队对应的主队编号

# ========== 第五步：计算背靠背比赛次数 ==========
# 5.1 计算主客场背靠背次数
bkb = np.zeros(30)  # 初始化背靠背计数数组
for i in range(30):
    for j in range(c.shape[1] - 1):  # 遍历每一天（避免越界）
        if c[i, j] == 1 and c[i, j+1] == 1:  # 连续两天都有比赛
            bkb[i] += 1  # 背靠背次数加1

# 5.2 计算客场背靠背次数
bkb_guest = np.zeros(30)  # 初始化客场背靠背计数数组
for i in range(30):
    for j in range(a.shape[1] - 1):  # 遍历每一天（避免越界）
        if a[i, j] == 1 and a[i, j+1] == 1:  # 连续两天都是客场比赛
            bkb_guest[i] += 1  # 客场背靠背次数加1

# 5.3 计算主场背靠背次数
bkb_home = np.zeros(30)  # 初始化主场背靠背计数数组
for i in range(30):
    for j in range(b.shape[1] - 1):  # 遍历每一天（避免越界）
        if b[i, j] == 1 and b[i, j+1] == 1:  # 连续两天都是主场比赛
            bkb_home[i] += 1  # 主场背靠背次数加1

# ========== 第六步：计算连续对强队比赛次数 ==========
# 6.1 定义强队范围
qiang = list(range(1, 16))  # 强队编号：1-15

# 6.2 标记对强队的比赛
chh = np.isin(ah, qiang)  # 判断对手是否为强队

# 6.3 计算连续对强队比赛次数
lxqd = np.zeros(30)  # 初始化连续对强队计数数组
for i in range(30):
    for j in range(chh.shape[1] - 1):  # 遍历每一天（避免越界）
        if chh[i, j] and chh[i, j+1]:  # 连续两天都对强队
            lxqd[i] += 1  # 连续对强队次数加1

# ========== 第七步：计算连续对弱队比赛次数 ==========
# 7.1 定义弱队范围
qiang = list(range(16, 31))  # 弱队编号：16-30

# 7.2 标记对弱队的比赛
chh = np.isin(ah, qiang)  # 判断对手是否为弱队

# 7.3 计算连续对弱队比赛次数
lxrd = np.zeros(30)  # 初始化连续对弱队计数数组
for i in range(30):
    for j in range(chh.shape[1] - 1):  # 遍历每一天（避免越界）
        if chh[i, j] and chh[i, j+1]:  # 连续两天都对弱队
            lxrd[i] += 1  # 连续对弱队次数加1

# ========== 第八步：计算比赛间隔天数标准差 ==========
# 8.1 计算每支队伍的比赛间隔标准差
bzc = np.zeros(30)  # 初始化间隔标准差数组
for i in range(30):
    xu = np.where(c[i, :] == 1)[0]  # 找到该队所有比赛日
    cha = np.diff(xu)  # 计算相邻比赛日的间隔
    if len(cha) > 0:
        bzc[i] = np.std(cha)  # 计算间隔的标准差

# ========== 第九步：计算赛程均衡度 ==========
# 9.1 计算每支队伍的赛程均衡度（与间隔标准差相同的计算方法）
jhd = np.zeros(30)  # 初始化均衡度数组
for j in range(30):
    xu = np.where(c[j, :] == 1)[0]  # 找到该队所有比赛日
    cha = np.diff(xu)  # 计算相邻比赛日的间隔
    if len(cha) > 0:
        jhd[j] = np.std(cha)  # 计算间隔的标准差作为均衡度指标

# ========== 第十步：整理和输出结果 ==========
# 10.1 创建结果DataFrame
result_df = pd.DataFrame({
    '队伍编号': np.arange(1, 31),  # 队伍编号1-30
    '主客场背靠背次数': bkb,  # 总背靠背次数
    '客场背靠背次数': bkb_guest,  # 客场背靠背次数
    '主场背靠背次数': bkb_home,  # 主场背靠背次数
    '连续对强队比赛次数': lxqd,  # 连续对强队次数
    '连续对弱队比赛次数': lxrd,  # 连续对弱队次数
    '间隔天数标准差': bzc,  # 比赛间隔标准差
    '均衡度': jhd  # 赛程均衡度
})

# 10.2 保存结果到Excel文件
output_path = (r'F:\NBA处理结果.xlsx')  # 输出文件路径
result_df.to_excel(output_path, index=False)  # 保存为Excel文件，不包含索引

# 10.3 输出完成信息
print("NBA赛程分析完成，结果已保存到:", output_path)
print("\n分析结果概览:")
print(result_df.head())  # 显示前5行结果
