# ================= 0. 模块准备 =================
import numpy as np                           # 导入NumPy库，用于科学计算
import pandas as pd                          # 导入Pandas库，用于数据处理
from scipy.stats import chi2_contingency      # 导入卡方检验函数
import matplotlib.pyplot as plt               # 导入Matplotlib库，用于数据可视化
import seaborn as sns                       # 导入Seaborn库，用于数据可视化
# =============== 1. 生成示例 Excel（身高/体重/年龄） ===============
np.random.seed(2024)                       # 设置随机数种子，确保结果可复现
n = 200                                    # 定义样本量大小为200

df_demo = pd.DataFrame({                   # 创建示例数据框
    '身高_cm': np.round(np.random.normal(170, 8, n), 1),       # 生成正态分布的身高数据，均值170，标准差8
    '体重_kg': np.round(np.random.normal(65, 12, n), 1),       # 生成正态分布的体重数据，均值65，标准差12
    '年龄'   : np.random.randint(18, 65, n)                   # 生成18-65岁的随机年龄数据
})

df_demo.to_excel('person_demo.xlsx', index=False)  # 将生成的数据保存为Excel文件，不包含索引

# =============== 2. 读取 Excel 多列并转为 Python 列表 ===============
file_path = 'person_demo.xlsx'            # 定义Excel文件路径
use_cols  = ['身高_cm', '体重_kg', '年龄']           # 指定需要读取的列名
df = pd.read_excel(file_path, usecols=use_cols)     # 读取Excel文件，只读取指定列

# 转成列表
height_list = df['身高_cm'].tolist()     # 将身高列转换为Python列表
weight_list = df['体重_kg'].tolist()     # 将体重列转换为Python列表
age_list    = df['年龄'].tolist()        # 将年龄列转换为Python列表

# =============== 3. 皮尔逊卡方检验（非参数） ===============
# 3.1 构造体重分级（三档）
df['体重分级'] = pd.cut(df['体重_kg'], bins=[0, 55, 75, 200],  # 将体重分为三档：偏瘦、正常、超重
                        labels=['偏瘦', '正常', '超重'])

# 3.2 构造年龄分组（青年/中年/老年）
df['年龄组'] = pd.cut(df['年龄'], bins=[0, 30, 50, 100],      # 将年龄分为三组：青年、中年、老年
                      labels=['青年', '中年', '老年'])

# 3.3 列联表（交叉表）
contingency = pd.crosstab(df['年龄组'], df['体重分级'])  # 创建年龄组和体重分级的列联表（交叉计数）

# 3.4 皮尔逊卡方检验
chi2_stat, p_val, dof, expected = chi2_contingency(contingency)  # 执行卡方检验，获取统计量、p值、自由度和期望频数

# =============== 4. 结果输出与解释 ===============
print('列联表：')                          # 打印列联表标题
print(contingency)                        # 打印列联表内容
print('\n【皮尔逊卡方检验结果】')           # 打印检验结果标题
print(f'χ² 统计量 = {chi2_stat:.3f}')    # 打印卡方统计量，保留3位小数
print(f'自由度    = {dof}')               # 打印自由度
print(f'p 值      = {p_val:.4f}')         # 打印p值，保留4位小数
if p_val < 0.05:                         # 判断p值是否小于显著性水平0.05
    print('p < 0.05 ⇒ 拒绝 H0，年龄与体重分级**相关**（不独立）。')  # 如果p值小于0.05，拒绝原假设
else:                                    # 否则
    print('p ≥ 0.05 ⇒ 不拒绝 H0，无充分证据表明年龄与体重分级相关。')  # 不拒绝原假设

# =============== 5. 可视化：列联表热力图 ===============
plt.figure(figsize=(6,4))                # 增大图形窗口，设置大小为6x4英寸
sns.heatmap(contingency, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar_kws={'label': '人数'},
            linewidths=0.5,
            square=True)  # 绘制列联表热力图，显示具体数值，使用蓝色系
plt.title('年龄组 × 体重分级 列联表', fontsize=14, pad=20)      # 设置图表标题
plt.xlabel('体重分级', fontsize=12)
plt.ylabel('年龄组', fontsize=12)
plt.tight_layout()
plt.show()                                # 显示图形
