# 导入必要的库：numpy用于数值计算，pandas用于数据处理，matplotlib用于数据可视化
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题

# 1. 生成数据
# 设置随机种子，确保每次运行生成的随机数相同，结果可复现
np.random.seed(42)           
# 生成100个服从正态分布的随机数，均值为50，标准差为10
x = np.random.normal(50, 10, 100)

# 2. 设定区间并分组
# 设置数据分组的起始值和结束值
a, b = 30, 70
# 在30到70之间生成11个等距点，形成10个小区间
bins = np.linspace(a, b, 11)           
# 为每个区间创建标签，格式为"起始值-结束值"
labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(10)]

# 3. 频数统计
# 创建DataFrame来存储数据，列名为'x'
df = pd.DataFrame({'数据': x})
# 使用cut函数将数据分到不同的区间中，并添加分组标签
df['group'] = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
# 计算每个区间的频数（数据点数量），并按区间顺序排序        
freq        = df['group'].value_counts().sort_index()        
# 计算每个区间的相对频率（频数占总数的比例）
rel_freq    = freq / freq.sum()                              
# 计算累积频数（前面所有区间的频数之和）
cum_freq    = freq.cumsum()                                  
# 计算累积相对频率（前面所有区间的相对频率之和）
cum_rel     = rel_freq.cumsum()                              

# 4. 可视化
# 设置绘图风格为seaborn-v0_8，使图表更美观
plt.style.use('seaborn-v0_8')
# 创建1行2列的子图，设置图形大小为12x4英寸
fig, ax = plt.subplots(1, 2, figsize=(12,4))

# 直方图 + 频率多边形
# 在第一个子图中绘制直方图，设置边框颜色为黑色，透明度为0.6
ax[0].hist(x, bins=bins, edgecolor='b', alpha=0.8, label='频数')
# 设置第一个子图的标题为"直方图"
ax[0].set_title('直方图')
# 设置x轴标签为"数值"，y轴标签为"频数"
ax[0].set_xlabel('数值'); ax[0].set_ylabel('频数')

# 计算每个区间的中心点位置，用于绘制频率多边形
bin_centers = (bins[:-1] + bins[1:]) / 2
# 在直方图上叠加绘制频率多边形，使用红色圆点连线
ax[0].plot(bin_centers, freq, 'r-o', label='频率多边形')
# 显示图例
ax[0].legend()

# 累积频率曲线
# 在第二个子图中绘制累积频率曲线，使用绿色方块连线
ax[1].plot(bin_centers, cum_rel, 'g-s')
# 设置第二个子图的标题为"累积频率曲线"
ax[1].set_title('累积频率曲线')
# 设置x轴标签为"数值"，y轴标签为"累积频率"
ax[1].set_xlabel('数值'); ax[1].set_ylabel('累积频率')
# 显示网格线，便于读取数值
ax[1].grid(True)

# 自动调整子图间距，避免重叠
plt.tight_layout()
# 显示图形
plt.show()

# 5. 打印统计表
# 创建汇总表格，包含频数、频率、累积频数、累积频率四列
summary = pd.DataFrame({'频数':freq,'频率':rel_freq,'累积频数':cum_freq,'累积频率':cum_rel})
# 打印统计表到控制台
print(summary)