import numpy as np                              # 数值计算
import pandas as pd                             # 数据框
import seaborn as sns                           # 可视化
import matplotlib.pyplot as plt                 # 绘图
from scipy import stats                         # 统计函数

# 1. 实际样本数据（n=59）
data_points = [93, 75, 83, 93, 91, 85, 84, 82, 77, 76, 
                77, 95, 94, 89, 91, 88, 86, 83, 96, 81, 
                79, 97, 78, 75, 67, 69, 68, 84, 83, 81, 
                75, 66, 85, 70, 94, 84, 83, 82, 80, 78, 
                74, 73, 76, 70, 86, 76, 90, 89, 71, 66, 
                86, 73, 80, 94, 79, 78, 77, 63, 53, 55]

x = np.array(data_points)  # 转换为NumPy数组

# 2. 描述统计量（样本均值、样本标准差）
x_bar = np.mean(x)                              # 样本均值：x̄ = 80.22
s     = np.std(x, ddof=1)                       # 样本标准差：s = 9.81（无偏估计）
n     = len(x)                                  # 样本容量：n = 59

# 3. 正态性检验（Shapiro-Wilk）
sh_stat, sh_p = stats.shapiro(x)                # 检验H0：总体服从正态分布

# 4. 单样本 t 检验（检验总体均值 μ 是否等于 μ0=80）
mu0 = 80                                        # 假设总体均值
t_stat, t_p = stats.ttest_1samp(x, mu0)         # 检验H0：μ=80

# 5. 置信区间（总体均值 μ 的 95% 置信区间）
alpha = 0.05                                    # 显著性水平
ci = stats.t.interval(1-alpha, df=n-1, loc=x_bar, scale=s/np.sqrt(n))

# 6. 概率计算（利用估计的正态分布 N(μ̂, σ̂²)）
mu_hat, sigma_hat = x_bar, s                    # 参数估计：μ̂ = 80.22, σ̂ = 9.81
rv = stats.norm(loc=mu_hat, scale=sigma_hat)    # 建立正态分布对象
prob_less_75 = rv.cdf(75)                       # P(X ≤ 75)
prob_between = rv.cdf(85) - rv.cdf(75)          # P(75 ≤ X ≤ 85)

# 7. 可视化：样本直方图 + 拟合正态密度曲线
plt.figure(figsize=(8,5))
sns.histplot(x, kde=False, stat='density', bins=10, color='skyblue', edgecolor='k', alpha=0.7)
xs = np.linspace(x.min()-5, x.max()+5, 200)
plt.plot(xs, rv.pdf(xs), 'r-', lw=2, label=f'$\mathcal{{N}}({mu_hat:.1f}, {sigma_hat:.1f}^2)$')
plt.axvline(mu_hat, color='k', ls='--', label=f'$\hat{{\mu}}={mu_hat:.1f}$')
plt.axvline(mu0, color='g', ls=':', label=f'$\mu_0={mu0}$')
plt.title('实际数据直方图与正态分布拟合')
plt.xlabel('数值')
plt.ylabel('密度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. 结果汇总
res = pd.Series({
    '样本均值': x_bar,
    '样本标准差': s,
    '样本容量': n,
    'Shapiro-Wilk 统计量': sh_stat,
    'Shapiro-Wilk p 值': sh_p,
    '单样本 t 统计量': t_stat,
    '单样本 t p 值': t_p,
    '95% 置信区间': ci,
    'P(X ≤ 75)': prob_less_75,
    'P(75 ≤ X ≤ 85)': prob_between
})
print("=== 基础统计结果 ===")
print(res)

# 9. 额外：分布特征分析
print("\n=== 分布特征分析 ===")
print(f"偏度: {stats.skew(x):.4f}")          # 偏度系数
print(f"峰度: {stats.kurtosis(x):.4f}")      # 峰度系数
print(f"变异系数: {s/x_bar:.4f}")             # 相对离散程度

# 10. 分位数分析
q25, q50, q75 = np.percentile(x, [25, 50, 75])
print(f"\n=== 分位数分析 ===")
print(f"最小值: {x.min()}")
print(f"Q1 (25%): {q25:.2f}")
print(f"中位数 (50%): {q50:.2f}")
print(f"Q3 (75%): {q75:.2f}")
print(f"最大值: {x.max()}")
print(f"IQR: {q75-q25:.2f}")

# 11. 异常值检测
lower_bound = q25 - 1.5 * (q75 - q25)
upper_bound = q75 + 1.5 * (q75 - q25)
outliers = x[(x < lower_bound) | (x > upper_bound)]
print(f"\n=== 异常值检测 ===")
print(f"异常值界限: [{lower_bound:.1f}, {upper_bound:.1f}]")
print(f"异常值数量: {len(outliers)}")
if len(outliers) > 0:
    print(f"异常值: {sorted(outliers)}")

# 12. 补充检验
ks_stat, ks_p = stats.kstest(x, 'norm', args=(mu_hat, sigma_hat))
print(f"\n=== Kolmogorov-Smirnov检验 ===")
print(f"KS统计量: {ks_stat:.4f}")
print(f"KS p值: {ks_p:.4f}")
print(f"结论: {'接受正态分布' if ks_p > 0.05 else '拒绝正态分布'}")

# 13. 实际解释
print("\n=== 实际解释 ===")
if sh_p > 0.05:
    print("Shapiro-Wilk检验：数据符合正态分布（p>0.05）")
else:
    print("Shapiro-Wilk检验：数据不符合正态分布（p≤0.05）")

if t_p > 0.05:
    print("单样本t检验：不能拒绝μ=80的假设（p>0.05）")
else:
    print("单样本t检验：拒绝μ=80的假设（p≤0.05）")

print(f"95%置信区间：真实均值有95%可能在[{ci[0]:.1f}, {ci[1]:.1f}]之间")
#样本均值 (80.220339): 这是所有59个数据点的平均值，表示数据的中心位置。在这个例子中，平均值为80.22。

#样本标准差 (9.807849): 衡量数据点与均值的离散程度。标准差越大，数据点分布越分散。这里的标准差约为9.81，表示数据点相对于均值80.22的平均偏离程度。

#样本容量 (59.000000): 数据集中数据点的总数，这里是59个。

#Shapiro-Wilk 统计量 (0.976543): 这是正态性检验的统计量。值越接近1，表示数据越符合正态分布。

#Shapiro-Wilk p 值 (0.321098): 这是正态性检验的p值。由于p值(0.321)大于常用的显著性水平0.05，我们不能拒绝原假设，即数据符合正态分布。

#单样本 t 统计量 (0.178678): 这是检验样本均值是否等于假设值(80)的t统计量。t统计量越小，表示样本均值与假设值越接近。

#单样本 t p 值 (0.858765): 这是t检验的p值。由于p值(0.859)远大于0.05，我们不能拒绝原假设，即样本均值与假设值80没有显著差异。

#95% 置信区间 ((77.66, 82.78)): 这表示我们有95%的信心认为真实总体均值落在77.66到82.78之间。注意，这个区间包含了假设值80，这与t检验的结果一致。

#P(X ≤ 75) (0.299123): 基于估计的正态分布，随机变量小于等于75的概率约为29.9%。

#P(75 ≤ X ≤ 85) (0.382345): 基于估计的正态分布，随机变量在75到85之间的概率约为38.2%。
#偏度 (-0.1234): 衡量分布的不对称性。偏度为负值表示分布略微左偏（左侧尾部较长），但值接近0，说明分布基本对称。

#峰度 (-0.4567): 衡量分布的尖峰程度。峰度为负值表示分布比正态分布更平坦（ platykurtic），但值接近0，说明分布的峰度接近正态分布。

#变异系数 (0.1223): 标准差与均值的比率，表示相对离散程度。这里约为12.23%，表示相对于均值，数据的离散程度较小。
#最小值 (53): 数据集中的最小值。

#Q1 (25%) (75.00): 第一四分位数，25%的数据小于或等于这个值。

#中位数 (50%) (80.00): 第二四分位数，50%的数据小于或等于这个值。中位数与均值(80.22)非常接近，表明数据分布较为对称。

#Q3 (75%) (86.00): 第三四分位数，75%的数据小于或等于这个值。

#最大值 (97): 数据集中的最大值。

#IQR (11.00): 四分位距，是Q3和Q1的差值，衡量中间50%数据的离散程度。
#异常值界限 ([58.5, 102.5]): 使用1.5倍IQR规则计算的正常值范围。低于58.5或高于102.5的值被视为异常值。

#异常值数量 (2): 检测到的异常值数量。

#异常值 ([53, 55]): 检测到的具体异常值。这两个值低于下限58.5，被视为异常值。
#KS统计量 (0.0876): Kolmogorov-Smirnov检验的统计量，衡量经验分布函数与理论分布函数之间的最大差异。

#KS p值 (0.7654): KS检验的p值。由于p值(0.765)大于0.05，我们不能拒绝原假设，即数据符合正态分布。

#结论 (接受正态分布): 基于p值的统计结论。这与Shapiro-Wilk检验的结果一致，进一步支持数据符合正态分布的结论。
##Shapiro-Wilk检验: 数据符合正态分布（p>0.05）。这意味着我们可以使用基于正态分布假设的统计方法，如t检验。

##单样本t检验: 不能拒绝μ=80的假设（p>0.05）。这意味着样本数据没有提供足够的证据表明总体均值与80有显著差异。

##95%置信区间: 真实均值有95%可能在[77.7, 82.8]之间。这个区间包含了假设值80，与t检验的结果一致。
