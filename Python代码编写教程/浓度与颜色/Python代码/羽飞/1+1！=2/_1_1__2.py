# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import numpy as np  # 数值计算和数组操作
import pandas as pd  # 数据处理和分析
from sklearn.cluster import KMeans  # K均值聚类算法
import matplotlib.pyplot as plt  # 数据可视化和图表绘制
import seaborn as sns  # 高级统计图表绘制

# ========== 第二步：定义多物质实验数据 ==========
# 以下数据定义为并列关系，构成完整的实验数据集

# 2.1 定义物质类别数组（多种物质检测实验）
substance = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])

# 2.2 定义浓度数组（不同单位）
concentration = np.array([0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5, 7.34, 8.14, 8.74, 9.19, 10.18, 11.8, 0, 0, 0.5, 1, 1.5, 2, 5, 0, 500, 1000, 1500, 2000])

# 2.3 定义颜色数据矩阵（蓝、绿、红、色调、饱和度）
color_data = np.array([
    [68,110,121, 23, 111],   # 物质1-浓度0样本
    [37, 66, 110, 12, 169],  # 物质1-浓度100样本
    [46,87, 117, 16, 155],   # 物质1-浓度50样本
    [62, 99, 120, 19, 122],  # 物质1-浓度25样本
    [66, 102, 118, 20, 112], # 物质1-浓度12.5样本
    [129, 141, 145, 22, 27], # 物质2-浓度0样本
    [7, 133, 145, 27, 241],  # 物质2-浓度100样本
    [60, 133, 141, 27, 145], # 物质2-浓度50样本
    [69, 136, 145, 26, 133], # 物质2-浓度25样本
    [85, 139, 145, 26, 106], # 物质2-浓度12.5样本
    [153, 140, 132, 108, 35],# 物质3-浓度7.34样本
    [151, 142, 133, 104, 29],# 物质3-浓度8.14样本
    [158, 126, 127, 120, 52],# 物质3-浓度8.74样本
    [161,85,118,132,120],    # 物质3-浓度9.19样本
    [127,21,119,147,211],    # 物质3-浓度10.18样本
    [94,6,91,148,237],       # 物质3-浓度11.8样本
    [152, 142, 132, 105, 32],# 物质3-浓度0样本
    [116,126,104,76,44] ,    # 物质4-浓度0样本
    [148,112,47,100,174],    # 物质4-浓度0.5样本
    [149,116,48,99,172],     # 物质4-浓度1样本
    [153,113,44,101,180],    # 物质4-浓度1.5样本
    [156,106,34,102,199],    # 物质4-浓度2样本
    [155,107,34,101,198],    # 物质4-浓度5样本
    [118,136,139,25,37],     # 物质5-浓度0样本
    [117,137,139,27,41],     # 物质5-浓度500样本
    [108,136,138,28,54],     # 物质5-浓度1000样本
    [110,136,139,26,52],     # 物质5-浓度1500样本
    [108,140,142,28,60]      # 物质5-浓度2000样本
])

# ========== 第三步：数据预处理和结构化 ==========
# 3.1 创建DataFrame数据结构
data = pd.DataFrame(color_data, columns=['B', 'G', 'R', 'H', 'S'])  # 创建颜色数据框
data['Concentration'] = concentration  # 添加浓度列

# ========== 第四步：K均值聚类分析 ==========
# 4.1 设置聚类参数
n_clusters = 4  # 设置聚类簇数为4

# 4.2 执行K均值聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # 创建K均值聚类器
data['Cluster'] = kmeans.fit_predict(color_data)  # 执行聚类并添加聚类标签

# ========== 第五步：可视化分析 ==========
# 以下可视化为并列关系，分别展示不同颜色通道与浓度的关系

# 5.1 绘制浓度与红色(R)值的关系图
plt.figure(figsize=(10, 6))  # 设置图表尺寸
sns.scatterplot(x=data['Concentration'], y=data['R'], hue=data['Cluster'], palette='Set1', s=100)  # 绘制散点图
plt.title('浓度与红色(R)值的关系')  # 设置图表标题
plt.xlabel('浓度 (ppm)')  # 设置x轴标签
plt.ylabel('红色(R)值')  # 设置y轴标签
plt.show()  # 显示图表

# 5.2 绘制浓度与绿色(G)值的关系图
plt.figure(figsize=(10, 6))  # 设置图表尺寸
sns.scatterplot(x=data['Concentration'], y=data['G'], hue=data['Cluster'], palette='Set1', s=100)  # 绘制散点图
plt.title('浓度与绿色(G)值的关系')  # 设置图表标题
plt.xlabel('浓度 (ppm)')  # 设置x轴标签
plt.ylabel('绿色(G)值')  # 设置y轴标签
plt.show()  # 显示图表

# 5.3 绘制浓度与蓝色(B)值的关系图
plt.figure(figsize=(10, 6))  # 设置图表尺寸
sns.scatterplot(x=data['Concentration'], y=data['B'], hue=data['Cluster'], palette='Set1', s=100)  # 绘制散点图
plt.title('浓度与蓝色(B)值的关系')  # 设置图表标题
plt.xlabel('浓度 (ppm)')  # 设置x轴标签
plt.ylabel('蓝色(B)值')  # 设置y轴标签
plt.show()  # 显示图表

# ========== 第六步：结论生成函数定义 ==========
def generate_conclusion(data, cluster_centers):
    """
    基于聚类结果生成分析结论
    
    参数:
    data: 包含聚类结果的数据框
    cluster_centers: 聚类中心点数据
    
    返回:
    conclusions: 分析结论列表
    """
    conclusions = []  # 初始化结论列表
    
    # 6.1 分析每个聚类簇的特征
    for i in range(len(cluster_centers)):
        cluster_data = data[data['Cluster'] == i]  # 提取当前簇的数据
        avg_concentration = cluster_data['Concentration'].mean()  # 计算平均浓度
        
        # 6.2 生成基于聚类特征的结论
        conclusion = f"聚类簇 {i}: 平均浓度为 {avg_concentration:.2f}, "
        conclusion += f"颜色特征为 B={cluster_centers.iloc[i]['B']:.1f}, "
        conclusion += f"G={cluster_centers.iloc[i]['G']:.1f}, "
        conclusion += f"R={cluster_centers.iloc[i]['R']:.1f}"
        
        conclusions.append(conclusion)  # 添加到结论列表
    
    return conclusions

# ========== 第七步：结果分析和输出 ==========
# 7.1 获取聚类中心点数据
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['B', 'G', 'R', 'H', 'S'])

# 7.2 生成分析结论
conclusions = generate_conclusion(data, cluster_centers)

# 7.3 输出分析结论
for conclusion in conclusions:
    print("\n结论:", conclusion)
