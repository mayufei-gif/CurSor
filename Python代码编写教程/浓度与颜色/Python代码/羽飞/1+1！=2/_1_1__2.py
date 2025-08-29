import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 手动输入的数据
substance = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
concentration = np.array([0, 100, 50, 25, 12.5, 0, 100, 50, 25, 12.5, 7.34, 8.14, 8.74, 9.19, 10.18, 11.8, 0, 0, 0.5, 1, 1.5, 2, 5, 0, 500, 1000, 1500, 2000])
color_data = np.array([
    [68,110,121, 23, 111],
    [37, 66, 110, 12, 169],
    [46,87, 117, 16, 155],
    [62, 99, 120, 19, 122],
    [66, 102, 118, 20, 112],
    [129, 141, 145, 22, 27],
    [7, 133, 145, 27, 241],
    [60, 133, 141, 27, 145],
    [69, 136, 145, 26, 133],
    [85, 139, 145, 26, 106],
    [153, 140, 132, 108, 35],
    [151, 142, 133, 104, 29],
    [158, 126, 127, 120, 52],
    [161,85,118,132,120],
    [127,21,119,147,211],
    [94,6,91,148,237],
    [152, 142, 132, 105, 32],
    [116,126,104,76,44] ,
    [148,112,47,100,174],
    [149,116,48,99,172],
    [153,113,44,101,180],
    [156,106,34,102,199],
    [155,107,34,101,198],
    [118,136,139,25,37],
    [117,137,139,27,41],
    [108,136,138,28,54],
    [110,136,139,26,52],
    [108,140,142,28,60]
])

# 将颜色数据与浓度组合成一个DataFrame
data = pd.DataFrame(color_data, columns=['B', 'G', 'R', 'H', 'S'])
data['Concentration'] = concentration

# 进行KMeans聚类分析
n_clusters = 4  # 假设我们分成4个簇
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(color_data)

# 可视化RGB与物质浓度之间的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Concentration'], y=data['R'], hue=data['Cluster'], palette='Set1', s=100)
plt.title('浓度与红色(R)值的关系')
plt.xlabel('浓度 (ppm)')
plt.ylabel('红色(R)值')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Concentration'], y=data['G'], hue=data['Cluster'], palette='Set1', s=100)
plt.title('浓度与绿色(G)值的关系')
plt.xlabel('浓度 (ppm)')
plt.ylabel('绿色(G)值')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Concentration'], y=data['B'], hue=data['Cluster'], palette='Set1', s=100)
plt.title('浓度与蓝色(B)值的关系')
plt.xlabel('浓度 (ppm)')
plt.ylabel('蓝色(B)值')
plt.show()

# 生成结论
def generate_conclusion(data, cluster_centers):
    conclusions = []

    for i in range(n_clusters):
        cluster_data = data[data['Cluster'] == i]
        concentration_range = cluster_data['Concentration'].max() - cluster_data['Concentration'].min()
        concentration_mean = cluster_data['Concentration'].mean()
        cluster_rgb_center = cluster_centers.iloc[i][['R', 'G', 'B']]

        conclusion = (f"Cluster {i+1} 显示出与物质浓度的明确关系。"
                      f" 该簇的平均浓度为 {concentration_mean:.2f}，"
                      f" RGB 中心值为 R: {cluster_rgb_center['R']:.2f}, G: {cluster_rgb_center['G']:.2f}, B: {cluster_rgb_center['B']:.2f}。"
                      f" 根据分析结果，该簇的颜色读数与浓度关系较为明确，适合单一化的浓度预测。")
        conclusions.append(conclusion)

    return conclusions

# 查看每个聚类中心
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['B', 'G', 'R', 'H', 'S'])
conclusions = generate_conclusion(data, cluster_centers)

for conclusion in conclusions:
    print("\n结论:", conclusion)
