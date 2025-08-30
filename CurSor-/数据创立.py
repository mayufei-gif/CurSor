import pandas as pd
import numpy as np
import os

# 设置随机种子确保可重复性
np.random.seed(42)

# 创建20行6列的数据
n_samples = 20
data = []

for i in range(n_samples):
    row = [
        # 奇数列（第1,3,5列）- 输入特征
        round(np.random.uniform(15, 35), 1),      # 温度
        round(np.random.uniform(30, 80), 1),      # 湿度
        round(np.random.uniform(0, 10), 1),       # 风速
        
        # 偶数列（第2,4,6列）- 输出结果
        round(np.random.uniform(100, 500), 1),     # 能耗
        round(np.random.uniform(50, 200), 1),       # 成本
        round(np.random.uniform(0.6, 0.95), 2)      # 效率
    ]
    data.append(row)

# 创建DataFrame
df = pd.DataFrame(data, columns=[
    'Temperature', 'EnergyConsumption', 'Humidity', 'Cost', 'WindSpeed', 'Efficiency'
])

# 保存到Excel文件
output_path = 'f:/数学建模/授课内容/odd_even_test_data.xlsx'
df.to_excel(output_path, index=False)

print(f"测试数据已创建：{output_path}")
print(f"数据形状：{df.shape}")
print("\n前5行数据：")
print(df.head())
print("\n奇数列（第1,3,5列）：")
print(df.iloc[:, [0, 2, 4]].head())
print("\n偶数列（第2,4,6列）：")
print(df.iloc[:, [1, 3, 5]].head())