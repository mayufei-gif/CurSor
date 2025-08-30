# ========== 第一步：导入必要的库 ==========
# 以下库导入为并列关系，用于不同功能模块
import numpy as np  # 数值计算和数组操作
import pandas as pd  # 数据处理和分析

# ========== 第二步：定义色差矩阵数据 ==========
# 2.1 定义DeltaE色差矩阵（6x6对称矩阵）
# 该矩阵表示不同样本之间的色差值，对角线为0（自身与自身的色差）
deltaE_matrix = np.array([
    [0.0, 20.66983125, 11.91134305, 11.53025216, 10.67268643, 11.25330427],  # 样本1与其他样本的色差
    [20.66983125, 0.0, 8.85030491, 29.0267477, 29.16662701, 29.75434547],   # 样本2与其他样本的色差
    [11.91134305, 8.85030491, 0.0, 21.00471443, 20.89078011, 21.50696746],  # 样本3与其他样本的色差
    [11.53025216, 29.0267477, 21.00471443, 0.0, 2.74564149, 2.55905394],    # 样本4与其他样本的色差
    [10.67268643, 29.16662701, 20.89078011, 2.74564149, 0.0, 0.642395],     # 样本5与其他样本的色差
    [11.25330427, 29.75434547, 21.50696746, 2.55905394, 0.642395, 0.0]      # 样本6与其他样本的色差
])

# ========== 第三步：数据处理和格式化 ==========
# 3.1 对色差矩阵进行数值精度处理
deltaE_matrix_rounded = np.round(deltaE_matrix, 3)  # 保留3位小数，提高数据可读性

# 3.2 转换为DataFrame格式
# 为矩阵添加行列标签，便于数据分析和可视化
df_deltaE_matrix = pd.DataFrame(
    deltaE_matrix_rounded,
    columns=[f'样本{i+1}' for i in range(6)],  # 列标签：样本1-6
    index=[f'样本{i+1}' for i in range(6)]     # 行标签：样本1-6
)

# ========== 第四步：数据展示和输出 ==========
# 4.1 输出处理后的色差矩阵
print("=== 色差分析矩阵 (DeltaE) ===")
print("矩阵说明：")
print("- 对角线元素为0，表示样本与自身的色差")
print("- 矩阵为对称矩阵，表示色差的双向性")
print("- 数值越大表示色差越明显\n")

# 4.2 显示格式化的色差矩阵
print("色差矩阵 (保留3位小数):")
print(df_deltaE_matrix)

# ========== 第五步：色差分析统计 ==========
# 5.1 计算色差统计指标
print("\n=== 色差统计分析 ===")

# 5.2 提取上三角矩阵的色差值（避免重复计算）
upper_triangle_mask = np.triu(np.ones_like(deltaE_matrix_rounded, dtype=bool), k=1)
color_differences = deltaE_matrix_rounded[upper_triangle_mask]

# 5.3 计算统计指标（并列关系）
max_difference = np.max(color_differences)  # 最大色差
min_difference = np.min(color_differences)  # 最小色差
mean_difference = np.mean(color_differences)  # 平均色差
std_difference = np.std(color_differences)  # 色差标准差

# 5.4 输出统计结果
print(f"最大色差值: {max_difference:.3f}")
print(f"最小色差值: {min_difference:.3f}")
print(f"平均色差值: {mean_difference:.3f}")
print(f"色差标准差: {std_difference:.3f}")

# ========== 第六步：色差等级分类 ==========
# 6.1 根据DeltaE值进行色差等级分类
print("\n=== 色差等级分类 ===")
print("色差等级标准：")
print("- DeltaE < 1.0: 人眼几乎无法察觉")
print("- 1.0 ≤ DeltaE < 3.0: 训练有素的眼睛可以察觉")
print("- 3.0 ≤ DeltaE < 6.0: 普通人可以察觉")
print("- DeltaE ≥ 6.0: 明显的色差\n")

# 6.2 统计各等级的色差数量
level_1 = np.sum(color_differences < 1.0)  # 几乎无法察觉
level_2 = np.sum((color_differences >= 1.0) & (color_differences < 3.0))  # 训练有素可察觉
level_3 = np.sum((color_differences >= 3.0) & (color_differences < 6.0))  # 普通人可察觉
level_4 = np.sum(color_differences >= 6.0)  # 明显色差

# 6.3 输出分类统计结果
print(f"几乎无法察觉 (DeltaE < 1.0): {level_1} 对样本")
print(f"训练有素可察觉 (1.0 ≤ DeltaE < 3.0): {level_2} 对样本")
print(f"普通人可察觉 (3.0 ≤ DeltaE < 6.0): {level_3} 对样本")
print(f"明显色差 (DeltaE ≥ 6.0): {level_4} 对样本")

# ========== 第七步：保存分析结果 ==========
# 7.1 将分析结果保存到Excel文件
try:
    # 创建Excel写入器
    with pd.ExcelWriter('color_differences.xlsx', engine='openpyxl') as writer:
        # 保存色差矩阵
        df_deltaE_matrix.to_excel(writer, sheet_name='色差矩阵', index=True)
        
        # 创建统计摘要
        summary_data = {
            '统计指标': ['最大色差', '最小色差', '平均色差', '标准差'],
            '数值': [max_difference, min_difference, mean_difference, std_difference]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='统计摘要', index=False)
        
        print("\n分析结果已保存到 'color_differences.xlsx' 文件")
except Exception as e:
    print(f"\n保存文件时出错: {e}")

print("\n=== 色差分析完成 ===")
