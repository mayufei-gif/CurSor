import numpy as np
import pandas as pd

# 您提供的矩阵
deltaE_matrix = np.array([
    [0.0, 20.66983125, 11.91134305, 11.53025216, 10.67268643, 11.25330427],
    [20.66983125, 0.0, 8.85030491, 29.0267477, 29.16662701, 29.75434547],
    [11.91134305, 8.85030491, 0.0, 21.00471443, 20.89078011, 21.50696746],
    [11.53025216, 29.0267477, 21.00471443, 0.0, 2.74564149, 2.55905394],
    [10.67268643, 29.16662701, 20.89078011, 2.74564149, 0.0, 0.642395],
    [11.25330427, 29.75434547, 21.50696746, 2.55905394, 0.642395, 0.0]
])

# 保留3位小数
deltaE_matrix_rounded = np.round(deltaE_matrix, 3)

# 转换为DataFrame
df_deltaE_matrix = pd.DataFrame(deltaE_matrix_rounded)

# 展示结果
import ace_tools as tools; tools.display_dataframe_to_user(name="Rounded DeltaE Matrix", dataframe=df_deltaE_matrix)
