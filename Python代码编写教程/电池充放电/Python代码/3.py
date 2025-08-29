import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(file_path):
    """读取Excel数据并提取各列"""
    data = pd.read_excel(file_path, sheet_name='附件2', skiprows=20)
    voltage = data.iloc[:, 0]  # 第一列为电压
    new_battery = data.iloc[:, 1] / 60  # 第二列为新电池状态的放电时间，转化为小时
    decay_state_1 = data.iloc[:, 2] / 60  # 第三列为衰减状态1的放电时间，转化为小时
    decay_state_2 = data.iloc[:, 3] / 60  # 第四列为衰减状态2的放电时间，转化为小时
    decay_state_3 = data.iloc[:, 4] / 60  # 第五列为衰减状态3的放电时间，转化为小时

    return voltage, new_battery, decay_state_1, decay_state_2, decay_state_3


def plot_curve(x, y, label, color, degree=2):
    """绘制曲线及其拟合"""
    # 去除空值和异常值
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # 标准化处理，防止数值不稳定
    x_mean = np.mean(x_clean)
    x_std = np.std(x_clean)
    x_normalized = (x_clean - x_mean) / x_std

    try:
        coeffs = np.polyfit(x_normalized, y_clean, degree)
        poly_eqn = np.poly1d(coeffs)
        return x_clean, poly_eqn((x_clean - x_mean) / x_std), coeffs
    except np.linalg.LinAlgError:
        print(f"Warning: {label} 的拟合出现问题，无法完成拟合")
        return x_clean, None, None


def print_sample_data(voltage, new_battery, decay_state_1, decay_state_2, decay_state_3, num_rows=5):
    """打印表格中的前几行数据作为示例"""
    print("表格中的前几行数据:")
    sample_data = pd.DataFrame({
        '电压（V）': voltage.head(num_rows),
        '新电池状态（小时）': new_battery.head(num_rows),
        '衰减状态1（小时）': decay_state_1.head(num_rows),
        '衰减状态2（小时）': decay_state_2.head(num_rows),
        '衰减状态3（小时）': decay_state_3.head(num_rows)
    })
    print(sample_data)


def estimate_remaining_time(coeffs, voltage_threshold):
    """根据拟合系数估算剩余放电时间"""
    poly_eqn = np.poly1d(coeffs)
    roots = np.roots(poly_eqn - voltage_threshold)
    real_roots = [r.real for r in roots if r.imag == 0]
    return max(real_roots) if real_roots else None


def main():
    # 文件路径
    file_path = r'D:\山西机电\数学建模\2024培训内容\电池充放电\简单处理后的表格1.xlsx'

    # 读取数据
    voltage, new_battery, decay_state_1, decay_state_2, decay_state_3 = load_data(file_path)

    # 打印表格中的前几行数据
    print_sample_data(voltage, new_battery, decay_state_1, decay_state_2, decay_state_3)

    # 分别绘制每个状态的散点图
    plt.figure(figsize=(14, 8))

    plt.scatter(new_battery, voltage, label='新电池状态', color='blue')
    plt.scatter(decay_state_1, voltage, label='衰减状态1', color='green')
    plt.scatter(decay_state_2, voltage, label='衰减状态2', color='orange')
    plt.scatter(decay_state_3, voltage, label='衰减状态3', color='red')

    plt.title('电池放电时间与电压关系 - 散点图')
    plt.ylabel('电压（V）')
    plt.xlabel('放电时间（小时）')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 分别绘制每个状态的拟合曲线
    plt.figure(figsize=(14, 8))

    voltage_threshold = 3.0  # 假设电压降到3.0V时剩余时间为0

    x_new, y_new_fit, coeffs_new = plot_curve(new_battery, voltage, '新电池状态', 'blue')
    if y_new_fit is not None:
        plt.plot(x_new, y_new_fit, 'b--', label='新电池状态拟合')

    x_1, y_1_fit, coeffs_1 = plot_curve(decay_state_1, voltage, '衰减状态1', 'green')
    if y_1_fit is not None:
        plt.plot(x_1, y_1_fit, 'g--', label='衰减状态1拟合')

    x_2, y_2_fit, coeffs_2 = plot_curve(decay_state_2, voltage, '衰减状态2', 'orange')
    if y_2_fit is not None:
        plt.plot(x_2, y_2_fit, 'orange', linestyle='--', label='衰减状态2拟合')

    x_3, y_3_fit, coeffs_3 = plot_curve(decay_state_3, voltage, '衰减状态3', 'red')
    if y_3_fit is not None:
        plt.plot(x_3, y_3_fit, 'r--', label='衰减状态3拟合')

    plt.title('电池放电时间与电压关系 - 拟合曲线')
    plt.ylabel('电压（V）')
    plt.xlabel('放电时间（小时）')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 打印每个状态的拟合方程和剩余放电时间
    if coeffs_new is not None:
        print(f"新电池状态的拟合方程: {np.poly1d(coeffs_new)}")
        remaining_time_new = estimate_remaining_time(coeffs_new, voltage_threshold)
        print(f"新电池状态的剩余放电时间（小时）: {remaining_time_new}")

    if coeffs_1 is not None:
        print(f"衰减状态1的拟合方程: {np.poly1d(coeffs_1)}")
        remaining_time_1 = estimate_remaining_time(coeffs_1, voltage_threshold)
        print(f"衰减状态1的剩余放电时间（小时）: {remaining_time_1}")

    if coeffs_2 is not None:
        print(f"衰减状态2的拟合方程: {np.poly1d(coeffs_2)}")
        remaining_time_2 = estimate_remaining_time(coeffs_2, voltage_threshold)
        print(f"衰减状态2的剩余放电时间（小时）: {remaining_time_2}")

    if coeffs_3 is not None:
        print(f"衰减状态3的拟合方程: {np.poly1d(coeffs_3)}")
        remaining_time_3 = estimate_remaining_time(coeffs_3, voltage_threshold)
        print(f"衰减状态3的剩余放电时间（小时）: {remaining_time_3}")


if __name__ == "__main__":
    main()
    plt.pause(0.1)  # 暂停以确保图形显示