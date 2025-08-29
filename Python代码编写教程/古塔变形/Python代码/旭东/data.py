import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from matplotlib import rcParams


# 模块1：设置字体
def set_chinese_font(font_name='SimHei'):
    """
    设置字体为支持中文的字体。

    参数:
    - font_name: 字体名称，默认为 'SimHei'（黑体），可以根据需要调整为其他支持中文的字体。
    """
    rcParams['font.sans-serif'] = [font_name]
    rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 模块2：读取并处理数据
def read_and_interpolate_excel(input_path, method='polynomial', order=2):
    """
    读取Excel文件并对数据进行插值处理。

    参数:
    - input_path: Excel 文件的路径。
    - method: 插值方法，默认是 'polynomial'（多项式）。
    - order: 多项式的阶数，默认为 2（即二次多项式）。

    返回:
    - 经过插值处理后的 DataFrame。
    """
    df = pd.read_excel(input_path)

    def interpolate_column(col):
        try:
            return col.interpolate(method=method, order=order)
        except Exception as e:
            print(f"插值失败，使用线性插值: {e}")
            return col.interpolate(method='linear')

    df_filled = df.apply(interpolate_column, axis=0)
    return df_filled


# 模块3：绘制表格并保存为图片
def save_dataframe_as_image(df, output_image_path, figsize=(12, 8), font_size=12, scale=(1.2, 1.2)):
    """
    将 DataFrame 绘制为表格并保存为图片。

    参数:
    - df: 要显示的数据框。
    - output_image_path: 保存图片的路径和文件名。
    - figsize: 图片大小，默认为 (12, 8)。
    - font_size: 表格字体大小，默认为 12。
    - scale: 表格缩放比例，默认为 (1.2, 1.2)。
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 隐藏坐标轴
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # 添加表格
    tbl = table(ax, df, loc='center', cellLoc='center')

    # 调整表格字体大小
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(*scale)

    # 保存表格为图片
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"数据已成功保存为图片: {output_image_path}")


# 主函数：整合各个模块
def main(input_path, output_image_path, font_name='SimHei', interpolate_method='polynomial', interpolate_order=2):
    """
    主函数，整合设置字体、读取并处理数据、保存图像的步骤。

    参数:
    - input_path: Excel 文件的路径。
    - output_image_path: 输出图片的路径。
    - font_name: 字体名称，默认为 'SimHei'。
    - interpolate_method: 插值方法，默认为 'polynomial'（多项式）。
    - interpolate_order: 多项式的阶数，默认为 2（即二次多项式）。
    """
    # 设置字体
    set_chinese_font(font_name)

    # 读取并处理数据
    df_filled = read_and_interpolate_excel(input_path, method=interpolate_method, order=interpolate_order)

    # 保存为图片
    save_dataframe_as_image(df_filled, output_image_path)


# 使用示例
if __name__ == "__main__":
    # 设定输入和输出文件路径
    input_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\未处理数据.xlsx'  # 输入文件路径
    output_image_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\补全后的数据.png'  # 输出图片路径

    # 执行主函数
    main(input_path, output_image_path)