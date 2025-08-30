# 第一步：导入必要的库（按功能分类并列导入）
# 1.1 数据处理库
import pandas as pd  # 用于Excel文件读取和数据处理
# 1.2 绘图库
import matplotlib.pyplot as plt  # 用于图形绘制和显示
# 1.3 表格绘制库
from pandas.plotting import table  # 用于将DataFrame绘制为表格
# 1.4 字体配置库
from matplotlib import rcParams  # matplotlib参数配置

# 第二步：中文字体配置模块（解决中文显示问题）
def set_chinese_font(font_name='SimHei'):
    """
    配置matplotlib支持中文显示的字体设置

    参数:
    - font_name: 字体名称，默认为 'SimHei'（黑体），可以根据需要调整为其他支持中文的字体。
    """
    # 步骤1：字体参数配置（并列设置两个字体参数）
    # 1.1 设置无衬线字体（支持中文字符显示）
    rcParams['font.sans-serif'] = [font_name]
    # 1.2 设置负号正常显示（避免中文字体导致的负号显示问题）
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 第三步：数据读取和插值处理模块（完整的Excel数据处理流程）
def read_and_interpolate_excel(input_path, method='polynomial', order=2):
    """
    读取Excel文件并对数据进行插值处理，填补缺失值

    参数:
    - input_path: Excel 文件的路径。
    - method: 插值方法，默认是 'polynomial'（多项式）。
    - order: 多项式的阶数，默认为 2（即二次多项式）。

    返回:
    - 经过插值处理后的 DataFrame。
    """
    # 步骤1：数据读取
    # 1.1 从Excel文件读取数据
    df = pd.read_excel(input_path)

    # 步骤2：插值函数定义（处理单列数据的插值逻辑）
    def interpolate_column(col):
        # 2.1 尝试使用指定方法进行插值（异常处理机制）
        try:
            # 2.1.1 分支1：使用指定的插值方法和阶数
            return col.interpolate(method=method, order=order)
        except Exception as e:
            # 2.1.2 分支2：插值失败时使用线性插值作为备选方案
            print(f"插值失败，使用线性插值: {e}")
            return col.interpolate(method='linear')

    # 步骤3：数据插值处理
    # 3.1 对DataFrame的每一列应用插值函数
    df_filled = df.apply(interpolate_column, axis=0)
    
    # 步骤4：返回处理结果
    return df_filled

# 第四步：表格可视化和图片保存模块（DataFrame转图片功能）
def save_dataframe_as_image(df, output_image_path, figsize=(12, 8), font_size=12, scale=(1.2, 1.2)):
    """
    将 DataFrame 绘制为表格并保存为图片

    参数:
    - df: 要显示的数据框。
    - output_image_path: 保存图片的路径和文件名。
    - figsize: 图片大小，默认为 (12, 8)。
    - font_size: 表格字体大小，默认为 12。
    - scale: 表格缩放比例，默认为 (1.2, 1.2)。
    """
    # 步骤1：图形对象创建
    # 1.1 创建图形和轴对象（设置图形大小）
    fig, ax = plt.subplots(figsize=figsize)

    # 步骤2：坐标轴隐藏（并列隐藏三个轴元素）
    # 2.1 隐藏x轴
    ax.xaxis.set_visible(False)
    # 2.2 隐藏y轴
    ax.yaxis.set_visible(False)
    # 2.3 隐藏边框
    ax.set_frame_on(False)

    # 步骤3：表格创建和配置
    # 3.1 添加表格到图形中（设置位置和对齐方式）
    tbl = table(ax, df, loc='center', cellLoc='center')

    # 3.2 表格样式设置（按顺序设置三个样式属性）
    # 3.2.1 禁用自动字体大小调整
    tbl.auto_set_font_size(False)
    # 3.2.2 设置表格字体大小
    tbl.set_fontsize(font_size)
    # 3.2.3 设置表格缩放比例
    tbl.scale(*scale)

    # 步骤4：图片保存和显示（按顺序执行保存和显示操作）
    # 4.1 保存表格为高分辨率图片
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    # 4.2 显示图形
    plt.show()

    # 步骤5：成功提示输出
    print(f"数据已成功保存为图片: {output_image_path}")

# 第五步：主函数模块（整合所有功能的程序入口）
def main(input_path, output_image_path, font_name='SimHei', interpolate_method='polynomial', interpolate_order=2):
    """
    主函数，整合设置字体、读取并处理数据、保存图像的步骤

    参数:
    - input_path: Excel 文件的路径。
    - output_image_path: 输出图片的路径。
    - font_name: 字体名称，默认为 'SimHei'。
    - interpolate_method: 插值方法，默认为 'polynomial'（多项式）。
    - interpolate_order: 多项式的阶数，默认为 2（即二次多项式）。
    """
    # 步骤1：环境配置
    # 1.1 设置中文字体支持
    set_chinese_font(font_name)

    # 步骤2：数据处理
    # 2.1 读取Excel文件并进行插值处理
    df_filled = read_and_interpolate_excel(input_path, method=interpolate_method, order=interpolate_order)

    # 步骤3：结果输出
    # 3.1 将处理后的数据保存为图片
    save_dataframe_as_image(df_filled, output_image_path)

# 第六步：程序执行入口（条件性执行主函数）
if __name__ == "__main__":
    # 步骤1：文件路径配置（并列设置两个文件路径）
    # 1.1 设定输入Excel文件路径
    input_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\未处理数据.xlsx'  # 输入文件路径
    # 1.2 设定输出图片文件路径
    output_image_path = r'F:\数学建模国赛-工作项目文件夹\小试牛刀\旭东\2024培训内容\2024培训内容\古塔变形\题目\旭东\补全后的数据.png'  # 输出图片路径

    # 步骤2：程序执行
    # 2.1 执行主函数（使用默认参数）
    main(input_path, output_image_path)