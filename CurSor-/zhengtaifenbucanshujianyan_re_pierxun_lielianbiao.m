%% ================= 0. 模块准备 =================
clc; clear; close all;                    % 清除命令窗口、工作区和图形窗口
rng(2024);                                % 设置随机数种子，确保结果可复现

%% =============== 1. 生成示例 Excel（身高/体重/年龄） ===============
n = 200;                                  % 定义样本量大小为200

% 生成示例数据
height_cm = round(normrnd(170, 8, n, 1), 1);      % 生成正态分布的身高数据，均值170，标准差8
weight_kg = round(normrnd(65, 12, n, 1), 1);      % 生成正态分布的体重数据，均值65，标准差12
age = randi([18, 65], n, 1);                   % 生成18-65岁的随机年龄数据

% 创建表格并保存为Excel
df_demo = table(height_cm, weight_kg, age);         % 创建MATLAB表格
writetable(df_demo, 'person_demo.xlsx');       % 保存为Excel文件

%% =============== 2. 读取 Excel 多列并转为 MATLAB 数组 ===============
file_path = 'person_demo.xlsx';               % 定义Excel文件路径
df = readtable(file_path);                     % 读取Excel文件

% 提取数据列
height_list = df.height_cm;                      % 身高数据
weight_list = df.weight_kg;                      % 体重数据
age_list = df.age;                            % 年龄数据

%% =============== 3. 皮尔逊卡方检验（非参数） ===============
% 3.1 构造体重分级（三档）
weight_level = discretize(weight_list, [0, 55, 75, 200], ...  % 将体重分为三档
    'categorical', {'偏瘦', '正常', '超重'});

% 3.2 构造年龄分组（青年/中年/老年）
age_group = discretize(age_list, [0, 30, 50, 100], ...      % 将年龄分为三组
    'categorical', {'青年', '中年', '老年'});

% 3.3 列联表（交叉表）
[contingency, chi2_stat, p_val] = crosstab(age_group, weight_level);  % 创建列联表并进行卡方检验
dof = (size(contingency, 1) - 1) * (size(contingency, 2) - 1);  % 计算自由度

%% =============== 4. 结果输出与解释 ===============
fprintf('列联表：\n');                         % 打印列联表标题
disp(contingency);                            % 显示列联表内容
fprintf('\n【皮尔逊卡方检验结果】\n');          % 打印检验结果标题
fprintf('chi2 统计量 = %.3f\n', chi2_stat);     % 打印卡方统计量，保留3位小数
fprintf('自由度    = %d\n', dof);             % 打印自由度
fprintf('p 值      = %.4f\n', p_val);         % 打印p值，保留4位小数

if p_val < 0.05                              % 判断p值是否小于显著性水平0.05
    fprintf('p < 0.05 => 拒绝 H0，年龄与体重分级相关（不独立）。\n');  % 如果p值小于0.05，拒绝原假设
else                                         % 否则
    fprintf('p >= 0.05 => 不拒绝 H0，无充分证据表明年龄与体重分级相关。\n');  % 不拒绝原假设
end

%% =============== 5. 可视化：列联表热力图 ===============
% 创建更美观的图形窗口
figure('Position', [100, 100, 700, 500], 'Color', 'white');    % 增大图形窗口，设置白色背景

% 创建子图，留出更多空间给标题和图例
subplot(1, 1, 1);
ax = gca;
hold on;

% 绘制热力图
h = imagesc(contingency);                        % 绘制热力图
colormap(ax, 'parula');                          % 使用更鲜艳的颜色映射
c = colorbar;                                    % 添加颜色条
c.Label.String = '人数';                         % 设置颜色条标签
c.Label.FontSize = 12;                           % 设置标签字体大小
c.Label.FontWeight = 'bold';                     % 设置标签字体粗细

% 添加数值标注，使用白色背景增强可读性
[rows, cols] = size(contingency);
for i = 1:rows
    for j = 1:cols
        text(j, i, num2str(contingency(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 12, ...
            'FontWeight', 'bold', ...
            'BackgroundColor', [1 1 1 0.5], ...  % 半透明白色背景
            'Margin', 3);                        % 文本边距
    end
end

% 设置坐标轴和标签
set(ax, 'XTick', 1:cols, 'XTickLabel', {'偏瘦', '正常', '超重'});
set(ax, 'YTick', 1:rows, 'YTickLabel', {'青年', '中年', '老年'});
set(ax, 'FontSize', 11, 'FontWeight', 'bold');   % 设置坐标轴字体
set(ax, 'TickLength', [0 0]);                    % 移除刻度线
box on;                                          % 显示边框

% 设置更美观的标题和标签
title('年龄组与体重分级的关联性分析', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'Color', [0.2 0.2 0.6]);                  % 深蓝色标题

xlabel('体重分级', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('年龄组', 'FontSize', 14, 'FontWeight', 'bold');

% 添加图例说明
legend_text = sprintf('卡方值: %.2f, p值: %.4f', chi2_stat, p_val);
annotation('textbox', [0.5, 0.02, 0, 0], ...
    'String', legend_text, ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center');

% 添加结论注释 - 修改这部分，移除FaceColor属性
if p_val < 0.05
    conclusion = '结论: 年龄与体重分级显著相关';
    conclusion_color = [0.8 0.2 0.2];  % 红色表示显著
else
    conclusion = '结论: 年龄与体重分级无显著相关';
    conclusion_color = [0.2 0.6 0.2];  % 绿色表示不显著
end

% 使用text函数替代annotation，避免FaceColor属性问题
t = text(cols/2, -0.5, conclusion, ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'Color', conclusion_color);

% 调整布局
axis equal tight;                            % 设置坐标轴比例和紧凑布局
grid off;                                    % 关闭网格
hold off;

% 优化整体布局
set(gcf, 'Name', '皮尔逊卡方检验可视化分析');
set(gcf, 'NumberTitle', 'off');