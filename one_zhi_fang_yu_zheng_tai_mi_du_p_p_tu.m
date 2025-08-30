%% 正态分布优化分析脚本
% 本脚本用于检验数据是否符合正态分布，并提供优化方法

%% 1. 原始数据 - 定义待检验的数据集
x = [93,75,83,93,91,85,84,82,77,76,77,95,94,89,91,88,86,83,96,81,79,97,78,75,67,69,68,84,83,81,75,66,85,70,94,84,83,82,80,78,74,73,76,70,86,76,90,89,71,66,86,73,80,94,79,78,77,63,53,55];

%% 2. 基础统计分析 - 计算并显示数据的基本统计量
fprintf('=== 原始数据统计分析 ===\n');  % 打印分隔标题
fprintf('X数据: 均值=%.2f, 标准差=%.2f\n', mean(x), std(x));  % 显示均值和标准差，保留2位小数

%% 3. 正态性检验 - 使用统计检验方法验证数据是否符合正态分布
% 3.1 Jarque-Bera检验（MATLAB内置）- 基于偏度和峰度的检验方法
fprintf('=== 正态性检验结果 ===\n');  % 打印检验结果标题
[h_jb, p_jb] = jbtest(x);  % 执行Jarque-Bera检验，h为检验结果，p为p值
fprintf('X数据: p值=%.4f, %s\n', p_jb, iif(p_jb > 0.05, '符合正态分布', '不符合正态分布'));  % 根据p值判断是否符合正态分布

% 3.2 Lilliefors检验（Kolmogorov-Smirnov改进版）- 更适用于小样本
[h_lillie, p_lillie] = lillietest(x);  % 执行Lilliefors检验
fprintf('X数据(Lilliefors检验): p值=%.4f, %s\n\n', p_lillie, iif(p_lillie > 0.05, '符合正态分布', '不符合正态分布'));  % 显示Lilliefors检验结果

%% 4. 数据转换（如果需要）- 当数据不符合正态分布时进行转换
% 如果数据不符合正态分布，尝试对数转换
if p_jb <= 0.05 || p_lillie <= 0.05  % 如果任一检验的p值小于0.05，认为不符合正态分布
    x_log = log(x);  % 对数据取自然对数进行转换
    [h_log, p_log] = jbtest(x_log);  % 对转换后的数据再次进行正态性检验
    fprintf('对数转换后: p值=%.4f, %s\n\n', p_log, iif(p_log > 0.05, '符合正态分布', '仍不符合正态分布'));  % 显示转换后的检验结果
else
    x_log = x;  % 如果已符合正态分布，不转换，直接赋值
end

%% 5. 可视化分析 - 通过图形直观展示数据分布特征
figure('Name', '正态分布检验', 'Position', [100, 100, 800, 600]);  % 创建图形窗口，设置标题和位置

% 5.1 原始数据直方图和密度曲线 - 显示数据分布形状
subplot(2, 2, 1);  % 创建2x2子图布局，选择第1个位置
histogram(x, 'Normalization', 'pdf');  % 绘制直方图，使用概率密度归一化
hold on;  % 保持当前图形，允许叠加绘制
x_fit = linspace(min(x), max(x), 100);  % 在数据范围内生成100个等间距点用于拟合
y_fit = normpdf(x_fit, mean(x), std(x));  % 计算正态分布的理论密度值
plot(x_fit, y_fit, 'r-', 'LineWidth', 2);  % 绘制红色正态密度曲线，线宽为2
title('X数据直方图与正态密度');  % 设置子图标题
legend('直方图', '正态密度');  % 添加图例

% 5.2 Q-Q图 - 检验数据是否符合正态分布的图形方法
subplot(2, 2, 2);  % 选择第2个子图位置
normplot(x);  % 绘制正态概率图（Q-Q图）
title('X数据Q-Q图');  % 设置子图标题

% 5.3 转换后数据对比（如果需要）- 显示数据转换后的效果
if p_jb <= 0.05 || p_lillie <= 0.05  % 只有当原始数据不符合正态分布时才显示转换结果
    subplot(2, 2, 3);  % 选择第3个子图位置
    histogram(x_log, 'Normalization', 'pdf');  % 绘制转换后数据的直方图
    hold on;  % 保持当前图形
    x_log_fit = linspace(min(x_log), max(x_log), 100);  % 为转换后数据生成拟合点
    y_log_fit = normpdf(x_log_fit, mean(x_log), std(x_log));  % 计算转换后数据的理论密度值
    plot(x_log_fit, y_log_fit, 'r-', 'LineWidth', 2);  % 绘制红色正态密度曲线
    title('X数据转换后直方图与正态密度');  % 设置子图标题
    legend('直方图', '正态密度');  % 添加图例
    
    subplot(2, 2, 4);  % 选择第4个子图位置
    normplot(x_log);  % 绘制转换后数据的Q-Q图
    title('X数据转换后Q-Q图');  % 设置子图标题
end

%% 6. 概率分布计算 - 计算数据在特定区间的概率
fprintf('=== 概率分布计算 ===\n');  % 打印概率计算标题
% 计算特定区间的概率 - 使用正态分布累积分布函数
P_x = normcdf(90, mean(x), std(x)) - normcdf(70, mean(x), std(x));  % 计算数据在[70,90]区间的概率
fprintf('X数据在[70,90]区间的概率: %.4f\n', P_x);  % 显示计算结果，保留4位小数

%% 7. 逆概率分布计算（分位数）- 计算给定概率对应的分位数值
fprintf('=== 逆概率分布（分位数）===\n');  % 打印分位数计算标题
q_25_x = norminv(0.25, mean(x), std(x));  % 计算25%分位数（下四分位数）
q_75_x = norminv(0.75, mean(x), std(x));  % 计算75%分位数（上四分位数）
fprintf('X数据: 25%%分位数=%.2f, 75%%分位数=%.2f\n', q_25_x, q_75_x);  % 显示分位数结果，保留2位小数

%% 8. 结果总结 - 根据检验结果给出最终结论
fprintf('=== 优化总结 ===\n');  % 打印总结标题
if p_jb > 0.05 && p_lillie > 0.05  % 如果两个检验都认为数据符合正态分布
    fprintf('原始数据已经符合正态分布，无需转换\n');  % 输出无需转换的结论
else  % 否则认为数据需要转换
    fprintf('数据需要进行转换处理\n');  % 输出需要转换的结论
end

%% 辅助函数 - 条件判断函数，用于简化代码
function result = iif(condition, true_val, false_val)  % 定义三目运算函数
if condition  % 如果条件为真
    result = true_val;  % 返回真值
else  % 如果条件为假
    result = false_val;  % 返回假值
end  % 结束条件判断
end  % 结束函数定义