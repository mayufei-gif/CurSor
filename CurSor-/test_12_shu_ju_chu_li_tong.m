%% 通用数据处理代码 - 异常值检测与剔除（修复版）
% 适用于数学建模中的原始数据预处理

clc; clear; close all;

%% ========== 输入模块 ==========
% 请在此处修改您的数据输入方式

% 方式1: 直接输入数据矩阵
data = [
    -2.05 -1.75 -1.52 -1.36 -1.23 -1.12 -1.03 -0.95 -0.88 -0.81;
    -0.75 -0.69 -0.63 -0.58 -0.52 -0.47 -0.42 -0.38 -0.33 -0.29;
    -0.25 -0.20 -0.16 -0.12 -0.08 -0.04  0.00  0.04  0.08  0.12;
    0.16  0.20  0.25  0.29  0.33  0.38  0.42  0.47  0.52  0.58;
    0.63  0.69  0.75  0.81  0.88  0.95  1.03  1.12  1.23  1.36;
    ]';
%{
 % 方式2: 从Excel文件读取数据（取消注释使用）
 filename = 'your_data_file.xlsx';
 data = readmatrix(filename);
    %}
    %{
 % 方式3: 从txt文件读取数据（取消注释使用）
 filename = 'your_data_file.txt';
 data = readmatrix(filename);
    %}
    %{
 % 方式4: 手动输入单列数据（取消注释使用）
fprintf('请输入数据个数: ');
n = input('');
data = zeros(n, 1);
for i = 1:n
    fprintf('请输入第%d个数据: ', i);
    data(i) = input('');
end
    %}
    
    %% ========== 数据验证 ==========
    % 验证数据有效性
    if isempty(data)
        error('数据为空，请检查输入！');
    end
    
    if ~isnumeric(data)
        error('数据必须为数值类型！');
    end
    
    %% ========== 数据预处理 ==========
    fprintf('原始数据维度: %d × %d\n', size(data, 1), size(data, 2));
    fprintf('原始数据总数: %d\n', numel(data));
    
    % 检查缺失值
    missing_count = sum(isnan(data(:)));
    fprintf('缺失值数量: %d\n', missing_count);
    
    % 处理缺失值（用均值填充）
    if missing_count > 0
        for j = 1:size(data, 2)
            col_mean = nanmean(data(:, j));
            data(isnan(data(:, j)), j) = col_mean;
        end
        fprintf('已用均值填充缺失值\n');
    end
    
    %% ========== 异常值检测方法 ==========
    % 提供多种异常值检测方法
    
    fprintf('\n========== 异常值检测分析 ==========\n');
    
    % 方法1: 3σ准则（正态分布假设）
    method1_outliers = [];
    for j = 1:size(data, 2)
        col_data = data(:, j);
        mu = mean(col_data);
        sigma = std(col_data);
        if sigma > 0  % 避免标准差为0的情况
            outlier_idx = abs(col_data - mu) > 3 * sigma;
            method1_outliers = [method1_outliers; find(outlier_idx)];
        end
    end
    method1_outliers = unique(method1_outliers);
    fprintf('3σ准则检测到异常值: %d个\n', length(method1_outliers));
    
    % 方法2: 四分位数法（IQR）
    method2_outliers = [];
    for j = 1:size(data, 2)
        col_data = data(:, j);
        Q1 = prctile(col_data, 25);
        Q3 = prctile(col_data, 75);
        IQR = Q3 - Q1;
        if IQR > 0  % 避免IQR为0的情况
            lower_bound = Q1 - 1.5 * IQR;
            upper_bound = Q3 + 1.5 * IQR;
            outlier_idx = col_data < lower_bound | col_data > upper_bound;
            method2_outliers = [method2_outliers; find(outlier_idx)];
        end
    end
    method2_outliers = unique(method2_outliers);
    fprintf('四分位数法检测到异常值: %d个\n', length(method2_outliers));
    
    % 方法3: Z-score方法
    method3_outliers = [];
    for j = 1:size(data, 2)
        col_data = data(:, j);
        if std(col_data) > 0  % 避免标准差为0
            z_scores = abs(zscore(col_data));
            outlier_idx = z_scores > 2.5;  % 阈值可调整
            method3_outliers = [method3_outliers; find(outlier_idx)];
        end
    end
    method3_outliers = unique(method3_outliers);
    fprintf('Z-score方法检测到异常值: %d个\n', length(method3_outliers));
    
    % 方法4: 修正Z-score方法（基于中位数，更稳健）
    method4_outliers = [];
    for j = 1:size(data, 2)
        col_data = data(:, j);
        median_val = median(col_data);
        mad_val = median(abs(col_data - median_val));
        if mad_val > 0  % 修复：避免MAD为0导致的除零错误
            modified_z_scores = 0.6745 * (col_data - median_val) / mad_val;
            outlier_idx = abs(modified_z_scores) > 3.5;
            method4_outliers = [method4_outliers; find(outlier_idx)];
        end
    end
    method4_outliers = unique(method4_outliers);
    fprintf('修正Z-score方法检测到异常值: %d个\n', length(method4_outliers));
    
    %% ========== 异常值可视化（优化版） ==========
    %% ========== 可视化选择菜单 ==========
    fprintf('\n==========可视化选择==========\n');
    fprintf('请选择要生成的图表类型:\n');
    fprintf('1. 箱线图\n');
    fprintf('2. 散点图\n');
    fprintf('3. 直方图（带频率统计和频率多边形）\n');
    fprintf('4. Q-Q图\n');
    fprintf('5. 异常值检测方法对比图\n');
    fprintf('6. 统计信息显示\n');
    fprintf('\n可以选择多个选项，用逗号分隔（例如：1,3 表示生成箱线图和直方图）\n');
    
    plot_choice = input('请输入选择的图表编号: ', 's');
    
    % 解析用户输入
    if isempty(plot_choice)
        fprintf('输入为空，将生成所有图表\n');
        selected_plots = [1, 2, 3, 4, 5, 6];
    else
        % 解析逗号分隔的输入
        selected_plots = str2num(['[' strrep(plot_choice, ',', ' ') ']']);
        % 验证输入有效性
        selected_plots = selected_plots(selected_plots >= 1 & selected_plots <= 6);
        if isempty(selected_plots)
            fprintf('输入无效，将生成所有图表\n');
            selected_plots = [1, 2, 3, 4, 5, 6];
        end
    end
    
    % 确定子图布局
    num_plots = length(selected_plots);
    if num_plots <= 2
        subplot_rows = 1;
        subplot_cols = num_plots;
        fig_width = 400 * num_plots;
        fig_height = 400;
    elseif num_plots <= 4
        subplot_rows = 2;
        subplot_cols = 2;
        fig_width = 800;
        fig_height = 600;
    else
        subplot_rows = 2;
        subplot_cols = 3;
        fig_width = 1200;
        fig_height = 800;
    end
    
    % 创建图形窗口
    figure('Position', [100, 100, fig_width, fig_height]);
    
    % 绘制选中的图表
    plot_index = 1;
    for i = 1:length(selected_plots)
        plot_type = selected_plots(i);
        
        subplot(subplot_rows, subplot_cols, plot_index);
        
        switch plot_type
            case 1  % 箱线图
                if size(data, 2) == 1
                    boxplot(data, 'Labels', {'数据'});
                    title('原始数据箱线图');
                    ylabel('数值');
                else
                    boxplot(data);
                    title('原始数据箱线图');
                    ylabel('数值');
                    xlabel('变量');
                end
                grid on;
                
            case 2  % 散点图
                if size(data, 2) >= 2
                    scatter(data(:, 1), data(:, 2), 'b', 'filled');
                    hold on;
                    if ~isempty(method2_outliers)
                        scatter(data(method2_outliers, 1), data(method2_outliers, 2), 'r', 'filled');
                        legend('正常值', '异常值', 'Location', 'best');
                    end
                    title('散点图（红色为异常值）');
                    xlabel('变量1');
                    ylabel('变量2');
                else
                    % 单变量数据绘制序列图
                    plot(1:length(data), data, 'b-o', 'MarkerSize', 4);
                    hold on;
                    if ~isempty(method2_outliers)
                        plot(method2_outliers, data(method2_outliers), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'red');
                        legend('正常值', '异常值', 'Location', 'best');
                    end
                    title('数据序列图（红色为异常值）');
                    xlabel('数据点索引');
                    ylabel('数值');
                end
                grid on;
                
            case 3  % 直方图（带频率统计和频率多边形）
                % 获取数据
                data_for_hist = data(:, 1);
                
                % 使用Sturges规则计算最优区间数
                n = length(data_for_hist);
                num_bins = max(5, min(20, ceil(1 + log2(n))));
                
                % 计算直方图参数
                [counts, centers] = hist(data_for_hist, num_bins);
                bin_width = centers(2) - centers(1);
                
                % 绘制直方图
                bar(centers, counts, 'hist', 'FaceColor', [0.7 0.9 1], 'EdgeColor', 'black', 'LineWidth', 1, 'BarWidth', 0.8);
                hold on;
                
                % 计算频率
                frequencies = counts / sum(counts);
                
                % 绘制频率多边形（不包含两端的零点）
                yyaxis right;
                plot(centers, frequencies, 'r-o', 'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor', 'red');
                ylabel('频率', 'Color', 'red');
                ylim([0, max(frequencies) * 1.2]);
                
                % 设置左侧y轴
                yyaxis left;
                ylabel('频数', 'Color', 'black');
                ylim([0, max(counts) * 1.3]);
                
                % 添加频数和频率标注
                for j = 1:length(centers)
                    % 频数标注
                    text(centers(j), counts(j) + max(counts)*0.05, num2str(counts(j)), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', 'blue');
                end
                
                title('直方图与频率多边形');
                xlabel('数值');
                legend('频数直方图', '频率多边形', 'Location', 'best');
                grid on;
                
                % 显示统计信息
                text_info = sprintf('样本数: %d\n区间数: %d\n区间宽度: %.3f', n, num_bins, bin_width);
                text(min(centers) + (max(centers) - min(centers)) * 0.02, max(counts) * 0.9, text_info, ...
                    'FontSize', 8, 'BackgroundColor', 'white', 'EdgeColor', 'black');
                
                hold off;
                
            case 4  % Q-Q图
                qqplot(data(:, 1));
                title('Q-Q图（正态性检验）');
                grid on;
                
            case 5  % 异常值检测方法对比图
                method_names = {'3σ准则', '四分位数法', 'Z-score', '修正Z-score'};
                outlier_counts = [length(method1_outliers), length(method2_outliers), ...
                    length(method3_outliers), length(method4_outliers)];
                bar(outlier_counts, 'FaceColor', [0.2 0.6 0.8]);
                set(gca, 'XTickLabel', method_names);
                title('各方法检测到的异常值数量');
                ylabel('异常值个数');
                xtickangle(45);
                grid on;
                
                % 添加数值标注
                for j = 1:length(outlier_counts)
                    text(j, outlier_counts(j) + max(outlier_counts)*0.02, num2str(outlier_counts(j)), ...
                        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
                end
                
            case 6  % 统计信息显示
                axis off;
                stats_text = sprintf(['数据统计信息:\n' ...
                    '总样本数: %d\n' ...
                    '均值: %.4f\n' ...
                    '标准差: %.4f\n' ...
                    '最小值: %.4f\n' ...
                    '最大值: %.4f\n' ...
                    '中位数: %.4f\n' ...
                    '四分位距: %.4f'], ...
                    numel(data), mean(data(:)), std(data(:)), ...
                    min(data(:)), max(data(:)), median(data(:)), iqr(data(:)));
                text(0.1, 0.8, stats_text, 'FontSize', 12, 'VerticalAlignment', 'top', ...
                    'BackgroundColor', 'white', 'EdgeColor', 'black');
                title('数据统计信息');
        end
        
        plot_index = plot_index + 1;
    end
    
    % 调整子图间距
    if num_plots > 1
        sgtitle('数据可视化分析结果', 'FontSize', 14, 'FontWeight', 'bold');
    end
    
    %% ========== 选择异常值剔除方法 ==========
    fprintf('\n==========异常值剔除选择==========\n');
    fprintf('请选择异常值检测方法:\n');
    fprintf('1. 3σ准则\n');
    fprintf('2. 四分位数法（推荐）\n');
    fprintf('3. Z-score方法\n');
    fprintf('4. 修正Z-score方法\n');
    fprintf('5. 手动选择异常值索引\n');
    fprintf('6. 不剔除异常值\n');
    
    method_choice = input('请输入选择的方法编号 (1-6): ');
    
    % 验证输入
    if isempty(method_choice) || ~isnumeric(method_choice) || method_choice < 1 || method_choice > 6
        fprintf('输入无效，使用默认方法（四分位数法）\n');
        method_choice = 2;
    end
    
    switch method_choice
        case 1
            outliers_to_remove = method1_outliers;
            method_name = '3σ准则';
        case 2
            outliers_to_remove = method2_outliers;
            method_name = '四分位数法';
        case 3
            outliers_to_remove = method3_outliers;
            method_name = 'Z-score方法';
        case 4
            outliers_to_remove = method4_outliers;
            method_name = '修正Z-score方法';
        case 5
            fprintf('请输入要剔除的数据行索引（用空格分隔）: ');
            outliers_input = input('', 's');
            outliers_to_remove = str2num(outliers_input);
            % 验证索引有效性
            outliers_to_remove = outliers_to_remove(outliers_to_remove > 0 & outliers_to_remove <= size(data, 1));
            method_name = '手动选择';
        case 6
            outliers_to_remove = [];
            method_name = '不剔除';
        otherwise
            outliers_to_remove = method2_outliers;  % 默认使用四分位数法
            method_name = '四分位数法（默认）';
    end
    
    %% ========== 剔除异常值 ==========
    if ~isempty(outliers_to_remove)
        fprintf('\n使用%s剔除%d个异常值\n', method_name, length(outliers_to_remove));
        fprintf('被剔除的数据行索引: ');
        fprintf('%d ', outliers_to_remove);
        fprintf('\n');
        
        % 显示被剔除的具体数据
        fprintf('\n被剔除的具体数据:\n');
        for i = 1:length(outliers_to_remove)
            row_idx = outliers_to_remove(i);
            if row_idx <= size(data, 1)  % 确保索引有效
                fprintf('第%d行: ', row_idx);
                fprintf('%.4f ', data(row_idx, :));
                fprintf('\n');
            end
        end
        
        % 剔除异常值
        cleaned_data = data;
        cleaned_data(outliers_to_remove, :) = [];
    else
        fprintf('\n未剔除任何异常值\n');
        cleaned_data = data;
    end
    
    %% ========== 结果分析 ==========
    fprintf('\n========== 处理结果分析 ==========\n');
    fprintf('原始数据行数: %d\n', size(data, 1));
    fprintf('处理后数据行数: %d\n', size(cleaned_data, 1));
    fprintf('剔除数据行数: %d\n', size(data, 1) - size(cleaned_data, 1));
    fprintf('数据保留率: %.2f%%\n', size(cleaned_data, 1) / size(data, 1) * 100);
    
    % 统计对比
    fprintf('\n数据统计对比:\n');
    fprintf('%-15s %-12s %-12s %-12s\n', '统计量', '原始数据', '处理后', '变化量');
    fprintf('%-15s %-12.4f %-12.4f %-12.4f\n', '均值', mean(data(:)), mean(cleaned_data(:)), mean(cleaned_data(:)) - mean(data(:)));
    fprintf('%-15s %-12.4f %-12.4f %-12.4f\n', '标准差', std(data(:)), std(cleaned_data(:)), std(cleaned_data(:)) - std(data(:)));
    fprintf('%-15s %-12.4f %-12.4f %-12.4f\n', '最小值', min(data(:)), min(cleaned_data(:)), min(cleaned_data(:)) - min(data(:)));
    fprintf('%-15s %-12.4f %-12.4f %-12.4f\n', '最大值', max(data(:)), max(cleaned_data(:)), max(cleaned_data(:)) - max(data(:)));
    
    
    %{
 %% ========== 保存处理后的数据 ==========
% 保存为txt文件
output_filename = '异常值处理后的数据.txt';

try
    % 创建文件头信息
    fid = fopen(output_filename, 'w');
    if fid == -1
        error('无法创建输出文件');
    end
    
    fprintf(fid, '%% 异常值处理后的数据\n');
    fprintf(fid, '%% 处理时间: %s\n', datestr(now));
    fprintf(fid, '%% 使用方法: %s\n', method_name);
    fprintf(fid, '%% 原始数据行数: %d\n', size(data, 1));
    fprintf(fid, '%% 处理后数据行数: %d\n', size(cleaned_data, 1));
    fprintf(fid, '%% 数据保留率: %.2f%%\n', size(cleaned_data, 1) / size(data, 1) * 100);
    fprintf(fid, '%%\n');
    
    % 保存数据
    for i = 1:size(cleaned_data, 1)
        for j = 1:size(cleaned_data, 2)
            fprintf(fid, '%.6f', cleaned_data(i, j));
            if j < size(cleaned_data, 2)
                fprintf(fid, '\t');  % 制表符分隔
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    fprintf('\n处理后的数据已保存到: %s\n', output_filename);
catch ME
    fprintf('保存文件时出错: %s\n', ME.message);
end

% 同时保存为mat文件（可选）
try
    mat_filename = '异常值处理后的数据.mat';
    save(mat_filename, 'cleaned_data', 'data', 'outliers_to_remove', 'method_name');
    fprintf('数据也已保存为MAT格式: %s\n', mat_filename);
catch ME
    fprintf('保存MAT文件时出错: %s\n', ME.message);
end

%% ========== 生成处理报告 ==========
try
    report_filename = '数据处理报告.txt';
    fid = fopen(report_filename, 'w');
    if fid == -1
        error('无法创建报告文件');
    end
    
    fprintf(fid, '数据异常值处理报告\n');
    fprintf(fid, '==================\n\n');
    fprintf(fid, '处理时间: %s\n', datestr(now));
    fprintf(fid, '使用方法: %s\n\n', method_name);
    
    fprintf(fid, '数据概况:\n');
    fprintf(fid, '原始数据维度: %d × %d\n', size(data, 1), size(data, 2));
    fprintf(fid, '处理后数据维度: %d × %d\n', size(cleaned_data, 1), size(cleaned_data, 2));
    fprintf(fid, '剔除数据行数: %d\n', length(outliers_to_remove));
    fprintf(fid, '数据保留率: %.2f%%\n\n', size(cleaned_data, 1) / size(data, 1) * 100);
    
    if ~isempty(outliers_to_remove)
        fprintf(fid, '被剔除的数据行索引: ');
        fprintf(fid, '%d ', outliers_to_remove);
        fprintf(fid, '\n\n');
    end
    
    fprintf(fid, '统计量对比:\n');
    fprintf(fid, '%-15s %-12s %-12s\n', '统计量', '原始数据', '处理后');
    fprintf(fid, '%-15s %-12.4f %-12.4f\n', '均值', mean(data(:)), mean(cleaned_data(:)));
    fprintf(fid, '%-15s %-12.4f %-12.4f\n', '标准差', std(data(:)), std(cleaned_data(:)));
    fprintf(fid, '%-15s %-12.4f %-12.4f\n', '最小值', min(data(:)), min(cleaned_data(:)));
    fprintf(fid, '%-15s %-12.4f %-12.4f\n', '最大值', max(data(:)), max(cleaned_data(:)));
    fclose(fid);
    
    fprintf('处理报告已保存到: %s\n', report_filename);
catch ME
    fprintf('生成报告时出错: %s\n', ME.message);
end

fprintf('\n========== 处理完成 ==========\n');
fprintf('所有文件已保存到当前目录\n');
    %}
