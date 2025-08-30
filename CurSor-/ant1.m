% 蚁群算法解决TSP问题 - MATLAB实现（优化版）
clear; clc; close all;

% 城市坐标
city_location = containers.Map(...
    {'A31', 'B44', 'C61', 'D83', 'E100', 'F115', 'G147', 'H158'}, ...
    {[111.042637, 38.112507], ...
    [111.073839, 38.117061], ...
    [111.069797, 38.116762], ...
    [111.057493, 38.124064], ...
    [111.044954, 38.119702], ...
    [111.040331, 38.110755], ...
    [111.046929, 38.118705], ...
    [111.065254, 38.127754]});

% 获取城市名称和坐标
cities = keys(city_location);
locations = values(city_location);
locations = cell2mat(locations');

% 计算距离矩阵
n_cities = size(locations, 1);
distances = zeros(n_cities);
for i = 1:n_cities
    for j = 1:n_cities
        distances(i, j) = norm(locations(i, :) - locations(j, :));
    end
end

% 蚁群算法参数
n_ants = 50;
n_best = 1;
n_iterations = 200;
decay = 0.9;
alpha = 1;
beta = 5;

% 初始化信息素矩阵
pheromone = ones(n_cities) / n_cities;

% 初始化最佳路径
all_time_shortest_path = [];
all_time_shortest_length = inf;
best_length_history = [];

% 创建图形窗口
fig = figure('Position', [100, 100, 1200, 800]);

% 创建主绘图区域
subplot(2, 2, [1, 3]);
hold on;
xlabel('X坐标', 'FontSize', 12);
ylabel('Y坐标', 'FontSize', 12);
title('蚁群算法求解TSP问题 - 动态路径优化', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
axis equal;

% 绘制城市点
city_scatter = scatter(locations(:, 1), locations(:, 2), 150, 'filled', ...
    'MarkerFaceColor', [0.8 0.2 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 2);
for i = 1:n_cities
    text(locations(i, 1)+0.0003, locations(i, 2)+0.0003, cities{i}, ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'blue');
end

% 初始化路径绘制对象（当前路径改为虚线，避免遮挡最优路径）
current_path_plot = plot(0, 0, 'b--', 'LineWidth', 1.5, 'DisplayName', '当前路径');
best_path_plot = plot(0, 0, 'r-', 'LineWidth', 3, 'DisplayName', '最优路径');

% 简化图例 - 只显示主要路径
legend([best_path_plot, current_path_plot], 'Location', 'northeast', 'FontSize', 10);

% 创建信息显示区域
info_text = text(min(locations(:, 1)), max(locations(:, 2))+0.003, '', ...
    'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'Margin', 5);

% 创建收敛曲线子图
subplot(2, 2, 2);
convergence_plot = plot(1, inf, 'b-', 'LineWidth', 2);
xlabel('迭代次数', 'FontSize', 10);
ylabel('最短路径长度', 'FontSize', 10);
title('收敛曲线', 'FontSize', 12);
grid on;

% 创建信息素强度热力图子图
subplot(2, 2, 4);
pheromone_heatmap = imagesc(pheromone);
colorbar;
title('信息素强度分布', 'FontSize', 12);
xlabel('城市编号', 'FontSize', 10);
ylabel('城市编号', 'FontSize', 10);
set(gca, 'XTick', 1:n_cities, 'YTick', 1:n_cities);
set(gca, 'XTickLabel', cities, 'YTickLabel', cities);

% 开始迭代
for iter = 1:n_iterations
    % 所有蚂蚁构建路径
    all_paths = cell(n_ants, 2);
    
    for ant = 1:n_ants
        path = zeros(1, n_cities+1);
        visited = false(1, n_cities);
        
        % 随机选择起始城市
        current = randi(n_cities);
        path(1) = current;
        visited(current) = true;
        
        % 构建路径
        for step = 2:n_cities
            % 计算选择概率
            prob = zeros(1, n_cities);
            for next_city = 1:n_cities
                if ~visited(next_city)
                    prob(next_city) = pheromone(current, next_city)^alpha * (1/distances(current, next_city))^beta;
                end
            end
            
            % 选择下一个城市
            prob = prob / sum(prob);
            next_city = randsample(n_cities, 1, true, prob);
            
            path(step) = next_city;
            visited(next_city) = true;
            current = next_city;
        end
        
        % 回到起点
        path(end) = path(1);
        
        % 计算路径长度
        path_length = 0;
        for i = 1:n_cities
            path_length = path_length + distances(path(i), path(i+1));
        end
        
        all_paths{ant, 1} = path;
        all_paths{ant, 2} = path_length;
    end
    
    % 更新信息素
    pheromone = pheromone * decay;
    
    % 找出最佳路径并更新信息素
    [shortest_length, idx] = min([all_paths{:, 2}]);
    shortest_path = all_paths{idx, 1};
    
    % 更新全局最佳路径
    if shortest_length < all_time_shortest_length
        all_time_shortest_length = shortest_length;
        all_time_shortest_path = shortest_path;
    end
    
    % 记录历史最优值
    best_length_history(iter) = all_time_shortest_length;
    
    % 增强最佳路径上的信息素
    for i = 1:n_cities
        pheromone(shortest_path(i), shortest_path(i+1)) = pheromone(shortest_path(i), shortest_path(i+1)) + 1/shortest_length;
        pheromone(shortest_path(i+1), shortest_path(i)) = pheromone(shortest_path(i+1), shortest_path(i)) + 1/shortest_length;
    end
    
    % 动态可视化更新（每5次迭代更新一次以提高速度）
    if mod(iter, 5) == 1 || iter == n_iterations
        subplot(2, 2, [1, 3]);
        
        % 先更新最优路径（实线，在下层）
        if ~isempty(all_time_shortest_path)
            set(best_path_plot, 'XData', locations(all_time_shortest_path, 1), 'YData', locations(all_time_shortest_path, 2));
            
            % 简化箭头显示 - 只在最优路径更新时显示少量箭头
            if shortest_length < all_time_shortest_length || iter == n_iterations
                % 清除之前的箭头
                h_arrows = findobj(gca, 'Type', 'line', 'Marker', '>');
                delete(h_arrows);
                
                % 只显示3个方向箭头
                arrow_indices = round(linspace(1, length(all_time_shortest_path)-1, 3));
                for idx = arrow_indices
                    i = arrow_indices(idx == arrow_indices);
                    if i <= length(all_time_shortest_path)-1
                        start_pos = locations(all_time_shortest_path(i), :);
                        end_pos = locations(all_time_shortest_path(i+1), :);
                        mid_pos = (start_pos + end_pos) / 2;
                        direction = (end_pos - start_pos) / norm(end_pos - start_pos) * 0.0002;
                        
                        quiver(mid_pos(1), mid_pos(2), direction(1), direction(2), ...
                            'Color', 'red', 'LineWidth', 1.5, 'MaxHeadSize', 0.8, ...
                            'HandleVisibility', 'off');  % 不显示在图例中
                    end
                end
            end
        end
        
        % 后更新当前路径（虚线，在上层，但不会完全遮挡）
        set(current_path_plot, 'XData', locations(shortest_path, 1), 'YData', locations(shortest_path, 2));
        
        % 更新信息显示
        info_str = sprintf(['迭代: %d/%d\n' ...
            '当前最优: %.6f\n' ...
            '全局最优: %.6f\n' ...
            '改进率: %.2f%%'], ...
            iter, n_iterations, shortest_length, all_time_shortest_length, ...
            (1 - all_time_shortest_length/best_length_history(1)) * 100);
        set(info_text, 'String', info_str);
        
        % 更新收敛曲线
        subplot(2, 2, 2);
        set(convergence_plot, 'XData', 1:iter, 'YData', best_length_history);
        xlim([1, n_iterations]);
        if iter > 1
            ylim([min(best_length_history) * 0.95, max(best_length_history) * 1.05]);
        end
        
        % 每10次迭代更新一次信息素热力图以提高速度
        if mod(iter, 10) == 1 || iter == n_iterations
            subplot(2, 2, 4);
            set(pheromone_heatmap, 'CData', pheromone);
        end
        
        % 刷新显示
        drawnow;
    end
    
    % 大幅加快迭代速度
    if mod(iter, 10) == 1  % 只在每10次迭代时暂停
        pause(0.02);  % 大幅减少暂停时间
    end
end

% 输出最终结果
fprintf('\n=== 蚁群算法优化结果 ===\n');
fprintf('最短路径: ');
for i = 1:length(all_time_shortest_path)
    if i > 1
        fprintf(' -> ');
    end
    fprintf('%s', cities{all_time_shortest_path(i)});
end
fprintf('\n最短距离: %.6f\n', all_time_shortest_length);
fprintf('总迭代次数: %d\n', n_iterations);
fprintf('最终改进率: %.2f%%\n', (1 - all_time_shortest_length/best_length_history(1)) * 100);

% 保持图形显示
subplot(2, 2, [1, 3]);
title('蚁群算法求解TSP问题 - 最终结果', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red');
