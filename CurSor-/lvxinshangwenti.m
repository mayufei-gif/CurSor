tic
clear; clc; close all;

% 38个城市数据
coord = [11003.611100,42102.500000;11108.611100,42373.888900;
    11133.333300,42885.833300;11155.833300,42712.500000;
    11183.333300,42933.333300;11297.500000,42853.333300;
    11310.277800,42929.444400;11416.666700,42983.333300;
    11423.888900,43000.277800;11438.333300,42057.222200;
    11461.111100,43252.777800;11485.555600,43187.222200;
    11503.055600,42855.277800;11511.388900,42106.388900;
    11522.222200,42841.944400;11569.444400,43136.666700;
    11583.333300,43150.000000;11595.000000,43148.055600;
    11600.000000,43150.000000;11690.555600,42686.666700;
    11715.833300,41836.111100;11751.111100,42814.444400;
    11770.277800,42651.944400;11785.277800,42884.444400;
    11822.777800,42673.611100;11846.944400,42660.555600;
    11963.055600,43290.555600;11973.055600,43026.111100;
    12058.333300,42195.555600;12149.444400,42477.500000;
    12286.944400,43355.555600;12300.000000,42433.333300;
    12355.833300,43156.388900;12363.333300,43189.166700;
    12372.777800,42711.388900;12386.666700,43334.722200;
    12421.666700,42895.555600;12645.000000,42973.333300];
n = size(coord, 1);  % 城市的数目

% 计算距离矩阵
d = zeros(n);
for i = 1:n
    for j = 1:n
        if i ~= j
            d(i, j) = sqrt((coord(i,1)-coord(j,1))^2 + (coord(i,2)-coord(j,2))^2);
        end
    end
end

% 蚁群算法参数
n_ants = 50;        % 蚂蚁数量
n_iterations = 200; % 迭代次数
alpha = 1;          % 信息素重要程度
beta = 5;           % 启发式因子重要程度
rho = 0.1;          % 信息素蒸发率
Q = 100;            % 信息素增加强度系数

% 初始化信息素矩阵
tau = ones(n, n);

% 创建图形窗口
figure('Position', [100, 100, 1200, 500])

% 子图1：城市分布和路径
subplot(1,2,1)
plot(coord(:,1), coord(:,2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold on
title('蚁群算法TSP路径优化过程')
xlabel('X坐标')
ylabel('Y坐标')
grid on

% 初始化路径绘制对象
best_path_plot = plot(0, 0, 'g-', 'LineWidth', 2);
current_path_plot = plot(0, 0, 'b--', 'LineWidth', 1);
iteration_text = text(min(coord(:,1)), max(coord(:,2))+200, '', 'FontSize', 12);
distance_text = text(min(coord(:,1)), max(coord(:,2))+100, '', 'FontSize', 12);

% 子图2：收敛曲线
subplot(1,2,2)
convergence_plot = plot(0, 0, 'b-', 'LineWidth', 1.5);
hold on
best_convergence_plot = plot(0, 0, 'r-', 'LineWidth', 2);
title('收敛曲线')
xlabel('迭代次数')
ylabel('路径距离')
legend('当前迭代最佳', '全局最佳')
grid on

% 初始化记录变量
best_path = [];
best_distance = inf;
all_best_distances = zeros(n_iterations, 1);
all_current_best = zeros(n_iterations, 1);

%% 蚁群算法主循环
for iter = 1:n_iterations
    % 每只蚂蚁构建路径
    ant_paths = zeros(n_ants, n+1); % 存储每只蚂蚁的路径
    ant_distances = zeros(n_ants, 1); % 存储每只蚂蚁的路径长度
    
    for ant = 1:n_ants
        % 初始化路径和已访问城市
        path = zeros(1, n+1);
        visited = false(1, n);
        
        % 随机选择起始城市
        current = randi(n);
        path(1) = current;
        visited(current) = true;
        
        % 构建路径
        for step = 2:n
            % 计算选择概率
            prob = zeros(1, n);
            for next_city = 1:n
                if ~visited(next_city)
                    prob(next_city) = (tau(current, next_city)^alpha) * ((1/d(current, next_city))^beta);
                end
            end
            
            % 选择下一个城市
            prob = prob / sum(prob);
            next_city = randsample(n, 1, true, prob);
            
            path(step) = next_city;
            visited(next_city) = true;
            current = next_city;
        end
        
        % 回到起点
        path(n+1) = path(1);
        
        % 计算路径长度
        distance = 0;
        for i = 1:n
            distance = distance + d(path(i), path(i+1));
        end
        
        ant_paths(ant, :) = path;
        ant_distances(ant) = distance;
    end
    
    % 找出当前迭代的最佳路径
    [current_best_distance, idx] = min(ant_distances);
    current_best_path = ant_paths(idx, :);
    
    % 更新全局最佳路径
    if current_best_distance < best_distance
        best_distance = current_best_distance;
        best_path = current_best_path;
    end
    
    % 记录结果
    all_best_distances(iter) = best_distance;
    all_current_best(iter) = current_best_distance;
    
    % 更新信息素
    % 信息素蒸发
    tau = (1 - rho) * tau;
    
    % 信息素增加 - 只对当前迭代最佳路径增加
    for i = 1:n
        from = current_best_path(i);
        to = current_best_path(i+1);
        tau(from, to) = tau(from, to) + Q / current_best_distance;
        tau(to, from) = tau(to, from) + Q / current_best_distance; % 对称矩阵
    end
    
    % 动态可视化
    % 更新路径图
    subplot(1,2,1)
    set(current_path_plot, 'XData', coord(current_best_path, 1), 'YData', coord(current_best_path, 2));
    set(best_path_plot, 'XData', coord(best_path, 1), 'YData', coord(best_path, 2));
    set(iteration_text, 'String', sprintf('迭代次数: %d/%d', iter, n_iterations));
    set(distance_text, 'String', sprintf('当前最佳: %.2f | 全局最佳: %.2f', current_best_distance, best_distance));
    
    % 更新收敛曲线
    subplot(1,2,2)
    set(convergence_plot, 'XData', 1:iter, 'YData', all_current_best(1:iter));
    set(best_convergence_plot, 'XData', 1:iter, 'YData', all_best_distances(1:iter));
    
    % 刷新图形
    drawnow;
    
    % 显示进度
    if mod(iter, 10) == 0
        fprintf('迭代: %d/%d, 当前最佳: %.2f, 全局最佳: %.2f\n', iter, n_iterations, current_best_distance, best_distance);
    end
end

%% 最终结果显示
fprintf('\n最优路径长度: %.2f\n', best_distance);
fprintf('最优路径: ');
for i = 1:length(best_path)
    if i > 1
        fprintf(' -> ');
    end
    fprintf('%d', best_path(i));
end
fprintf('\n');

% 绘制最终结果
figure('Position', [100, 100, 800, 600])
plot(coord(:,1), coord(:,2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold on
plot(coord(best_path, 1), coord(best_path, 2), 'g-', 'LineWidth', 2);
title(sprintf('蚁群算法最优路径: 距离 = %.2f', best_distance))
xlabel('X坐标')
ylabel('Y坐标')
grid on

% 标记城市编号
for i = 1:n
    text(coord(i,1)+50, coord(i,2)+50, num2str(i), 'FontSize', 10);
end

% 绘制收敛曲线
figure
plot(1:n_iterations, all_current_best, 'b-', 'LineWidth', 1.5);
hold on
plot(1:n_iterations, all_best_distances, 'r-', 'LineWidth', 2);
xlabel('迭代次数')
ylabel('路径距离')
title('蚁群算法收敛曲线')
legend('当前迭代最佳', '全局最佳')
grid on

toc