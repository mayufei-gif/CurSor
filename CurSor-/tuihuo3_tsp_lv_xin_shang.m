
%% 模拟退火解决TSP问题

tic
rng('shuffle')
clear;clc
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
n = size(coord,1);  % 城市的数目

% 创建图形窗口
figure('Position', [100, 100, 1200, 500])%设置窗口位置和大小

% 子图1：城市分布和路径
subplot(1,2,1)  %=subplot(m,n,p)创建一个m*n的子图，并在第p个位置创建一个新图形
plot(coord(:,1),coord(:,2),'o','MarkerSize',8,'MarkerFaceColor','r');% coord(:,1)表示所有城市的x坐标，coord(:,2)表示所有城市的y坐标,指定标记的形状为圆形（circle），大小为8，颜色为红色
plot(coord(1,1),coord(1,2),'d','MarkerSize',10,'MarkerFaceColor','g');% 起点为绿色菱形
hold on
title('TSP路径优化过程')
xlabel('X坐标')
ylabel('Y坐标')
grid on  %显示网格线

% 计算距离矩阵
d = zeros(n); % 初始化距离矩阵
for i = 2:n  % 从第2个城市开始计算距离
    for j = 1:i
        coord_i = coord(i,:);   x_i = coord_i(1);     y_i = coord_i(2);
        coord_j = coord(j,:);   x_j = coord_j(1);     y_j = coord_j(2);
        d(i,j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2);% 计算欧氏距离
    end
end
d = d+d'; %通过d = d+d'将下三角部分复制到上三角部分，得到完整的对称距离矩阵

%% 参数初始化
T0 = 1000;   % 初始温度
T = T0;
maxgen = 90;  % 减少迭代次数以便测试
Lk = 100;     % 减少内循环次数
alpfa = 0.95;

%% 随机生成一个初始解
path0 = randperm(n);% 随机生成一个初始解
result0 = calculate_tsp_d(path0,d);% 计算初始路径的总距离

% 绘制初始路径
path0_closed = [path0, path0(1)];% 将路径闭合，形成一个闭环
path_plot = plot(coord(path0_closed,1), coord(path0_closed,2), 'b-', 'LineWidth', 1.5);% 绘制初始路径
title(sprintf('初始路径: 距离 = %.2f', result0));% 显示初始路径的总距离

%% 定义保存中间过程的量
min_result = result0;% 初始化最小结果
RESULT = zeros(maxgen,1);   % 初始化结果数组
best_path = path0;   % 初始化最佳路径

% 子图2：收敛曲线
subplot(1,2,2)%=subplot(m,n,p)创建一个m*n的子图，并在第p个位置创建一个新图形
% 初始化收敛曲线
convergence_plot = plot(1, result0, 'b-', 'LineWidth', 1.5);    % 绘制初始路径
title('收敛曲线')   % 显示标题
xlabel('迭代次数')   % 显示x轴标签
ylabel('最短路径距离')  % 显示y轴标签
grid on % 显示网格线
hold on % 保持图形窗口打开，以便绘制新的曲线
current_plot = plot(1, result0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % 绘制当前路径

%% 模拟退火过程
for iter = 1 : maxgen
    for i = 1 : Lk
        path1 = gen_new_path(path0);
        result1 = calculate_tsp_d(path1,d);
        
        if result1 < result0
            path0 = path1;
            result0 = result1;
        else
            p = exp(-(result1 - result0)/T);
            if rand(1) < p
                path0 = path1;
                result0 = result1;
            end
        end
        
        % 更新最佳解
        if result0 < min_result
            min_result = result0;
            best_path = path0;
            
            % 更新最佳路径显示
            subplot(1,2,1)
            % 删除旧的路径图
            if exist('path_plot', 'var') && isvalid(path_plot)
                delete(path_plot);
            end
            best_path_closed = [best_path, best_path(1)];
            path_plot = plot(coord(best_path_closed,1), coord(best_path_closed,2), 'g-', 'LineWidth', 2);
            title(sprintf('迭代: %d, 温度: %.2f, 最佳距离: %.2f', iter, T, min_result))
        end
        
        % 每10次内循环更新一次显示
        if mod(i, 10) == 0
            subplot(1,2,1)
            current_path_closed = [path0, path0(1)];
            temp_plot = plot(coord(current_path_closed,1), coord(current_path_closed,2), 'r--', 'LineWidth', 1);
            title(sprintf('迭代: %d-%d, 当前距离: %.2f', iter, i, result0))
            pause(0.01)
            if isvalid(temp_plot)
                delete(temp_plot);
            end
        end
    end
    
    RESULT(iter) = min_result;
    
    % 更新收敛曲线 - 修复这里
    subplot(1,2,2)  % 选择第二个子图
    % 检查图形对象是否仍然有效
    if ishandle(convergence_plot)
        set(convergence_plot, 'XData', 1:iter, 'YData', RESULT(1:iter));
    else
        % 重新创建收敛曲线
        convergence_plot = plot(1:iter, RESULT(1:iter), 'b-', 'LineWidth', 1.5);
    end
    
    if ishandle(current_plot)
        set(current_plot, 'XData', iter, 'YData', min_result);
    else
        current_plot = plot(iter, min_result, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    end
    
    title(sprintf('收敛曲线 - 当前最佳: %.2f', min_result))
    drawnow; % 强制刷新图形
    
    T = alpfa * T;
    
    % 显示进度
    if mod(iter, 5) == 0
        fprintf('迭代: %d/%d, 温度: %.2f, 最佳距离: %.2f\n', iter, maxgen, T, min_result);
        pause(0.05)
    end
end

%% 最终路径动画展示
disp('开始最终路径动画展示...');
final_path = [best_path, best_path(1)];

subplot(1,2,1)
cla
plot(coord(:,1),coord(:,2),'o','MarkerSize',8,'MarkerFaceColor','r');
hold on
title(sprintf('最终最优路径: 距离 = %.2f', min_result))
xlabel('X坐标')
ylabel('Y坐标')
grid on

% 逐段绘制最终路径
for i = 1:length(final_path)-1
    j = i+1;
    coord_i = coord(final_path(i),:);
    coord_j = coord(final_path(j),:);
    
    % 绘制线段
    plot([coord_i(1), coord_j(1)], [coord_i(2), coord_j(2)], '-b', 'LineWidth', 2);
    
    % 标记城市编号
    text(coord_i(1), coord_i(2), sprintf('%d', final_path(i)), 'FontSize', 8);
    if i == length(final_path)-1
        text(coord_j(1), coord_j(2), sprintf('%d', final_path(j)), 'FontSize', 8);
    end
    
    pause(0.2)
    drawnow;
end

disp('最佳的方案是：'); disp(mat2str(best_path))
disp('此时最优值是：'); disp(min_result)

%% 画出完整的收敛曲线
figure
plot(1:maxgen,RESULT,'b-','LineWidth',1.5);
xlabel('迭代次数');
ylabel('最短路径距离');
title('TSP模拟退火算法收敛曲线');
grid on

toc

% 计算TSP路径距离的函数
function result = calculate_tsp_d(path, d)
n = length(path);
result = 0;
for i = 1:n-1
    result = result + d(path(i), path(i+1));
end
result = result + d(path(n), path(1));
end

% 生成新路径的函数
function new_path = gen_new_path(old_path)
n = length(old_path);
if rand < 0.5
    % 交换两个随机城市的位置
    indices = randperm(n,2);
    new_path = old_path;
    new_path(indices(1)) = old_path(indices(2));
    new_path(indices(2)) = old_path(indices(1));
else
    % 逆转一段随机路径
    indices = sort(randperm(n, 2));
    new_path = old_path;
    new_path(indices(1):indices(2)) = old_path(indices(2):-1:indices(1));
end
end
