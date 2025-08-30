% 原始数据
x = 1:5;
y = 1:3;
temps = [82 81 80 82 84; 79 63 61 65 81; 84 84 82 85 86];

% 创建插值网格（更密集的采样点）
[xq, yq] = meshgrid(1:0.1:5, 1:0.1:3);

% 方法一：使用griddata（推荐，最稳定）
% 先将原始数据转换为点坐标格式
[X, Y] = meshgrid(x, y);
points = [X(:), Y(:)];
values = temps(:);

% 使用griddata进行插值
interp_temps = griddata(points(:,1), points(:,2), values, xq, yq, 'cubic');

% 绘制插值后的温度分布曲面图
figure;
surf(xq, yq, interp_temps);

% 设置图形属性
xlabel('X坐标');
ylabel('Y坐标');
zlabel('温度');
title('插值后的温度分布曲面图');
colorbar;

% 设置视角
view(45, 30);

% 添加原始数据点作为对比
hold on;
plot3(X(:), Y(:), temps(:), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
legend('插值曲面', '原始数据点', 'Location', 'best');
hold off;

% 方法二：如果一定要使用interp2，正确的写法是：
% 注意：temps的行对应y，列对应x
% interp_temps = interp2(x, y, temps, xq, yq, 'cubic');

% 可选：添加等高线投影
hold on;
contour3(xq, yq, interp_temps, 20, 'k', 'LineWidth', 0.5);
hold off;

% 或者使用mesh绘制网格图
figure;
mesh(xq, yq, interp_temps);
xlabel('X坐标');
ylabel('Y坐标');
zlabel('温度');
title('插值后的温度分布网格图');
colorbar;