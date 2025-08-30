clc,clear;

% 原始数据
x = 1:5;
y = 1:3;
temps = [82 81 80 82 84; 79 63 61 65 81; 84 84 82 85 86];

% 创建插值网格
[xq, yq] = meshgrid(1:0.1:5, 1:0.1:3);
[X, Y] = meshgrid(x, y);

% 执行二维插值
points = [X(:), Y(:)];
values = temps(:);
interp_temps = griddata(points(:,1), points(:,2), values, xq, yq, 'cubic');

% 绘制二维可视化
figure;
surf(xq, yq, interp_temps);
xlabel('X坐标'); ylabel('Y坐标'); zlabel('温度');
title('插值后的温度分布');
colorbar;

% 显示原始数据点
hold on;
plot3(X(:), Y(:), temps(:), 'ro', 'MarkerSize', 8);
legend('插值曲面', '原始数据点');