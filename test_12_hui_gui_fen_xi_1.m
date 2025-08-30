% 1.画出散点图，判断为线性关系
x = [0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18];  % 与y对应的x值
y = [42,41.5,45.0,45.5,45.0,47.5,49.0,55.0,50.0];

% 创建新图形并设置白色背景
figure('Color','white');
plot(x,y,'+','MarkerSize',10,'LineWidth',2);
grid on;
xlabel('x');
ylabel('y');
title('散点图 - 线性关系判断');

% 定义因变量y的9个观测值，同样为列向量格式
x1 = x';
y = y';
% 构建设计矩阵x，第一列全为1（截距项）
% 第二列为x1变量，用于一元线性回归模型：y = b0 + b1*x1 + ε。
X = [ones(9,1), x1];

% 使用regress函数进行线性回归分析
[b,bint,r,rint,stats] = regress(y,X);

% 输出回归系数、置信区间和统计量
b, bint, stats

% 创建新的白色背景图形用于残差图
figure('Color','white');
rcoplot(r,rint);
title('残差置信区间图');

% 创建新的白色背景图形用于残差个案次序图
figure('Color','white');
plot(1:length(r), r, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
plot([1 length(r)], [0 0], 'k--'); % 添加零线
xlabel('观测序号');
ylabel('残差值');
title('残差个案次序图');
grid on;
