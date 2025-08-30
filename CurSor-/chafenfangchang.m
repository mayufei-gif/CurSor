%{
这是一个耦合的二阶线性常系数差分方程组
具体特征：
1. 阶数：二阶（涉及k-1, k和k+1时刻）
2. 线性性质：线性（变量关系都是一次的）
3. 系数类型：常系数（a, b, c, d都是常数）
4. 耦合关系：耦合系统（x和y相互依赖）
5. 模型类型：考虑历史影响的动态系统
%}
% 二阶差分方程迭代模型
% 模拟x和y两个变量的动态变化关系，考虑历史值的影响
% 初始参数设置
x0 = 100;          % x的初始参考值
y0 = 10;           % y的初始参考值
x = [110, 105];    % x的初始实际值序列（需要两个初始值）
y = [12, 11];      % y的初始实际值序列（需要两个初始值）
a = 0.1;           % y对x变化的敏感系数
b = 9;             % x对y变化的敏感系数
c = 0.2;           % x历史值的影响系数
d = 0.15;          % y历史值的影响系数
k_max = 30;        % 最大迭代次数

% 二阶差分方程迭代计算
for k = 2:k_max
    % 根据当前x值和历史x值计算y值：y_k = y0 - a*(x_k - x0) - d*(y_{k-1} - y0)
    y_k = y0 - a*(x(k) - x0) - d*(y(k-1) - y0);
    y = [y, y_k];   % 将新计算的y值添加到序列
    
    % 如果不是最后一次迭代，计算下一个x值
    if k < k_max
        x_next = b*(y_k - y0) + c*(x(k) - x0);
        x = [x, x_next];
    end
end

% 确保绘图时维度一致
k_plot = 1:length(x);  % 使用与x相同的长度
y_plot = y(1:length(x));  % 确保y与x长度相同

% 绘制结果
figure;
subplot(1,2,1);
plot(k_plot, x, 'b-o', 'linewidth', 1.5);
xlabel('k');
ylabel('x');
title('x与k的关系');
legend('x序列', 'Location', 'best');  % 添加图例
grid on;

subplot(1,2,2);
plot(k_plot, y_plot, 'r-o', 'linewidth', 1.5);
xlabel('k');
ylabel('y');
title('y与k的关系');
legend('y序列', 'Location', 'best');  % 添加图例
grid on;
