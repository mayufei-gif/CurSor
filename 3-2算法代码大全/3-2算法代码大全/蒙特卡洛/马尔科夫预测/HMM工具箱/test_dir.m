% 文件: test_dir.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% # of sample points
n_samples = 1000;  % 详解: 赋值：计算表达式并保存到 n_samples

p = ones(3,1)/3;  % 详解: 赋值：将 ones(...) 的结果保存到 p

alpha = 0.5*p;  % 详解: 赋值：计算表达式并保存到 alpha


points = zeros(3,n_samples);  % 详解: 赋值：将 zeros(...) 的结果保存到 points
for i = 1:n_samples  % 详解: for 循环：迭代变量 i 遍历 1:n_samples
    points(:,i) = dirichletrnd(alpha);  % 详解: 调用函数：points(:,i) = dirichletrnd(alpha)
end  % 详解: 执行语句

scatter3(points(1,:)', points(2,:)', points(3,:)', 'r', '.', 'filled');  % 调用函数：scatter3  % 详解: 调用函数：scatter3(points(1,:)', points(2,:)', points(3,:)', 'r', '.', 'filled')  % 详解: 调用函数：scatter3(points(1,:)', points(2,:)', points(3,:)', 'r', '.', 'filled'); % 调用函数：scatter3 % 详解: 调用函数：scatter3(points(1,:)', points(2,:)', points(3,:)', 'r', '.', 'filled')



