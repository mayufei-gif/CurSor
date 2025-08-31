% 文件: histCmpChi2.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function d = histCmpChi2(h1, h2)  % 详解: 绘图：直方图

[N B] = size(h1);  % 详解: 获取向量/矩阵尺寸
d = zeros(N,N);  % 详解: 赋值：将 zeros(...) 的结果保存到 d
for i=1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
  h1i = repmat(h1(i,:), N, 1);  % 详解: 赋值：将 repmat(...) 的结果保存到 h1i
  numer = (h1i - h2).^2;  % 详解: 赋值：计算表达式并保存到 numer
  denom = h1i + h2 + eps;  % 详解: 赋值：计算表达式并保存到 denom
  d(i,:) = sum(numer ./ denom, 2);  % 详解: 调用函数：d(i,:) = sum(numer ./ denom, 2)
end  % 详解: 执行语句
  




