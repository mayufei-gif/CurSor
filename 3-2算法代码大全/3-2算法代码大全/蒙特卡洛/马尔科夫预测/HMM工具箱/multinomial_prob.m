% 文件: multinomial_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function B = eval_pdf_cond_multinomial(data, obsmat)  % 详解: 执行语句

[Q O] = size(obsmat);  % 详解: 获取向量/矩阵尺寸
T = prod(size(data));  % 详解: 赋值：将 prod(...) 的结果保存到 T
B = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B

for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
  B(:,t) = obsmat(:, data(t));  % 详解: 调用函数：B(:,t) = obsmat(:, data(t))
end  % 详解: 执行语句




