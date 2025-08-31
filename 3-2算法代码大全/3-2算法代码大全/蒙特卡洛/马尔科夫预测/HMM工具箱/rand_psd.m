% 文件: rand_psd.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function M = rand_psd(d, d2, k)  % 详解: 执行语句

if nargin<2, d2 = d; end  % 详解: 条件判断：if (nargin<2, d2 = d; end)
if nargin<3, k = 1; end  % 详解: 条件判断：if (nargin<3, k = 1; end)
if d2 ~= d, error('must be square'); end  % 详解: 条件判断：if (d2 ~= d, error('must be square'); end)

M = zeros(d,d,k);  % 详解: 赋值：将 zeros(...) 的结果保存到 M
for i=1:k  % 详解: for 循环：迭代变量 i 遍历 1:k
  A = rand(d);  % 详解: 赋值：将 rand(...) 的结果保存到 A
  M(:,:,i) = A*A';  % 调用函数：M  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句




