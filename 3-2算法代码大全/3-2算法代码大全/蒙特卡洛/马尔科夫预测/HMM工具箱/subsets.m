% 文件: subsets.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [T, bitv] = subsets(S, U, L, sorted, N)  % 详解: 函数定义：subsets(S, U, L, sorted, N), 返回：T, bitv

n = length(S);  % 详解: 赋值：将 length(...) 的结果保存到 n

if nargin < 2, U = n; end  % 详解: 条件判断：if (nargin < 2, U = n; end)
if nargin < 3, L = 0; end  % 详解: 条件判断：if (nargin < 3, L = 0; end)
if nargin < 4, sorted = 0; end  % 详解: 条件判断：if (nargin < 4, sorted = 0; end)
if nargin < 5, N = max(S); end  % 详解: 条件判断：if (nargin < 5, N = max(S); end)

bits = ind2subv(2*ones(1,n), 1:2^n)-1;  % 详解: 赋值：将 ind2subv(...) 的结果保存到 bits
sm = sum(bits,2);  % 详解: 赋值：将 sum(...) 的结果保存到 sm
masks = bits((sm <= U) & (sm >= L), :);  % 详解: 赋值：将 bits(...) 的结果保存到 masks
m = size(masks, 1);  % 详解: 赋值：将 size(...) 的结果保存到 m
T = cell(1, m);  % 详解: 赋值：将 cell(...) 的结果保存到 T
for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
  s = S(find(masks(i,:)));  % 详解: 赋值：将 S(...) 的结果保存到 s
  T{i} = s;  % 详解: 执行语句
end  % 详解: 执行语句

if sorted  % 详解: 条件判断：if (sorted)
  T = sortcell(T);  % 详解: 赋值：将 sortcell(...) 的结果保存到 T
end  % 详解: 执行语句

bitv = zeros(m, N);  % 详解: 赋值：将 zeros(...) 的结果保存到 bitv
bitv(:, S) = masks;  % 详解: 执行语句




