% 文件: unaryEncoding.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function U = unaryEncoding(data, K)  % 详解: 执行语句

if nargin < 2, K = max(data); end  % 详解: 条件判断：if (nargin < 2, K = max(data); end)
N = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 N
U = zeros(K,N);  % 详解: 赋值：将 zeros(...) 的结果保存到 U
ndx = subv2ind([K N], [data(:)'; 1:N]');  % 详解: 赋值：将 subv2ind(...) 的结果保存到 ndx
U(ndx) = 1;  % 详解: 执行语句
U = reshape(U, [K N]);  % 详解: 赋值：将 reshape(...) 的结果保存到 U




