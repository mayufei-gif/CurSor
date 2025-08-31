% 文件: f10.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function f10=f10(x)  % 详解: 执行语句
Bound=[-32 32];  % 详解: 赋值：计算表达式并保存到 Bound

if nargin==0  % 详解: 条件判断：if (nargin==0)
    f10 = Bound;  % 详解: 赋值：计算表达式并保存到 f10
else  % 详解: 条件判断：else 分支
    [Dim, PopSize] = size(x);  % 详解: 获取向量/矩阵尺寸
    indices = repmat(Dim, PopSize, 1)';  % 赋值：设置变量 indices  % 详解: 赋值：将 repmat(...) 的结果保存到 indices  % 详解: 赋值：将 repmat(...) 的结果保存到 indices
    f10 = -20*exp(-0.2*(sum(x.^2)./indices).^.5)-exp(sum(cos(2*pi.*x))./indices)+20+exp(1);  % 详解: 赋值：计算表达式并保存到 f10
end  % 详解: 执行语句



