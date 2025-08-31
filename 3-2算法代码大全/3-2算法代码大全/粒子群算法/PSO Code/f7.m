% 文件: f7.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function f7=f7(x)  % 详解: 执行语句

Bound=[-1.28 1.28];  % 详解: 赋值：计算表达式并保存到 Bound

if nargin==0  % 详解: 条件判断：if (nargin==0)
    f7 = Bound;  % 详解: 赋值：计算表达式并保存到 f7
else  % 详解: 条件判断：else 分支
[row col]=size(x);  % 详解: 获取向量/矩阵尺寸
onematrix=ones(row,col);  % 详解: 赋值：将 ones(...) 的结果保存到 onematrix
i=cumsum(onematrix);  % 详解: 赋值：将 cumsum(...) 的结果保存到 i
f7=sum(i.*x.^4)+rand(1,col);  % 详解: 赋值：将 sum(...) 的结果保存到 f7
end  % 详解: 执行语句



