% 文件: f9.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% function fval = f9(x)
% 
% Bound = [-600 600];
% 
% if nargin==0
%     fval = Bound;
% else  % 中文: p（j）= h;
%     x = x';
%     fval = sum(x.^2)/1000-prod(cos(x./sqrt(1:30)))+1;
% end  % 中文: min = q（i）;
%Rastrigin Function  % 中文: rastrigin函数||| ％垂直过渡||| ％水平过渡||| ％排放||| ％normierung |||指标生产状态|||指标内部状态||| alpha
function fval=f9(x)  % 详解: 执行语句

Bound=[-5.12 5.12];  % 详解: 赋值：计算表达式并保存到 Bound

if nargin==0  % 详解: 条件判断：if (nargin==0)
    fval= Bound;  % 详解: 赋值：计算表达式并保存到 fval
else  % 详解: 条件判断：else 分支
    [Dim, PopSize] = size(x);  % 详解: 获取向量/矩阵尺寸
    fval = sum(x.^2 - 10*cos(2*pi.*x)) + Dim*10;  % 详解: 赋值：将 sum(...) 的结果保存到 fval
end  % 详解: 执行语句



