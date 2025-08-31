% 文件: f6.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% function fval = f6(x)
% % Unimodal function f_1
% 
% Bound = [-500 500];
% 
% if nargin == 0
%     fval = Bound;
% else  % 中文: p（j）= h;
%     fval = sum(-x.*sin(abs(x).^.5));
% end  % 中文: min = q（i）;
function fval=f6(x)  % 详解: 执行语句
Bound=[-100 100];  % 详解: 赋值：计算表达式并保存到 Bound

if nargin==0  % 详解: 条件判断：if (nargin==0)
    fval = Bound;  % 详解: 赋值：计算表达式并保存到 fval
else  % 详解: 条件判断：else 分支
    x=abs(floor(x+0.5));  % 详解: 赋值：将 abs(...) 的结果保存到 x
    fval= sum(x);  % 详解: 赋值：将 sum(...) 的结果保存到 fval
end  % 详解: 执行语句



