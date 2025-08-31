% 文件: LP.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,lambda,how]=lp(f,A,B,vlb,vub,x,neqcstr,verbosity)  % 详解: 函数定义：lp(f,A,B,vlb,vub,x,neqcstr,verbosity), 返回：x,lambda,how

         
if nargin<8, verbosity = 0;  % 详解: 条件判断：if (nargin<8, verbosity = 0;)
    if nargin<7, neqcstr = 0;  % 详解: 条件判断：if (nargin<7, neqcstr = 0;)
        if nargin<6, x = [];  % 详解: 条件判断：if (nargin<6, x = [];)
            if nargin<5, vub = [];  % 详解: 条件判断：if (nargin<5, vub = [];)
                if nargin<4, vlb = [];  % 详解: 条件判断：if (nargin<4, vlb = [];)
        end, end, end, end, end  % 详解: 执行语句
[ncstr,nvars]=size(A);  % 详解: 获取向量/矩阵尺寸
nvars = max([length(f),nvars]);  % 详解: 赋值：将 max(...) 的结果保存到 nvars
        
if isempty(verbosity), verbosity = 0; end  % 详解: 条件判断：if (isempty(verbosity), verbosity = 0; end)
if isempty(neqcstr), neqcstr = 0; end  % 详解: 条件判断：if (isempty(neqcstr), neqcstr = 0; end)
if isempty(x), x=zeros(nvars,1); end  % 详解: 条件判断：if (isempty(x), x=zeros(nvars,1); end)
        
if isempty(A), A=zeros(0,nvars); end  % 详解: 条件判断：if (isempty(A), A=zeros(0,nvars); end)
if isempty(B), B=zeros(0,1); end  % 详解: 条件判断：if (isempty(B), B=zeros(0,1); end)

f = f(:);  % 详解: 赋值：将 f(...) 的结果保存到 f
B = B(:);  % 详解: 赋值：将 B(...) 的结果保存到 B
x = x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x

caller = 'lp';  % 详解: 赋值：计算表达式并保存到 caller
[x,lambda,how]=qpsubold([],f,A,B,vlb,vub,x,neqcstr,verbosity,caller,ncstr,nvars);  % 详解: 执行语句





