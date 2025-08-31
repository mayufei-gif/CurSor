% 文件: DBLQUAD.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function result=dblquad(intfcn,inmin,inmax,outmin,outmax,tol,method)  % 详解: 执行语句


if nargin < 5  % 详解: 条件判断：if (nargin < 5)
  error('Requires at least five inputs');  % 详解: 调用函数：error('Requires at least five inputs')
elseif ~isstr(intfcn) & ~isa(intfcn,'inline')  % 详解: 条件判断：elseif (~isstr(intfcn) & ~isa(intfcn,'inline'))
  error(sprintf(['First argument must be a string, or inline object,'...  % 详解: 执行语句
        ' representing\n the integrand function.']));  % 详解: 执行语句
end  % 详解: 执行语句

if nargin < 7, method = 'quad'; end  % 详解: 条件判断：if (nargin < 7, method = 'quad'; end)
if nargin < 6, tol = []; end  % 详解: 条件判断：if (nargin < 6, tol = []; end)


trace = [];  % 详解: 赋值：计算表达式并保存到 trace
result= feval(method,'innerlp', outmin, outmax, tol, trace, intfcn, ...  % 详解: 赋值：将 feval(...) 的结果保存到 result
           inmin, inmax, tol, method);  % 详解: 执行语句




