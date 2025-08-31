% 文件: TRAPZ.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function z = trapz(x,y,dim)  % 详解: 执行语句



perm = []; nshifts = 0;  % 详解: 赋值：计算表达式并保存到 perm
if nargin == 3,  % 详解: 条件判断：if (nargin == 3,)
  perm = [dim:max(ndims(y),dim) 1:dim-1];  % 详解: 赋值：计算表达式并保存到 perm
  y = permute(y,perm);  % 详解: 赋值：将 permute(...) 的结果保存到 y
  m = size(y,1);  % 详解: 赋值：将 size(...) 的结果保存到 m
elseif nargin==2 & length(y)==1  % 详解: 条件判断：elseif (nargin==2 & length(y)==1)
  dim = y; y = x;  % 详解: 赋值：计算表达式并保存到 dim
  perm = [dim:max(ndims(y),dim) 1:dim-1];  % 详解: 赋值：计算表达式并保存到 perm
  y = permute(y,perm);  % 详解: 赋值：将 permute(...) 的结果保存到 y
  m = size(y,1);  % 详解: 赋值：将 size(...) 的结果保存到 m
  x = 1:m;  % 详解: 赋值：计算表达式并保存到 x
else  % 详解: 条件判断：else 分支
  if nargin < 2, y = x; end  % 详解: 条件判断：if (nargin < 2, y = x; end)
  [y,nshifts] = shiftdim(y);  % 详解: 执行语句
  m = size(y,1);  % 详解: 赋值：将 size(...) 的结果保存到 m
  if nargin < 2, x = 1:m; end  % 详解: 条件判断：if (nargin < 2, x = 1:m; end)
end  % 详解: 执行语句
x = x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x
if length(x) ~= m  % 详解: 条件判断：if (length(x) ~= m)
    error('length(x) must equal length of first non-singleton dim of y.');  % 详解: 调用函数：error('length(x) must equal length of first non-singleton dim of y.')
end  % 详解: 执行语句

z = diff(x,1,1)' * (y(1:m-1,:) + y(2:m,:))/2;  % 赋值：设置变量 z  % 详解: 赋值：将 diff(...) 的结果保存到 z  % 详解: 赋值：将 diff(...) 的结果保存到 z

siz = size(y); siz(1) = 1;  % 详解: 赋值：将 size(...) 的结果保存到 siz
z = reshape(z,[ones(1,nshifts),siz]);  % 详解: 赋值：将 reshape(...) 的结果保存到 z
if ~isempty(perm), z = ipermute(z,perm); end  % 详解: 条件判断：if (~isempty(perm), z = ipermute(z,perm); end)




