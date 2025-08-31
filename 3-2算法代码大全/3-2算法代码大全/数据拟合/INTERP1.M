% 文件: INTERP1.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function yi = interp1(varargin)  % 详解: 执行语句


error(nargchk(2,4,nargin));  % 详解: 调用函数：error(nargchk(2,4,nargin))

bypass = 0;  % 详解: 赋值：计算表达式并保存到 bypass
uniform = 1;  % 详解: 赋值：计算表达式并保存到 uniform
if isstr(varargin{end}),  % 详解: 条件判断：if (isstr(varargin{end}),)
  narg = nargin-1;  % 详解: 赋值：计算表达式并保存到 narg
  method = [varargin{end} '    '];  % 详解: 赋值：计算表达式并保存到 method
  if method(1)=='*',  % 详解: 条件判断：if (method(1)=='*',)
    if method(2)=='l',  % 详解: 条件判断：if (method(2)=='l',)
      yi = linear(varargin{1:end-1});  % 详解: 赋值：将 linear(...) 的结果保存到 yi
      return  % 详解: 返回：从当前函数返回

    elseif method(2)=='c',  % 详解: 条件判断：elseif (method(2)=='c',)
      yi = cubic(varargin{1:end-1});  % 详解: 赋值：将 cubic(...) 的结果保存到 yi
      return  % 详解: 返回：从当前函数返回

    elseif method(2)=='n',  % 详解: 条件判断：elseif (method(2)=='n',)
      yi = nearest(varargin{1:end-1});  % 详解: 赋值：将 nearest(...) 的结果保存到 yi
      return  % 详解: 返回：从当前函数返回

    elseif method(2)=='s',  % 详解: 条件判断：elseif (method(2)=='s',)
      method = 'spline'; bypass = 1;  % 详解: 赋值：计算表达式并保存到 method

    else  % 详解: 条件判断：else 分支
      error([deblank(method),' is an invalid method.']);  % 详解: 调用函数：error([deblank(method),' is an invalid method.'])

    end  % 详解: 执行语句
  elseif method(1)=='s',  % 详解: 条件判断：elseif (method(1)=='s',)
    method = 'spline'; bypass = 1;  % 详解: 赋值：计算表达式并保存到 method
  end  % 详解: 执行语句
  
else  % 详解: 条件判断：else 分支
  narg = nargin;  % 详解: 赋值：计算表达式并保存到 narg
  method = 'linear';  % 详解: 赋值：计算表达式并保存到 method
end  % 详解: 执行语句

if narg==2,  % 详解: 条件判断：if (narg==2,)
  y = varargin{1};  % 详解: 赋值：计算表达式并保存到 y
  if min(size(y))==1, x = 1:length(y); else x = 1:size(y,1); end  % 详解: 条件判断：if (min(size(y))==1, x = 1:length(y); else x = 1:size(y,1); end)
  [msg,x,y,xi] = xychk(x,varargin{1:2});  % 详解: 执行语句

elseif narg==3,  % 详解: 条件判断：elseif (narg==3,)
  [msg,x,y,xi] = xychk(varargin{1:3});  % 详解: 执行语句

end  % 详解: 执行语句

if ~isempty(msg), error(msg); end  % 详解: 条件判断：if (~isempty(msg), error(msg); end)

if isempty(xi), yi = []; return, end  % 详解: 条件判断：if (isempty(xi), yi = []; return, end)
if min(size(xi))~=1, error('XI must be a vector.'); end  % 详解: 条件判断：if (min(size(xi))~=1, error('XI must be a vector.'); end)

x = x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x
if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
siz = size(xi); xi = xi(:);  % 详解: 赋值：将 size(...) 的结果保存到 siz

if length(x)>2 & ~bypass,  % 详解: 条件判断：if (length(x)>2 & ~bypass,)
  dx = diff(x);  % 详解: 赋值：将 diff(...) 的结果保存到 dx
  if max(abs(diff(dx))) > eps*max(x),  % 详解: 条件判断：if (max(abs(diff(dx))) > eps*max(x),)
    if any(dx < 0),  % 详解: 条件判断：if (any(dx < 0),)
      if size(x,1)==1,  % 详解: 条件判断：if (size(x,1)==1,)
        x = fliplr(x); y = fliplr(y);  % 详解: 赋值：将 fliplr(...) 的结果保存到 x
        dx = -fliplr(dx);  % 详解: 赋值：计算表达式并保存到 dx
      else  % 详解: 条件判断：else 分支
        x(:) = flipud(x); y(:) = flipud(y);  % 详解: 调用函数：x(:) = flipud(x); y(:) = flipud(y)
        dx(:) = -flipud(dx);  % 详解: 调用函数：dx(:) = -flipud(dx)
      end  % 详解: 执行语句
    end  % 详解: 执行语句

    if any(dx<=0),  % 详解: 条件判断：if (any(dx<=0),)
      error('X must be monotonic.');  % 详解: 调用函数：error('X must be monotonic.')
      return  % 详解: 返回：从当前函数返回
    end  % 详解: 执行语句

   if method(1)=='n',  % 详解: 条件判断：if (method(1)=='n',)

      [xxi,j] = sort(xi(:));  % 详解: 执行语句
      [dum,i] = sort([x;xxi]);  % 详解: 执行语句
      ui(i) = 1:length(i);  % 详解: 调用函数：ui(i) = 1:length(i)
      ui = (ui(length(x)+1:end) - (1:length(xxi)))';  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 ui  % 详解: 赋值：计算表达式并保存到 ui
      ui(j) = ui;  % 详解: 执行语句
    
      ui(ui<1) = 1;  % 详解: 执行语句
      ui(ui>length(x)-1) = length(x)-1;  % 详解: 获取向量/矩阵尺寸
      ui = ui + (xi(:)-x(ui))./(x(ui+1)-x(ui));  % 详解: 赋值：计算表达式并保存到 ui
    
      x = (1:length(x)).';  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 x  % 详解: 赋值：计算表达式并保存到 x
      xi = ui;  % 详解: 赋值：计算表达式并保存到 xi
    else  % 详解: 条件判断：else 分支
      uniform = 0;  % 详解: 赋值：计算表达式并保存到 uniform
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

if method(1)=='n',  % 详解: 条件判断：if (method(1)=='n',)
  yi = nearest(x,y,xi);  % 详解: 赋值：将 nearest(...) 的结果保存到 yi

elseif method(1)=='l',  % 详解: 条件判断：elseif (method(1)=='l',)
  if uniform  % 详解: 条件判断：if (uniform)
    yi = linear(x,y,xi);  % 详解: 赋值：将 linear(...) 的结果保存到 yi
  else  % 详解: 条件判断：else 分支
    yi = interp1q(x,y,xi);  % 详解: 赋值：将 interp1q(...) 的结果保存到 yi
  end  % 详解: 执行语句

elseif method(1)=='s',  % 详解: 条件判断：elseif (method(1)=='s',)
  yi = spline(x,y.',xi(:).').';  % 赋值：设置变量 yi  % 详解: 赋值：将 spline(...) 的结果保存到 yi  % 详解: 赋值：将 spline(...) 的结果保存到 yi

elseif method(1)=='c',  % 详解: 条件判断：elseif (method(1)=='c',)
  if uniform  % 详解: 条件判断：if (uniform)
    yi = cubic(x,y,xi);  % 详解: 赋值：将 cubic(...) 的结果保存到 yi
  else  % 详解: 条件判断：else 分支
    d = find(xi < min(x) | xi > max(x));  % 详解: 赋值：将 find(...) 的结果保存到 d
    yi = spline(x,y.',xi(:).').';  % 赋值：设置变量 yi  % 详解: 赋值：将 spline(...) 的结果保存到 yi  % 详解: 赋值：将 spline(...) 的结果保存到 yi
    if min(size(yi))==1, yi(d) = NaN; else yi(d,:) = NaN; end  % 详解: 条件判断：if (min(size(yi))==1, yi(d) = NaN; else yi(d,:) = NaN; end)
  end  % 详解: 执行语句

else  % 详解: 条件判断：else 分支
  error([deblank(method),' is an invalid method.']);  % 详解: 调用函数：error([deblank(method),' is an invalid method.'])

end  % 详解: 执行语句

if (min(size(yi))==1) & (prod(siz)>1), yi = reshape(yi,siz); end  % 详解: 条件判断：if ((min(size(yi))==1) & (prod(siz)>1), yi = reshape(yi,siz); end)


function F=linear(x,y,u)  % 详解: 执行语句


if nargin==2,  % 详解: 条件判断：if (nargin==2,)
  u = y; y = x;  % 详解: 赋值：计算表达式并保存到 u
  if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
  [nrows,ncols] = size(y);  % 详解: 获取向量/矩阵尺寸

elseif nargin==3,  % 详解: 条件判断：elseif (nargin==3,)
  if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
  if min(size(x))==1, x = x(:); end  % 详解: 条件判断：if (min(size(x))==1, x = x(:); end)
  [nrows,ncols] = size(y);  % 详解: 获取向量/矩阵尺寸
  if (min(size(x))~=1), error('X must be a vector.'); end  % 详解: 条件判断：if ((min(size(x))~=1), error('X must be a vector.'); end)
  [m,n] = size(x);  % 详解: 获取向量/矩阵尺寸
  if m ~= nrows,  % 详解: 条件判断：if (m ~= nrows,)
    error('The length of X must match the number of rows of Y.');  % 详解: 调用函数：error('The length of X must match the number of rows of Y.')
  end  % 详解: 执行语句
  u = 1 + (u-x(1))/(x(m)-x(1))*(nrows-1);  % 详解: 赋值：计算表达式并保存到 u
  
else  % 详解: 条件判断：else 分支
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')
end  % 详解: 执行语句

if isempty(u), F = []; return, end  % 详解: 条件判断：if (isempty(u), F = []; return, end)
if nrows<2, error('Y must have at least 2 rows.'); end  % 详解: 条件判断：if (nrows<2, error('Y must have at least 2 rows.'); end)

siz = size(u);  % 详解: 赋值：将 size(...) 的结果保存到 siz
u = u(:);  % 详解: 赋值：将 u(...) 的结果保存到 u
u = u(:,ones(1,ncols));  % 详解: 赋值：将 u(...) 的结果保存到 u
[m,n] = size(u);  % 详解: 获取向量/矩阵尺寸

uout = find(u<1 | u>nrows);  % 详解: 赋值：将 find(...) 的结果保存到 uout
if ~isempty(uout), u(uout) = 1; end  % 详解: 条件判断：if (~isempty(uout), u(uout) = 1; end)

s = (u - floor(u));  % 详解: 赋值：计算表达式并保存到 s
u = floor(u);  % 详解: 赋值：将 floor(...) 的结果保存到 u
if isempty(u), d = u; else d = find(u==nrows); end  % 详解: 条件判断：if (isempty(u), d = u; else d = find(u==nrows); end)
if length(d)>0, u(d) = u(d)-1; s(d) = s(d)+1; end  % 详解: 条件判断：if (length(d)>0, u(d) = u(d)-1; s(d) = s(d)+1; end)

v = (0:n-1)*nrows;  % 详解: 赋值：计算表达式并保存到 v
ndx = u+v(ones(m,1),:);  % 详解: 赋值：计算表达式并保存到 ndx
F =  ( y(ndx).*(1-s) + y(ndx+1).*s );  % 详解: 赋值：计算表达式并保存到 F

if ~isempty(uout), F(uout) = NaN; end  % 详解: 条件判断：if (~isempty(uout), F(uout) = NaN; end)

if (min(size(F))==1) & (prod(siz)>1), F = reshape(F,siz); end  % 详解: 条件判断：if ((min(size(F))==1) & (prod(siz)>1), F = reshape(F,siz); end)


function F=cubic(x,y,u)  % 详解: 执行语句



if nargin==2,  % 详解: 条件判断：if (nargin==2,)
  u = y; y = x;  % 详解: 赋值：计算表达式并保存到 u
  if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
  [nrows,ncols] = size(y);  % 详解: 获取向量/矩阵尺寸

elseif nargin==3,  % 详解: 条件判断：elseif (nargin==3,)
  if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
  if min(size(x))==1, x = x(:); end  % 详解: 条件判断：if (min(size(x))==1, x = x(:); end)
  [nrows,ncols] = size(y);  % 详解: 获取向量/矩阵尺寸
  if (min(size(x))~=1), error('X must be a vector.'); end  % 详解: 条件判断：if ((min(size(x))~=1), error('X must be a vector.'); end)
  [m,n] = size(x);  % 详解: 获取向量/矩阵尺寸
  if m ~= nrows,  % 详解: 条件判断：if (m ~= nrows,)
    error('The length of X must match the number of rows of Y.');  % 详解: 调用函数：error('The length of X must match the number of rows of Y.')
  end  % 详解: 执行语句
  u = 1 + (u-x(1))/(x(m)-x(1))*(nrows-1);  % 详解: 赋值：计算表达式并保存到 u
  
else  % 详解: 条件判断：else 分支
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')
end  % 详解: 执行语句

if isempty(u), F = []; return, end  % 详解: 条件判断：if (isempty(u), F = []; return, end)
if nrows<3, error('Y must have at least 3 rows.'); end  % 详解: 条件判断：if (nrows<3, error('Y must have at least 3 rows.'); end)

siz = size(u); u = u(:);  % 详解: 赋值：将 size(...) 的结果保存到 siz
u = u(:,ones(1,ncols));  % 详解: 赋值：将 u(...) 的结果保存到 u
[m,n] = size(u);  % 详解: 获取向量/矩阵尺寸

if isempty(u), uout = u; else uout = find(u<1 | u>nrows); end  % 详解: 条件判断：if (isempty(u), uout = u; else uout = find(u<1 | u>nrows); end)
if ~isempty(uout), u(uout) = 1; end  % 详解: 条件判断：if (~isempty(uout), u(uout) = 1; end)

s = (u - floor(u));  % 详解: 赋值：计算表达式并保存到 s
u = floor(u);  % 详解: 赋值：将 floor(...) 的结果保存到 u
if isempty(u), d = u; else d = find(u==nrows); end  % 详解: 条件判断：if (isempty(u), d = u; else d = find(u==nrows); end)
if length(d)>0, u(d) = u(d)-1; s(d) = s(d)+1; end  % 详解: 条件判断：if (length(d)>0, u(d) = u(d)-1; s(d) = s(d)+1; end)

y = [3*y(1,:)-3*y(2,:)+y(3,:);y;3*y(nrows,:)-3*y(nrows-1,:)+y(nrows-2,:)];  % 详解: 赋值：计算表达式并保存到 y
nrows = nrows + 2;  % 详解: 赋值：计算表达式并保存到 nrows

s2 = s.*s; s3 = s.*s2;  % 详解: 赋值：计算表达式并保存到 s2
v = (0:n-1)*nrows;  % 详解: 赋值：计算表达式并保存到 v
ndx = u+v(ones(m,1),:);  % 详解: 赋值：计算表达式并保存到 ndx
F = y(ndx).*(-s3+2*s2-s) + y(ndx+1).*(3*s3-5*s2+2) + ...  % 详解: 赋值：将 y(...) 的结果保存到 F
    y(ndx+2).*(-3*s3+4*s2+s) + y(ndx+3).*(s3-s2);  % 详解: 调用函数：y(ndx+2).*(-3*s3+4*s2+s) + y(ndx+3).*(s3-s2)
F = F/2;  % 详解: 赋值：计算表达式并保存到 F

if ~isempty(uout), F(uout) = NaN; end  % 详解: 条件判断：if (~isempty(uout), F(uout) = NaN; end)

if (min(size(F))==1) & (prod(siz)>1), F = reshape(F,siz); end  % 详解: 条件判断：if ((min(size(F))==1) & (prod(siz)>1), F = reshape(F,siz); end)


function F = nearest(x,y,u)  % 详解: 执行语句


if nargin==2,  % 详解: 条件判断：if (nargin==2,)
  u = y; y = x;  % 详解: 赋值：计算表达式并保存到 u
  if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
  [nrows,ncols] = size(y);  % 详解: 获取向量/矩阵尺寸

elseif nargin==3,  % 详解: 条件判断：elseif (nargin==3,)
  if min(size(y))==1, y = y(:); end  % 详解: 条件判断：if (min(size(y))==1, y = y(:); end)
  if min(size(x))==1, x = x(:); end  % 详解: 条件判断：if (min(size(x))==1, x = x(:); end)
  [nrows,ncols] = size(y);  % 详解: 获取向量/矩阵尺寸
  if (min(size(x))~=1), error('X must be a vector.'); end  % 详解: 条件判断：if ((min(size(x))~=1), error('X must be a vector.'); end)
  [m,n] = size(x);  % 详解: 获取向量/矩阵尺寸
  if m ~= nrows,  % 详解: 条件判断：if (m ~= nrows,)
    error('The length of X must match the number of rows of Y.');  % 详解: 调用函数：error('The length of X must match the number of rows of Y.')
  end  % 详解: 执行语句
  u = 1 + (u-x(1))/(x(m)-x(1))*(nrows-1);  % 详解: 赋值：计算表达式并保存到 u
  
else  % 详解: 条件判断：else 分支
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')
end  % 详解: 执行语句

if isempty(u), F = []; return, end  % 详解: 条件判断：if (isempty(u), F = []; return, end)
if nrows<2, error('Y must have at least 2 rows.'); end  % 详解: 条件判断：if (nrows<2, error('Y must have at least 2 rows.'); end)

siz = size(u); u = u(:);  % 详解: 赋值：将 size(...) 的结果保存到 siz
u = u(:,ones(1,ncols));  % 详解: 赋值：将 u(...) 的结果保存到 u
[m,n] = size(u);  % 详解: 获取向量/矩阵尺寸

uout = find(u<.5 | u>=nrows+.5);  % 详解: 赋值：将 find(...) 的结果保存到 uout
if ~isempty(uout), u(uout) = 1; end  % 详解: 条件判断：if (~isempty(uout), u(uout) = 1; end)

s = (u - round(u));  % 详解: 赋值：计算表达式并保存到 s
u = round(u);  % 详解: 赋值：将 round(...) 的结果保存到 u

v = (0:n-1)*nrows;  % 详解: 赋值：计算表达式并保存到 v
ndx = u+v(ones(m,1),:);  % 详解: 赋值：计算表达式并保存到 ndx
F = y(ndx);  % 详解: 赋值：将 y(...) 的结果保存到 F

if ~isempty(uout), F(uout) = NaN; end  % 详解: 条件判断：if (~isempty(uout), F(uout) = NaN; end)

if (min(size(F))==1) & (prod(siz)>1), F = reshape(F,siz); end  % 详解: 条件判断：if ((min(size(F))==1) & (prod(siz)>1), F = reshape(F,siz); end)




