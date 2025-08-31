% 文件: INTERP2.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function zi = interp2(varargin)  % 详解: 执行语句


error(nargchk(1,6,nargin));  % 详解: 调用函数：error(nargchk(1,6,nargin))

bypass = 0;  % 详解: 赋值：计算表达式并保存到 bypass
uniform = 1;  % 详解: 赋值：计算表达式并保存到 uniform
if isstr(varargin{end}),  % 详解: 条件判断：if (isstr(varargin{end}),)
  narg = nargin-1;  % 详解: 赋值：计算表达式并保存到 narg
  method = [varargin{end} '    '];  % 详解: 赋值：计算表达式并保存到 method
  if method(1)=='*',  % 详解: 条件判断：if (method(1)=='*',)
    if method(2)=='l' | all(method(2:4)=='bil'),  % 详解: 条件判断：if (method(2)=='l' | all(method(2:4)=='bil'),)
      zi = linear(varargin{1:end-1});  % 详解: 赋值：将 linear(...) 的结果保存到 zi
      return  % 详解: 返回：从当前函数返回

    elseif method(2)=='c' | all(method(2:4)=='bic'),  % 详解: 条件判断：elseif (method(2)=='c' | all(method(2:4)=='bic'),)
      zi = cubic(varargin{1:end-1});  % 详解: 赋值：将 cubic(...) 的结果保存到 zi
      return  % 详解: 返回：从当前函数返回

    elseif method(2)=='n',  % 详解: 条件判断：elseif (method(2)=='n',)
      zi = nearest(varargin{1:end-1});  % 详解: 赋值：将 nearest(...) 的结果保存到 zi
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

if narg==1,  % 详解: 条件判断：if (narg==1,)
  [nrows,ncols] = size(varargin{1});  % 详解: 获取向量/矩阵尺寸
  xi = 1:.5:ncols; yi = (1:.5:nrows)';  % 赋值：设置变量 xi  % 详解: 赋值：计算表达式并保存到 xi  % 详解: 赋值：计算表达式并保存到 xi
  x = 1:ncols; y = (1:nrows);  % 详解: 赋值：计算表达式并保存到 x
  [msg,x,y,z,xi,yi] = xyzchk(x,y,varargin{1},xi,yi);  % 详解: 执行语句

elseif narg==2.  % 详解: 条件判断：elseif (narg==2.)
  [nrows,ncols] = size(varargin{1});  % 详解: 获取向量/矩阵尺寸
  ntimes = floor(varargin{2}(1));  % 详解: 赋值：将 floor(...) 的结果保存到 ntimes
  xi = 1:1/(2^ntimes):ncols; yi = (1:1/(2^ntimes):nrows)';  % 赋值：设置变量 xi  % 详解: 赋值：计算表达式并保存到 xi  % 详解: 赋值：计算表达式并保存到 xi
  x = 1:ncols; y = (1:nrows);  % 详解: 赋值：计算表达式并保存到 x
  [msg,x,y,z,xi,yi] = xyzchk(x,y,varargin{1},xi,yi);  % 详解: 执行语句

elseif narg==3,  % 详解: 条件判断：elseif (narg==3,)
  [nrows,ncols] = size(varargin{1});  % 详解: 获取向量/矩阵尺寸
  x = 1:ncols; y = (1:nrows);  % 详解: 赋值：计算表达式并保存到 x
  [msg,x,y,z,xi,yi] = xyzchk(x,y,varargin{1:3});  % 详解: 执行语句

elseif narg==4,  % 详解: 条件判断：elseif (narg==4,)
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')

elseif narg==5,  % 详解: 条件判断：elseif (narg==5,)
  [msg,x,y,z,xi,yi] = xyzchk(varargin{1:5});  % 详解: 执行语句

end  % 详解: 执行语句

if ~isempty(msg), error(msg); end  % 详解: 条件判断：if (~isempty(msg), error(msg); end)

xx = x(1,:); yy = y(:,1);  % 详解: 赋值：将 x(...) 的结果保存到 xx
if (size(x,2)>1 & ~isequal(repmat(xx,size(x,1),1),x)) | ...  % 详解: 条件判断：if ((size(x,2)>1 & ~isequal(repmat(xx,size(x,1),1),x)) | ...)
   (size(y,1)>1 & ~isequal(repmat(yy,1,size(y,2)),y)),  % 详解: 获取向量/矩阵尺寸
  error(sprintf(['X and Y must be matrices produced by MESHGRID. Use' ...  % 详解: 执行语句
     ' GRIDDATA instead \nof INTERP2 for scattered data.']));  % 详解: 执行语句
end  % 详解: 执行语句

if ~bypass,  % 详解: 条件判断：if (~bypass,)
  xx = xx.'; % Make sure it's a column.  % 详解: 赋值：计算表达式并保存到 xx
  dx = diff(xx); dy = diff(yy);  % 详解: 赋值：将 diff(...) 的结果保存到 dx
  xdiff = max(abs(diff(dx))); if isempty(xdiff), xdiff = 0; end  % 详解: 赋值：将 max(...) 的结果保存到 xdiff
  ydiff = max(abs(diff(dy))); if isempty(ydiff), ydiff = 0; end  % 详解: 赋值：将 max(...) 的结果保存到 ydiff
  if (xdiff > eps*max(abs(xx))) | (ydiff > eps*max(abs(yy))),  % 详解: 条件判断：if ((xdiff > eps*max(abs(xx))) | (ydiff > eps*max(abs(yy))),)
    if any(dx < 0),  % 详解: 条件判断：if (any(dx < 0),)
      x = fliplr(x); y = fliplr(y); z = fliplr(z);  % 详解: 赋值：将 fliplr(...) 的结果保存到 x
      xx = flipud(xx); dx = -flipud(dx);  % 详解: 赋值：将 flipud(...) 的结果保存到 xx
    end  % 详解: 执行语句
    if any(dy < 0),  % 详解: 条件判断：if (any(dy < 0),)
      x = flipud(x); y = flipud(y); z = flipud(z);  % 详解: 赋值：将 flipud(...) 的结果保存到 x
      yy = flipud(yy); dy = -flipud(dy);  % 详解: 赋值：将 flipud(...) 的结果保存到 yy
    end  % 详解: 执行语句
  
    if any(dx<=0) | any(dy<=0),  % 详解: 条件判断：if (any(dx<=0) | any(dy<=0),)
      error('X and Y must be monotonic vectors or matrices produced by MESHGRID.');  % 详解: 调用函数：error('X and Y must be monotonic vectors or matrices produced by MESHGRID.')
    end  % 详解: 执行语句
  
    if method(1)~='c',  % 详解: 条件判断：if (method(1)~='c',)
      [xxi,j] = sort(xi(:));  % 详解: 执行语句
      [dum,i] = sort([xx;xxi]);  % 详解: 执行语句
      ui(i) = (1:length(i));  % 详解: 调用函数：ui(i) = (1:length(i))
      ui = (ui(length(xx)+1:end)-(1:length(xxi)))';  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 ui  % 详解: 赋值：计算表达式并保存到 ui
      ui(j) = ui;  % 详解: 执行语句
    
      ui(ui<1) = 1;  % 详解: 执行语句
      ui(ui>length(xx)-1) = length(xx)-1;  % 详解: 获取向量/矩阵尺寸
      ui = ui + (xi(:)-xx(ui))./(xx(ui+1)-xx(ui));  % 详解: 赋值：计算表达式并保存到 ui
     
      [yyi,j] = sort(yi(:));  % 详解: 执行语句
      [dum,i] = sort([yy;yyi(:)]);  % 详解: 执行语句
      vi(i) = (1:length(i));  % 详解: 调用函数：vi(i) = (1:length(i))
      vi = (vi(length(yy)+1:end)-(1:length(yyi)))';  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 vi  % 详解: 赋值：计算表达式并保存到 vi
      vi(j) = vi;  % 详解: 执行语句
    
      vi(vi<1) = 1;  % 详解: 执行语句
      vi(vi>length(yy)-1) = length(yy)-1;  % 详解: 获取向量/矩阵尺寸
      vi = vi + (yi(:)-yy(vi))./(yy(vi+1)-yy(vi));  % 详解: 赋值：计算表达式并保存到 vi
      
      [x,y] = meshgrid(1:size(x,2),1:size(y,1));  % 详解: 获取向量/矩阵尺寸
      xi(:) = ui; yi(:) = vi;  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      uniform = 0;  % 详解: 赋值：计算表达式并保存到 uniform
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

method = [lower(method),'   '];  % 详解: 赋值：计算表达式并保存到 method

if method(1)=='l' | all(method(1:3)=='bil'),  % 详解: 条件判断：if (method(1)=='l' | all(method(1:3)=='bil'),)
  zi = linear(x,y,z,xi,yi);  % 详解: 赋值：将 linear(...) 的结果保存到 zi

elseif method(1)=='c' | all(method(1:3)=='bic'),  % 详解: 条件判断：elseif (method(1)=='c' | all(method(1:3)=='bic'),)
  if uniform  % 详解: 条件判断：if (uniform)
    zi = cubic(x,y,z,xi,yi);  % 详解: 赋值：将 cubic(...) 的结果保存到 zi
  else  % 详解: 条件判断：else 分支
    d = find(xi < min(x(:)) | xi > max(x(:)) | ...  % 详解: 赋值：将 find(...) 的结果保存到 d
             yi < min(y(:)) | yi > max(y(:)));  % 详解: 统计：最大/最小值
    zi = spline2(x,y,z,xi,yi);  % 详解: 赋值：将 spline2(...) 的结果保存到 zi
    zi(d) = NaN;  % 详解: 执行语句
  end  % 详解: 执行语句

elseif method(1)=='n',  % 详解: 条件判断：elseif (method(1)=='n',)
  zi = nearest(x,y,z,xi,yi);  % 详解: 赋值：将 nearest(...) 的结果保存到 zi

elseif method(1)=='s',  % 详解: 条件判断：elseif (method(1)=='s',)
  zi = spline2(x,y,z,xi,yi);  % 详解: 赋值：将 spline2(...) 的结果保存到 zi

else  % 详解: 条件判断：else 分支
  error([deblank(method),' is an invalid method.']);  % 详解: 调用函数：error([deblank(method),' is an invalid method.'])

end  % 详解: 执行语句

function F = linear(arg1,arg2,arg3,arg4,arg5)  % 详解: 执行语句


if nargin==1,  % 详解: 条件判断：if (nargin==1,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  s = 1:.5:ncols; sizs = size(s);  % 详解: 赋值：计算表达式并保存到 s
  t = (1:.5:nrows)'; sizt = size(t);  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 t  % 详解: 赋值：计算表达式并保存到 t
  s = s(ones(sizt),:);  % 详解: 赋值：将 s(...) 的结果保存到 s
  t = t(:,ones(sizs));  % 详解: 赋值：将 t(...) 的结果保存到 t

elseif nargin==2,  % 详解: 条件判断：elseif (nargin==2,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  ntimes = floor(arg2);  % 详解: 赋值：将 floor(...) 的结果保存到 ntimes
  s = 1:1/(2^ntimes):ncols; sizs = size(s);  % 详解: 赋值：计算表达式并保存到 s
  t = (1:1/(2^ntimes):nrows)'; sizt = size(t);  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 t  % 详解: 赋值：计算表达式并保存到 t
  s = s(ones(sizt),:);  % 详解: 赋值：将 s(...) 的结果保存到 s
  t = t(:,ones(sizs));  % 详解: 赋值：将 t(...) 的结果保存到 t

elseif nargin==3,  % 详解: 条件判断：elseif (nargin==3,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  s = arg2; t = arg3;  % 详解: 赋值：计算表达式并保存到 s

elseif nargin==4,  % 详解: 条件判断：elseif (nargin==4,)
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')

elseif nargin==5,  % 详解: 条件判断：elseif (nargin==5,)
  [nrows,ncols] = size(arg3);  % 详解: 获取向量/矩阵尺寸
  mx = prod(size(arg1)); my = prod(size(arg2));  % 详解: 赋值：将 prod(...) 的结果保存到 mx
  if any([mx my] ~= [ncols nrows]) & ...  % 详解: 条件判断：if (any([mx my] ~= [ncols nrows]) & ...)
     ~isequal(size(arg1),size(arg2),size(arg3))  % 详解: 获取向量/矩阵尺寸
    error('The lengths of the X and Y vectors must match Z.');  % 详解: 调用函数：error('The lengths of the X and Y vectors must match Z.')
  end  % 详解: 执行语句
  if any([nrows ncols]<[2 2]), error('Z must be at least 2-by-2.'); end  % 详解: 条件判断：if (any([nrows ncols]<[2 2]), error('Z must be at least 2-by-2.'); end)
  s = 1 + (arg4-arg1(1))/(arg1(mx)-arg1(1))*(ncols-1);  % 详解: 赋值：计算表达式并保存到 s
  t = 1 + (arg5-arg2(1))/(arg2(my)-arg2(1))*(nrows-1);  % 详解: 赋值：计算表达式并保存到 t
  
end  % 详解: 执行语句

if any([nrows ncols]<[2 2]), error('Z must be at least 2-by-2.'); end  % 详解: 条件判断：if (any([nrows ncols]<[2 2]), error('Z must be at least 2-by-2.'); end)
if ~isequal(size(s),size(t)),  % 详解: 条件判断：if (~isequal(size(s),size(t)),)
  error('XI and YI must be the same size.');  % 详解: 调用函数：error('XI and YI must be the same size.')
end  % 详解: 执行语句

sout = find((s<1)|(s>ncols));  % 详解: 赋值：将 find(...) 的结果保存到 sout
if length(sout)>0, s(sout) = ones(size(sout)); end  % 详解: 条件判断：if (length(sout)>0, s(sout) = ones(size(sout)); end)

tout = find((t<1)|(t>nrows));  % 详解: 赋值：将 find(...) 的结果保存到 tout
if length(tout)>0, t(tout) = ones(size(tout)); end  % 详解: 条件判断：if (length(tout)>0, t(tout) = ones(size(tout)); end)

ndx = floor(t)+floor(s-1)*nrows;  % 详解: 赋值：将 floor(...) 的结果保存到 ndx

if isempty(s), d = s; else d = find(s==ncols); end  % 详解: 条件判断：if (isempty(s), d = s; else d = find(s==ncols); end)
s(:) = (s - floor(s));  % 详解: 调用函数：s(:) = (s - floor(s))
if length(d)>0, s(d) = s(d)+1; ndx(d) = ndx(d)-nrows; end  % 详解: 条件判断：if (length(d)>0, s(d) = s(d)+1; ndx(d) = ndx(d)-nrows; end)

if isempty(t), d = t; else d = find(t==nrows); end  % 详解: 条件判断：if (isempty(t), d = t; else d = find(t==nrows); end)
t(:) = (t - floor(t));  % 详解: 调用函数：t(:) = (t - floor(t))
if length(d)>0, t(d) = t(d)+1; ndx(d) = ndx(d)-1; end  % 详解: 条件判断：if (length(d)>0, t(d) = t(d)+1; ndx(d) = ndx(d)-1; end)
d = [];  % 详解: 赋值：计算表达式并保存到 d

if nargin==5,  % 详解: 条件判断：if (nargin==5,)
  F =  ( arg3(ndx).*(1-t) + arg3(ndx+1).*t ).*(1-s) + ...  % 详解: 赋值：计算表达式并保存到 F
       ( arg3(ndx+nrows).*(1-t) + arg3(ndx+(nrows+1)).*t ).*s;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  F =  ( arg1(ndx).*(1-t) + arg1(ndx+1).*t ).*(1-s) + ...  % 详解: 赋值：计算表达式并保存到 F
       ( arg1(ndx+nrows).*(1-t) + arg1(ndx+(nrows+1)).*t ).*s;  % 详解: 执行语句
end  % 详解: 执行语句

if length(sout)>0, F(sout) = NaN; end  % 详解: 条件判断：if (length(sout)>0, F(sout) = NaN; end)
if length(tout)>0, F(tout) = NaN; end  % 详解: 条件判断：if (length(tout)>0, F(tout) = NaN; end)



function F = cubic(arg1,arg2,arg3,arg4,arg5)  % 详解: 执行语句



if nargin==1,  % 详解: 条件判断：if (nargin==1,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  s = 1:.5:ncols; sizs = size(s);  % 详解: 赋值：计算表达式并保存到 s
  t = (1:.5:nrows)'; sizt = size(t);  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 t  % 详解: 赋值：计算表达式并保存到 t
  s = s(ones(sizt),:);  % 详解: 赋值：将 s(...) 的结果保存到 s
  t = t(:,ones(sizs));  % 详解: 赋值：将 t(...) 的结果保存到 t

elseif nargin==2,  % 详解: 条件判断：elseif (nargin==2,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  ntimes = floor(arg2);  % 详解: 赋值：将 floor(...) 的结果保存到 ntimes
  s = 1:1/(2^ntimes):ncols; sizs = size(s);  % 详解: 赋值：计算表达式并保存到 s
  t = (1:1/(2^ntimes):nrows)'; sizt = size(t);  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 t  % 详解: 赋值：计算表达式并保存到 t
  s = s(ones(sizt),:);  % 详解: 赋值：将 s(...) 的结果保存到 s
  t = t(:,ones(sizs));  % 详解: 赋值：将 t(...) 的结果保存到 t

elseif nargin==3,  % 详解: 条件判断：elseif (nargin==3,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  s = arg2; t = arg3;  % 详解: 赋值：计算表达式并保存到 s

elseif nargin==4,  % 详解: 条件判断：elseif (nargin==4,)
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')

elseif nargin==5,  % 详解: 条件判断：elseif (nargin==5,)
  [nrows,ncols] = size(arg3);  % 详解: 获取向量/矩阵尺寸
  mx = prod(size(arg1)); my = prod(size(arg2));  % 详解: 赋值：将 prod(...) 的结果保存到 mx
  if any([mx my] ~= [ncols nrows]) & ...  % 详解: 条件判断：if (any([mx my] ~= [ncols nrows]) & ...)
     ~isequal(size(arg1),size(arg2),size(arg3))  % 详解: 获取向量/矩阵尺寸
    error('The lengths of the X and Y vectors must match Z.');  % 详解: 调用函数：error('The lengths of the X and Y vectors must match Z.')
  end  % 详解: 执行语句
  if any([nrows ncols]<[3 3]), error('Z must be at least 3-by-3.'); end  % 详解: 条件判断：if (any([nrows ncols]<[3 3]), error('Z must be at least 3-by-3.'); end)
  s = 1 + (arg4-arg1(1))/(arg1(mx)-arg1(1))*(ncols-1);  % 详解: 赋值：计算表达式并保存到 s
  t = 1 + (arg5-arg2(1))/(arg2(my)-arg2(1))*(nrows-1);  % 详解: 赋值：计算表达式并保存到 t
  
end  % 详解: 执行语句

if any([nrows ncols]<[3 3]), error('Z must be at least 3-by-3.'); end  % 详解: 条件判断：if (any([nrows ncols]<[3 3]), error('Z must be at least 3-by-3.'); end)
if ~isequal(size(s),size(t)),  % 详解: 条件判断：if (~isequal(size(s),size(t)),)
  error('XI and YI must be the same size.');  % 详解: 调用函数：error('XI and YI must be the same size.')
end  % 详解: 执行语句

sout = find((s<1)|(s>ncols));  % 详解: 赋值：将 find(...) 的结果保存到 sout
if length(sout)>0, s(sout) = ones(size(sout)); end  % 详解: 条件判断：if (length(sout)>0, s(sout) = ones(size(sout)); end)

tout = find((t<1)|(t>nrows));  % 详解: 赋值：将 find(...) 的结果保存到 tout
if length(tout)>0, t(tout) = ones(size(tout)); end  % 详解: 条件判断：if (length(tout)>0, t(tout) = ones(size(tout)); end)

ndx = floor(t)+floor(s-1)*(nrows+2);  % 详解: 赋值：将 floor(...) 的结果保存到 ndx

if isempty(s), d = s; else d = find(s==ncols); end  % 详解: 条件判断：if (isempty(s), d = s; else d = find(s==ncols); end)
s(:) = (s - floor(s));  % 详解: 调用函数：s(:) = (s - floor(s))
if length(d)>0, s(d) = s(d)+1; ndx(d) = ndx(d)-nrows-2; end  % 详解: 条件判断：if (length(d)>0, s(d) = s(d)+1; ndx(d) = ndx(d)-nrows-2; end)

if isempty(t), d = t; else d = find(t==nrows); end  % 详解: 条件判断：if (isempty(t), d = t; else d = find(t==nrows); end)
t(:) = (t - floor(t));  % 详解: 调用函数：t(:) = (t - floor(t))
if length(d)>0, t(d) = t(d)+1; ndx(d) = ndx(d)-1; end  % 详解: 条件判断：if (length(d)>0, t(d) = t(d)+1; ndx(d) = ndx(d)-1; end)
d = [];  % 详解: 赋值：计算表达式并保存到 d

if nargin==5,  % 详解: 条件判断：if (nargin==5,)
  zz = zeros(size(arg3)+2);  % 详解: 赋值：将 zeros(...) 的结果保存到 zz
  zz(1,2:ncols+1) = 3*arg3(1,:)-3*arg3(2,:)+arg3(3,:);  % 详解: 调用函数：zz(1,2:ncols+1) = 3*arg3(1,:)-3*arg3(2,:)+arg3(3,:)
  zz(2:nrows+1,2:ncols+1) = arg3;  % 详解: 执行语句
  zz(nrows+2,2:ncols+1) = 3*arg3(nrows,:)-3*arg3(nrows-1,:)+arg3(nrows-2,:);  % 详解: 调用函数：zz(nrows+2,2:ncols+1) = 3*arg3(nrows,:)-3*arg3(nrows-1,:)+arg3(nrows-2,:)
  zz(:,1) = 3*zz(:,2)-3*zz(:,3)+zz(:,4);  % 详解: 调用函数：zz(:,1) = 3*zz(:,2)-3*zz(:,3)+zz(:,4)
  zz(:,ncols+2) = 3*zz(:,ncols+1)-3*zz(:,ncols)+zz(:,ncols-1);  % 详解: 调用函数：zz(:,ncols+2) = 3*zz(:,ncols+1)-3*zz(:,ncols)+zz(:,ncols-1)
  nrows = nrows+2; ncols = ncols+2;  % 详解: 赋值：计算表达式并保存到 nrows
else  % 详解: 条件判断：else 分支
  zz = zeros(size(arg1)+2);  % 详解: 赋值：将 zeros(...) 的结果保存到 zz
  zz(1,2:ncols+1) = 3*arg1(1,:)-3*arg1(2,:)+arg1(3,:);  % 详解: 调用函数：zz(1,2:ncols+1) = 3*arg1(1,:)-3*arg1(2,:)+arg1(3,:)
  zz(2:nrows+1,2:ncols+1) = arg1;  % 详解: 执行语句
  zz(nrows+2,2:ncols+1) = 3*arg1(nrows,:)-3*arg1(nrows-1,:)+arg1(nrows-2,:);  % 详解: 调用函数：zz(nrows+2,2:ncols+1) = 3*arg1(nrows,:)-3*arg1(nrows-1,:)+arg1(nrows-2,:)
  zz(:,1) = 3*zz(:,2)-3*zz(:,3)+zz(:,4);  % 详解: 调用函数：zz(:,1) = 3*zz(:,2)-3*zz(:,3)+zz(:,4)
  zz(:,ncols+2) = 3*zz(:,ncols+1)-3*zz(:,ncols)+zz(:,ncols-1);  % 详解: 调用函数：zz(:,ncols+2) = 3*zz(:,ncols+1)-3*zz(:,ncols)+zz(:,ncols-1)
  nrows = nrows+2; ncols = ncols+2;  % 详解: 赋值：计算表达式并保存到 nrows
end  % 详解: 执行语句

t0 = ((2-t).*t-1).*t;  % 详解: 赋值：计算表达式并保存到 t0
t1 = (3*t-5).*t.*t+2;  % 详解: 赋值：计算表达式并保存到 t1
t2 = ((4-3*t).*t+1).*t;  % 详解: 赋值：计算表达式并保存到 t2
t(:) = (t-1).*t.*t;  % 详解: 执行语句
F     = ( zz(ndx).*t0 + zz(ndx+1).*t1 + zz(ndx+2).*t2 + zz(ndx+3).*t ) ...  % 详解: 赋值：计算表达式并保存到 F
        .* (((2-s).*s-1).*s);  % 详解: 执行语句
ndx(:) = ndx + nrows;  % 详解: 执行语句
F(:)  = F + ( zz(ndx).*t0 + zz(ndx+1).*t1 + zz(ndx+2).*t2 + zz(ndx+3).*t ) ...  % 详解: 执行语句
        .* ((3*s-5).*s.*s+2);  % 详解: 执行语句
ndx(:) = ndx + nrows;  % 详解: 执行语句
F(:)  = F + ( zz(ndx).*t0 + zz(ndx+1).*t1 + zz(ndx+2).*t2 + zz(ndx+3).*t ) ...  % 详解: 执行语句
        .* (((4-3*s).*s+1).*s);  % 详解: 执行语句
ndx(:) = ndx + nrows;  % 详解: 执行语句
F(:)  = F + ( zz(ndx).*t0 + zz(ndx+1).*t1 + zz(ndx+2).*t2 + zz(ndx+3).*t ) ...  % 详解: 执行语句
       .* ((s-1).*s.*s);  % 详解: 执行语句
F(:) = F/4;  % 详解: 执行语句

if length(sout)>0, F(sout) = NaN; end  % 详解: 条件判断：if (length(sout)>0, F(sout) = NaN; end)
if length(tout)>0, F(tout) = NaN; end  % 详解: 条件判断：if (length(tout)>0, F(tout) = NaN; end)


function F = nearest(arg1,arg2,arg3,arg4,arg5)  % 详解: 执行语句


if nargin==1,  % 详解: 条件判断：if (nargin==1,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  u = ones(2*nrows-1,1)*(1:.5:ncols);  % 详解: 赋值：将 ones(...) 的结果保存到 u
  v = (1:.5:nrows)'*ones(1,2*ncols-1);  % 创建全 1 矩阵/数组  % 详解: 赋值：计算表达式并保存到 v  % 详解: 赋值：计算表达式并保存到 v

elseif nargin==2,  % 详解: 条件判断：elseif (nargin==2,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  ntimes = floor(arg2);  % 详解: 赋值：将 floor(...) 的结果保存到 ntimes
  u = 1:1/(2^ntimes):ncols; sizu = size(u);  % 详解: 赋值：计算表达式并保存到 u
  v = (1:1/(2^ntimes):nrows)'; sizv = size(v);  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 v  % 详解: 赋值：计算表达式并保存到 v
  u = u(ones(sizv),:);  % 详解: 赋值：将 u(...) 的结果保存到 u
  v = v(:,ones(sizu));  % 详解: 赋值：将 v(...) 的结果保存到 v

elseif nargin==3,  % 详解: 条件判断：elseif (nargin==3,)
  [nrows,ncols] = size(arg1);  % 详解: 获取向量/矩阵尺寸
  u = arg2; v = arg3;  % 详解: 赋值：计算表达式并保存到 u

elseif nargin==4,  % 详解: 条件判断：elseif (nargin==4,)
  error('Wrong number of input arguments.');  % 详解: 调用函数：error('Wrong number of input arguments.')

elseif nargin==5,  % 详解: 条件判断：elseif (nargin==5,)
  [nrows,ncols] = size(arg3);  % 详解: 获取向量/矩阵尺寸
  mx = prod(size(arg1)); my = prod(size(arg2));  % 详解: 赋值：将 prod(...) 的结果保存到 mx
  if any([mx my] ~= [ncols nrows]) & (size(arg1)~=size(arg3) |   ...  % 详解: 条件判断：if (any([mx my] ~= [ncols nrows]) & (size(arg1)~=size(arg3) |   ...)
    size(arg2)~=size(arg3)),  % 详解: 获取向量/矩阵尺寸
    error('The lengths of the X and Y vectors must match Z.');  % 详解: 调用函数：error('The lengths of the X and Y vectors must match Z.')
  end  % 详解: 执行语句
  if all([nrows ncols]>[1 1]),  % 详解: 条件判断：if (all([nrows ncols]>[1 1]),)
    u = 1 + (arg4-arg1(1))/(arg1(mx)-arg1(1))*(ncols-1);  % 详解: 赋值：计算表达式并保存到 u
    v = 1 + (arg5-arg2(1))/(arg2(my)-arg2(1))*(nrows-1);  % 详解: 赋值：计算表达式并保存到 v
  else  % 详解: 条件判断：else 分支
    u = 1 + (arg4-arg1(1));  % 详解: 赋值：计算表达式并保存到 u
    v = 1 + (arg5-arg2(1));  % 详解: 赋值：计算表达式并保存到 v
  end  % 详解: 执行语句
end  % 详解: 执行语句

if size(u)~=size(v), error('XI and YI must be the same size.'); end  % 详解: 条件判断：if (size(u)~=size(v), error('XI and YI must be the same size.'); end)

uout = (u<.5)|(u>=ncols+.5);  % 详解: 赋值：计算表达式并保存到 uout
nuout = sum(uout(:));  % 详解: 赋值：将 sum(...) 的结果保存到 nuout
if any(uout(:)), u(uout) = ones(nuout,1); end  % 详解: 条件判断：if (any(uout(:)), u(uout) = ones(nuout,1); end)

vout = (v<.5)|(v>=nrows+.5);  % 详解: 赋值：计算表达式并保存到 vout
nvout = sum(vout(:));  % 详解: 赋值：将 sum(...) 的结果保存到 nvout
if any(vout(:)), v(vout) = ones(nvout,1); end  % 详解: 条件判断：if (any(vout(:)), v(vout) = ones(nvout,1); end)

s = (u - round(u));  t = (v - round(v));  % 详解: 赋值：计算表达式并保存到 s
u = round(u); v = round(v);  % 详解: 赋值：将 round(...) 的结果保存到 u

ndx = v+(u-1)*nrows;  % 详解: 赋值：计算表达式并保存到 ndx
if nargin==5,  % 详解: 条件判断：if (nargin==5,)
  F = arg3(ndx);  % 详解: 赋值：将 arg3(...) 的结果保存到 F
else  % 详解: 条件判断：else 分支
  F = arg1(ndx);  % 详解: 赋值：将 arg1(...) 的结果保存到 F
end  % 详解: 执行语句

if any(uout(:)), F(uout) = NaN; end  % 详解: 条件判断：if (any(uout(:)), F(uout) = NaN; end)
if any(vout(:)), F(vout) = NaN; end  % 详解: 条件判断：if (any(vout(:)), F(vout) = NaN; end)

function F = spline2(varargin)  % 详解: 执行语句

varargin{1} = varargin{1}(1,:);  % 详解: 执行语句
varargin{2} = varargin{2}(:,1).';  % 执行语句  % 详解: 执行语句  % 详解: 执行语句

xi = varargin{4}; yi = varargin{5};  % 详解: 赋值：计算表达式并保存到 xi
xxi = xi(1,:); yyi = yi(:,1);  % 详解: 赋值：将 xi(...) 的结果保存到 xxi
if (size(xi,2)>1 & ~isequal(repmat(xxi,size(xi,1),1),xi)) | ...  % 详解: 条件判断：if ((size(xi,2)>1 & ~isequal(repmat(xxi,size(xi,1),1),xi)) | ...)
   (size(yi,1)>1 & ~isequal(repmat(yyi,1,size(yi,2)),yi)),  % 详解: 获取向量/矩阵尺寸
  F = splncore(varargin(2:-1:1),varargin{3},varargin(5:-1:4));  % 详解: 赋值：将 splncore(...) 的结果保存到 F
else  % 详解: 条件判断：else 分支
  F = splncore(varargin(2:-1:1),varargin{3},{yyi(:).' xxi},'gridded');  % 赋值：设置变量 F  % 详解: 赋值：将 splncore(...) 的结果保存到 F  % 详解: 赋值：将 splncore(...) 的结果保存到 F
end  % 详解: 执行语句




