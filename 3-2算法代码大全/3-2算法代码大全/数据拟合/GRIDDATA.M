% 文件: GRIDDATA.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [xi,yi,zi] = griddata(x,y,z,xi,yi,method)  % 详解: 函数定义：griddata(x,y,z,xi,yi,method), 返回：xi,yi,zi


error(nargchk(5,6,nargin))  % 详解: 调用函数：error(nargchk(5,6,nargin))

[msg,x,y,z,xi,yi] = xyzchk(x,y,z,xi,yi);  % 详解: 执行语句
if ~isempty(msg), error(msg); end  % 详解: 条件判断：if (~isempty(msg), error(msg); end)

if nargin<6, method = 'linear'; end  % 详解: 条件判断：if (nargin<6, method = 'linear'; end)
if ~isstr(method),  % 详解: 条件判断：if (~isstr(method),)
  error('METHOD must be one of ''linear'',''cubic'',''nearest'', or ''v4''.');  % 详解: 调用函数：error('METHOD must be one of ''linear'',''cubic'',''nearest'', or ''v4''.')
end  % 详解: 执行语句



sz = prod(size(x));  % 详解: 赋值：将 prod(...) 的结果保存到 sz
x = reshape(x,sz,1);  % 详解: 赋值：将 reshape(...) 的结果保存到 x
y = reshape(y,sz,1);  % 详解: 赋值：将 reshape(...) 的结果保存到 y
z = reshape(z,sz,1);  % 详解: 赋值：将 reshape(...) 的结果保存到 z
sxyz = sortrows([x y z],[2 1]);  % 详解: 赋值：将 sortrows(...) 的结果保存到 sxyz
x = sxyz(:,1);  % 详解: 赋值：将 sxyz(...) 的结果保存到 x
y = sxyz(:,2);  % 详解: 赋值：将 sxyz(...) 的结果保存到 y
z = sxyz(:,3);  % 详解: 赋值：将 sxyz(...) 的结果保存到 z
ind = [0; y(2:end) == y(1:end-1) & x(2:end) == x(1:end-1); 0];  % 详解: 赋值：计算表达式并保存到 ind
if sum(ind) > 0  % 详解: 条件判断：if (sum(ind) > 0)
  warning('Duplicate x-y data points detected: using average of the z values');  % 详解: 调用函数：warning('Duplicate x-y data points detected: using average of the z values')
  fs = find(ind(1:end-1) == 0 & ind(2:end) == 1);  % 详解: 赋值：将 find(...) 的结果保存到 fs
  fe = find(ind(1:end-1) == 1 & ind(2:end) == 0);  % 详解: 赋值：将 find(...) 的结果保存到 fe
  for i = 1 : length(fs)  % 详解: for 循环：迭代变量 i 遍历 1 : length(fs)
    z(fe(i)) = mean(z(fs(i):fe(i)));  % 详解: 调用函数：z(fe(i)) = mean(z(fs(i):fe(i)))
  end  % 详解: 执行语句
  x = x(~ind(2:end));  % 详解: 赋值：将 x(...) 的结果保存到 x
  y = y(~ind(2:end));  % 详解: 赋值：将 y(...) 的结果保存到 y
  z = z(~ind(2:end));  % 详解: 赋值：将 z(...) 的结果保存到 z
end  % 详解: 执行语句

switch lower(method),  % 详解: 多分支选择：switch (lower(method),)
  case 'linear'  % 详解: 分支：case 'linear'
    zi = linear(x,y,z,xi,yi);  % 详解: 赋值：将 linear(...) 的结果保存到 zi
  case 'cubic'  % 详解: 分支：case 'cubic'
    zi = cubic(x,y,z,xi,yi);  % 详解: 赋值：将 cubic(...) 的结果保存到 zi
  case 'nearest'  % 详解: 分支：case 'nearest'
    zi = nearest(x,y,z,xi,yi);  % 详解: 赋值：将 nearest(...) 的结果保存到 zi
  case {'invdist','v4'}  % 详解: 分支：case {'invdist','v4'}
    zi = gdatav4(x,y,z,xi,yi);  % 详解: 赋值：将 gdatav4(...) 的结果保存到 zi
  otherwise  % 详解: 默认分支：otherwise
    error('Unknown method.');  % 详解: 调用函数：error('Unknown method.')
end  % 详解: 执行语句
  
if nargout<=1, xi = zi; end  % 详解: 条件判断：if (nargout<=1, xi = zi; end)


function zi = linear(x,y,z,xi,yi)  % 详解: 执行语句


siz = size(xi);  % 详解: 赋值：将 size(...) 的结果保存到 siz
xi = xi(:); yi = yi(:);  % 详解: 赋值：将 xi(...) 的结果保存到 xi
x = x(:); y = y(:);  % 详解: 赋值：将 x(...) 的结果保存到 x

tri = delaunay(x,y,'sorted');  % 详解: 赋值：将 delaunay(...) 的结果保存到 tri
if isempty(tri),  % 详解: 条件判断：if (isempty(tri),)
  warning('Data cannot be triangulated.');  % 详解: 调用函数：warning('Data cannot be triangulated.')
  zi = repmat(NaN,size(xi));  % 详解: 赋值：将 repmat(...) 的结果保存到 zi
  return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

t = tsearch(x,y,tri,xi,yi);  % 详解: 赋值：将 tsearch(...) 的结果保存到 t

out = find(isnan(t));  % 详解: 赋值：将 find(...) 的结果保存到 out
if ~isempty(out), t(out) = ones(size(out)); end  % 详解: 条件判断：if (~isempty(out), t(out) = ones(size(out)); end)
tri = tri(t,:);  % 详解: 赋值：将 tri(...) 的结果保存到 tri

del = (x(tri(:,2))-x(tri(:,1))) .* (y(tri(:,3))-y(tri(:,1))) - ...  % 详解: 赋值：计算表达式并保存到 del
      (x(tri(:,3))-x(tri(:,1))) .* (y(tri(:,2))-y(tri(:,1)));  % 详解: 执行语句
w(:,3) = ((x(tri(:,1))-xi).*(y(tri(:,2))-yi) - ...  % 详解: 执行语句
          (x(tri(:,2))-xi).*(y(tri(:,1))-yi)) ./ del;  % 详解: 执行语句
w(:,2) = ((x(tri(:,3))-xi).*(y(tri(:,1))-yi) - ...  % 详解: 执行语句
          (x(tri(:,1))-xi).*(y(tri(:,3))-yi)) ./ del;  % 详解: 执行语句
w(:,1) = ((x(tri(:,2))-xi).*(y(tri(:,3))-yi) - ...  % 详解: 执行语句
          (x(tri(:,3))-xi).*(y(tri(:,2))-yi)) ./ del;  % 详解: 执行语句
w(out,:) = zeros(length(out),3);  % 详解: 调用函数：w(out,:) = zeros(length(out),3)

z = z(:).'; % Treat z as a row so that code below involving  % 详解: 赋值：将 z(...) 的结果保存到 z  % 详解: 赋值：将 z(...) 的结果保存到 z
zi = sum(z(tri) .* w,2);  % 详解: 赋值：将 sum(...) 的结果保存到 zi

zi = reshape(zi,siz);  % 详解: 赋值：将 reshape(...) 的结果保存到 zi

if ~isempty(out), zi(out) = NaN; end  % 详解: 条件判断：if (~isempty(out), zi(out) = NaN; end)

function zi = cubic(x,y,z,xi,yi)  % 详解: 执行语句


siz = size(xi);  % 详解: 赋值：将 size(...) 的结果保存到 siz
xi = xi(:); yi = yi(:);  % 详解: 赋值：将 xi(...) 的结果保存到 xi
x = x(:); y = y(:); z = z(:);  % 详解: 赋值：将 x(...) 的结果保存到 x

tri = delaunay(x,y,'sorted');  % 详解: 赋值：将 delaunay(...) 的结果保存到 tri
if isempty(tri),  % 详解: 条件判断：if (isempty(tri),)
  warning('Data cannot be triangulated.');  % 详解: 调用函数：warning('Data cannot be triangulated.')
  zi = repmat(NaN,size(xi));  % 详解: 赋值：将 repmat(...) 的结果保存到 zi
  return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

t1 = [x(tri(:,1)) y(tri(:,1)) z(tri(:,1))];  % 详解: 赋值：计算表达式并保存到 t1
t2 = [x(tri(:,2)) y(tri(:,2)) z(tri(:,2))];  % 详解: 赋值：计算表达式并保存到 t2
t3 = [x(tri(:,3)) y(tri(:,3)) z(tri(:,3))];  % 详解: 赋值：计算表达式并保存到 t3
Area = ((x(tri(:,2))-x(tri(:,1))) .* (y(tri(:,3))-y(tri(:,1))) - ...  % 详解: 赋值：计算表达式并保存到 Area
       (x(tri(:,3))-x(tri(:,1))) .* (y(tri(:,2))-y(tri(:,1))))/2;  % 详解: 执行语句
nv = cross((t3-t1).',(t2-t1).').';  % 赋值：设置变量 nv  % 详解: 赋值：将 cross(...) 的结果保存到 nv  % 详解: 赋值：将 cross(...) 的结果保存到 nv

nv = nv ./ repmat(nv(:,3),1,3);  % 详解: 赋值：计算表达式并保存到 nv

m = size(tri,1);  % 详解: 赋值：将 size(...) 的结果保存到 m
n = length(x);  % 详解: 赋值：将 length(...) 的结果保存到 n
i = repmat((1:m)',1,3);  % 赋值：设置变量 i  % 详解: 赋值：将 repmat(...) 的结果保存到 i  % 详解: 赋值：将 repmat(...) 的结果保存到 i
T = sparse(i,tri,repmat(-nv(1:m,1).*Area,1,3),m,n);  % 详解: 赋值：将 sparse(...) 的结果保存到 T
A = sparse(i,tri,repmat(Area,1,3),m,n);  % 详解: 赋值：将 sparse(...) 的结果保存到 A
s = full(sum(A));  % 详解: 赋值：将 full(...) 的结果保存到 s
gx = (full(sum(T))./(s + (s==0)))';  % 统计：求和/均值/中位数  % 详解: 赋值：计算表达式并保存到 gx  % 详解: 赋值：计算表达式并保存到 gx
T = sparse(i,tri,repmat(-nv(1:m,2).*Area,1,3),m,n);  % 详解: 赋值：将 sparse(...) 的结果保存到 T
gy = (full(sum(T))./(s + (s==0)))';  % 统计：求和/均值/中位数  % 详解: 赋值：计算表达式并保存到 gy  % 详解: 赋值：计算表达式并保存到 gy

i1 = [1 2 3]; i2 = [2 3 1]; i3 = [3 1 2];  % 详解: 赋值：计算表达式并保存到 i1
xx = x(tri);  % 详解: 赋值：将 x(...) 的结果保存到 xx
yy = y(tri);  % 详解: 赋值：将 y(...) 的结果保存到 yy
zz = z(tri);  % 详解: 赋值：将 z(...) 的结果保存到 zz
gx = gx(tri);  % 详解: 赋值：将 gx(...) 的结果保存到 gx
gy = gy(tri);  % 详解: 赋值：将 gy(...) 的结果保存到 gy
len = sqrt((xx(:,i3)-xx(:,i2)).^2 + (yy(:,i3)-yy(:,i2)).^2);  % 详解: 赋值：将 sqrt(...) 的结果保存到 len

gn = ((gx(:,i2)+gx(:,i3)).*(yy(:,i2)-yy(:,i3)) - ...  % 详解: 赋值：计算表达式并保存到 gn
      (gy(:,i2)+gy(:,i3)).*(xx(:,i2)-xx(:,i3)))/2./len;  % 详解: 执行语句

Area = repmat(Area,1,3);  % 详解: 赋值：将 repmat(...) 的结果保存到 Area
Wna = 1/4*(-2*yy(:,i2).*yy(:,i3)+yy(:,i2).^2+yy(:,i3).^2+xx(:,i2).^2 - ...  % 详解: 赋值：计算表达式并保存到 Wna
             2*xx(:,i2).*xx(:,i3)+xx(:,i3).^2).*zz;  % 详解: 执行语句
Wna(:) = Wna-1/16.*(yy(:,i2).^2-2*yy(:,i2).*yy(:,i3)+yy(:,i3).^2+xx(:,i2).^2- ...  % 详解: 执行语句
         2.*xx(:,i2).*xx(:,i3)+xx(:,i3).^2).*(-xx(:,i2)+2.*xx(:,i1)-xx(:,i3)).*gx;  % 详解: 执行语句
Wna(:) = Wna-1/16.*(yy(:,i2).^2-2.*yy(:,i2).*yy(:,i3)+yy(:,i3).^2+xx(:,i2).^2- ...  % 详解: 执行语句
         2.*xx(:,i2).*xx(:,i3)+xx(:,i3).^2).*(-yy(:,i2)+2.*yy(:,i1)-yy(:,i3)).*gy;  % 详解: 执行语句
Wna(:) = Wna./Area./len(:,i1);  % 详解: 调用函数：Wna(:) = Wna./Area./len(:,i1)

Wnb = 1/4*(yy(:,i1).^2+yy(:,i1).*yy(:,i3)-3.*yy(:,i2).*yy(:,i1)+ ...  % 详解: 赋值：计算表达式并保存到 Wnb
        3.*yy(:,i2).*yy(:,i3)-2.*yy(:,i3).^2+xx(:,i1).^2+xx(:,i1).*xx(:,i3)- ...  % 详解: 执行语句
        3.*xx(:,i2).*xx(:,i1)+3.*xx(:,i2).*xx(:,i3)-2.*xx(:,i3).^2).*zz;  % 详解: 执行语句
Wnb(:) = Wnb-1/16*(6*yy(:,i1).*xx(:,i2).*yy(:,i3)-3*yy(:,i1).^2.*xx(:,i2)- ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i1).*yy(:,i3)+2*yy(:,i1).^2.*xx(:,i1)- ...  % 详解: 执行语句
         4*yy(:,i1).*xx(:,i3).*yy(:,i3)+yy(:,i1).^2.*xx(:,i3)- ...  % 详解: 执行语句
         2*yy(:,i2).*xx(:,i3).*yy(:,i3)+2*yy(:,i2).*xx(:,i3).*yy(:,i1)+ ...  % 详解: 执行语句
         2*yy(:,i2).*xx(:,i1).*yy(:,i3)-2*yy(:,i2).*xx(:,i1).*yy(:,i1)+ ...  % 详解: 执行语句
         3*yy(:,i3).^2.*xx(:,i3)-3*yy(:,i3).^2.*xx(:,i2)-xx(:,i1).^2.*xx(:,i3)+ ...  % 详解: 执行语句
         2*xx(:,i1).^3+10*xx(:,i2).*xx(:,i1).*xx(:,i3)-5*xx(:,i2).*xx(:,i1).^2- ...  % 详解: 执行语句
         4*xx(:,i1).*xx(:,i3).^2-5*xx(:,i3).^2.*xx(:,i2)+3*xx(:,i3).^3).*gx;  % 详解: 执行语句
Wnb(:) = Wnb-1/16*(-yy(:,i1).^2.*yy(:,i3)+2*yy(:,i1).^3+ ...  % 详解: 执行语句
         10*yy(:,i2).*yy(:,i1).*yy(:,i3)-5*yy(:,i2).*yy(:,i1).^2- ...  % 详解: 执行语句
         4*yy(:,i1).*yy(:,i3).^2-5*yy(:,i3).^2.*yy(:,i2)+3*yy(:,i3).^3+ ...  % 详解: 执行语句
         6*yy(:,i2).*xx(:,i1).*xx(:,i3)-3*yy(:,i2).*xx(:,i1).^2- ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i1).*xx(:,i3)+2*yy(:,i1).*xx(:,i1).^2- ...  % 详解: 执行语句
         4*yy(:,i3).*xx(:,i1).*xx(:,i3)+yy(:,i3).*xx(:,i1).^2- ...  % 详解: 执行语句
         2*yy(:,i3).*xx(:,i2).*xx(:,i3)+2*yy(:,i3).*xx(:,i2).*xx(:,i1)+ ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i2).*xx(:,i3)-2*yy(:,i1).*xx(:,i2).*xx(:,i1)+ ...  % 详解: 执行语句
         3*xx(:,i3).^2.*yy(:,i3)-3*yy(:,i2).*xx(:,i3).^2).*gy;  % 详解: 执行语句
Wnb(:) = Wnb./Area./len(:,i2);  % 详解: 调用函数：Wnb(:) = Wnb./Area./len(:,i2)

Wnc = 1/4*(yy(:,i2).*yy(:,i1)+yy(:,i1).^2-2*yy(:,i2).^2+3*yy(:,i2).*yy(:,i3)- ...  % 详解: 赋值：计算表达式并保存到 Wnc
         3.*yy(:,i1).*yy(:,i3)+xx(:,i2).*xx(:,i1)+xx(:,i1).^2-2.*xx(:,i2).^2+ ...  % 详解: 执行语句
         3.*xx(:,i2).*xx(:,i3)-3.*xx(:,i1).*xx(:,i3)).*zz;  % 详解: 执行语句
Wnc(:) = Wnc-1/16*(yy(:,i1).^2.*xx(:,i2)-4*yy(:,i1).*xx(:,i2).*yy(:,i2)+ ...  % 详解: 执行语句
         2*yy(:,i1).^2.*xx(:,i1)-2*yy(:,i2).*xx(:,i1).*yy(:,i1)- ...  % 详解: 执行语句
         3*yy(:,i1).^2.*xx(:,i3)+6*yy(:,i2).*xx(:,i3).*yy(:,i1)- ...  % 详解: 执行语句
         3*yy(:,i2).^2.*xx(:,i3)+3*yy(:,i2).^2.*xx(:,i2)+ ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i2).*yy(:,i3)-2*yy(:,i2).*xx(:,i2).*yy(:,i3)- ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i1).*yy(:,i3)+2*yy(:,i2).*xx(:,i1).*yy(:,i3)+ ...  % 详解: 执行语句
         2*xx(:,i1).^3-xx(:,i2).*xx(:,i1).^2-4*xx(:,i2).^2.*xx(:,i1)- ...  % 详解: 执行语句
         5*xx(:,i1).^2.*xx(:,i3)+10*xx(:,i2).*xx(:,i1).*xx(:,i3)+ ...  % 详解: 执行语句
         3*xx(:,i2).^3-5*xx(:,i2).^2.*xx(:,i3)).*gx;  % 详解: 执行语句
Wnc(:) = Wnc-1/16*(2*yy(:,i1).^3-yy(:,i2).*yy(:,i1).^2- ...  % 详解: 执行语句
         4*yy(:,i2).^2.*yy(:,i1)-5*yy(:,i1).^2.*yy(:,i3)+ ...  % 详解: 执行语句
         10*yy(:,i2).*yy(:,i1).*yy(:,i3)+3*yy(:,i2).^3- ...  % 详解: 执行语句
         5*yy(:,i2).^2.*yy(:,i3)+yy(:,i2).*xx(:,i1).^2- ...  % 详解: 执行语句
         4*yy(:,i2).*xx(:,i1).*xx(:,i2)+2*yy(:,i1).*xx(:,i1).^2- ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i2).*xx(:,i1)-3*yy(:,i3).*xx(:,i1).^2+ ...  % 详解: 执行语句
         6*yy(:,i3).*xx(:,i2).*xx(:,i1)-3*yy(:,i3).*xx(:,i2).^2+ ...  % 详解: 执行语句
         3*yy(:,i2).*xx(:,i2).^2+2*yy(:,i2).*xx(:,i1).*xx(:,i3)- ...  % 详解: 执行语句
         2*yy(:,i2).*xx(:,i2).*xx(:,i3)-2*yy(:,i1).*xx(:,i1).*xx(:,i3)+ ...  % 详解: 执行语句
         2*yy(:,i1).*xx(:,i2).*xx(:,i3)).*gy;  % 详解: 执行语句
Wnc(:) = Wnc./Area./len(:,i3);  % 详解: 调用函数：Wnc(:) = Wnc./Area./len(:,i3)

Wn = Wna(:,[1 2 3]) + Wnb(:,[3 1 2]) + Wnc(:,[2 3 1]);  % 详解: 赋值：将 Wna(...) 的结果保存到 Wn

t = tsearch(x,y,tri,xi,yi);  % 详解: 赋值：将 tsearch(...) 的结果保存到 t

out = find(isnan(t));  % 详解: 赋值：将 find(...) 的结果保存到 out
if ~isempty(out), t(out) = ones(size(out)); end  % 详解: 条件判断：if (~isempty(out), t(out) = ones(size(out)); end)
tri = tri(t,:);  % 详解: 赋值：将 tri(...) 的结果保存到 tri
Area = Area(t,:);  % 详解: 赋值：将 Area(...) 的结果保存到 Area
len = len(t,:);  % 详解: 赋值：将 len(...) 的结果保存到 len
xx = xx(t,:);  % 详解: 赋值：将 xx(...) 的结果保存到 xx
yy = yy(t,:);  % 详解: 赋值：将 yy(...) 的结果保存到 yy
zz = zz(t,:);  % 详解: 赋值：将 zz(...) 的结果保存到 zz
gx = gx(t,:);  % 详解: 赋值：将 gx(...) 的结果保存到 gx
gy = gy(t,:);  % 详解: 赋值：将 gy(...) 的结果保存到 gy
gn = gn(t,:);  % 详解: 赋值：将 gn(...) 的结果保存到 gn
Wn = Wn(t,:);  % 详解: 赋值：将 Wn(...) 的结果保存到 Wn

w = 1/2.*((xx(:,i2)-repmat(xi,1,3)).*(yy(:,i3)-repmat(yi,1,3)) - ...  % 详解: 赋值：计算表达式并保存到 w
          (xx(:,i3)-repmat(xi,1,3)).*(yy(:,i2)-repmat(yi,1,3)))./Area;  % 详解: 执行语句
w(out,:) = ones(length(out),3);  % 详解: 调用函数：w(out,:) = ones(length(out),3)

N1 = w(:,i1) + w(:,i1).^2.*w(:,i2) + w(:,i1).^2.*w(:,i3) - ...  % 详解: 赋值：将 w(...) 的结果保存到 N1
               w(:,i1).*w(:,i2).^2 - w(:,i1).*w(:,i3).^2;  % 详解: 执行语句
N2 = (xx(:,i2)-xx(:,i1)).*(w(:,i1).^2.*w(:,i2)+1/2.*w(:,i1).*w(:,i2).*w(:,i3))+ ...  % 详解: 赋值：计算表达式并保存到 N2
     (xx(:,i3)-xx(:,i1)).*(w(:,i1).^2.*w(:,i3)+1/2.*w(:,i1).*w(:,i2).*w(:,i3));  % 详解: 执行语句
N3 = (yy(:,i2)-yy(:,i1)).*(w(:,i1).^2.*w(:,i2)+1/2.*w(:,i1).*w(:,i2).*w(:,i3))+ ...  % 详解: 赋值：计算表达式并保存到 N3
     (yy(:,i3)-yy(:,i1)).*(w(:,i1).^2.*w(:,i3)+1/2.*w(:,i1).*w(:,i2).*w(:,i3));  % 详解: 执行语句
N1(out) = zeros(size(out));  % 详解: 调用函数：N1(out) = zeros(size(out))
N2(out) = zeros(size(out));  % 详解: 调用函数：N2(out) = zeros(size(out))
N3(out) = zeros(size(out));  % 详解: 调用函数：N3(out) = zeros(size(out))

M = 8*Area./len.*w(:,i1).*w(:,i2).^2.*w(:,i3).^2 ./ ...  % 详解: 赋值：计算表达式并保存到 M
                (w(:,i1)+w(:,i2)+(w(:,i1)+w(:,i2)==0)) ./ ...  % 详解: 执行语句
                (w(:,i1)+w(:,i3)+(w(:,i1)+w(:,i3)==0));  % 详解: 执行语句
M(out,:) = zeros(length(out),3);  % 详解: 调用函数：M(out,:) = zeros(length(out),3)

zi = sum((N1.*zz + N2.*gx + N3.*gy + M.*(gn - Wn)).').';  % 详解: 赋值：将 sum(...) 的结果保存到 zi

zi = reshape(zi,siz);  % 详解: 赋值：将 reshape(...) 的结果保存到 zi

if ~isempty(out), zi(out) = NaN; end  % 详解: 条件判断：if (~isempty(out), zi(out) = NaN; end)

function zi = nearest(x,y,z,xi,yi)  % 详解: 执行语句


siz = size(xi);  % 详解: 赋值：将 size(...) 的结果保存到 siz
xi = xi(:); yi = yi(:);  % 详解: 赋值：将 xi(...) 的结果保存到 xi
x = x(:); y = y(:); z = z(:);  % 详解: 赋值：将 x(...) 的结果保存到 x

tri = delaunay(x,y,'sorted');  % 详解: 赋值：将 delaunay(...) 的结果保存到 tri
if isempty(tri),  % 详解: 条件判断：if (isempty(tri),)
  warning('Data cannot be triangulated.');  % 详解: 调用函数：warning('Data cannot be triangulated.')
  zi = repmat(NaN,size(xi));  % 详解: 赋值：将 repmat(...) 的结果保存到 zi
  return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

k = dsearch(x,y,tri,xi,yi);  % 详解: 赋值：将 dsearch(...) 的结果保存到 k

zi = k;  % 详解: 赋值：计算表达式并保存到 zi
d = find(isfinite(k));  % 详解: 赋值：将 find(...) 的结果保存到 d
zi(d) = z(k(d));  % 详解: 调用函数：zi(d) = z(k(d))
zi = reshape(zi,siz);  % 详解: 赋值：将 reshape(...) 的结果保存到 zi


function [xi,yi,zi] = gdatav4(x,y,z,xi,yi)  % 详解: 函数定义：gdatav4(x,y,z,xi,yi), 返回：xi,yi,zi


xy = x(:) + y(:)*sqrt(-1);  % 详解: 赋值：将 x(...) 的结果保存到 xy

d = xy(:,ones(1,length(xy)));  % 详解: 赋值：将 xy(...) 的结果保存到 d
d = abs(d - d.');  % 赋值：设置变量 d  % 详解: 赋值：将 abs(...) 的结果保存到 d  % 详解: 赋值：将 abs(...) 的结果保存到 d
n = size(d,1);  % 详解: 赋值：将 size(...) 的结果保存到 n
d(1:n+1:prod(size(d))) = ones(1,n);  % 详解: 调用函数：d(1:n+1:prod(size(d))) = ones(1,n)

non = find(d == 0);  % 详解: 赋值：将 find(...) 的结果保存到 non
if ~isempty(non),  % 详解: 条件判断：if (~isempty(non),)
  [r,c] = find(d == 0);  % 详解: 执行语句
  k = find(r < c);  % 详解: 赋值：将 find(...) 的结果保存到 k
  r = r(k); c = c(k);  % 详解: 赋值：将 r(...) 的结果保存到 r
  v = (z(r) + z(c))/2;  % 详解: 赋值：计算表达式并保存到 v
  
  rep = find(diff(c)==0);  % 详解: 赋值：将 find(...) 的结果保存到 rep
  if ~isempty(rep),  % 详解: 条件判断：if (~isempty(rep),)
    runs = find(diff(diff(c)==0)==1)+1;  % 详解: 赋值：将 find(...) 的结果保存到 runs
    for i=1:length(runs),  % 详解: for 循环：迭代变量 i 遍历 1:length(runs),
      k = find(c==c(runs(i)));  % 详解: 赋值：将 find(...) 的结果保存到 k
      v(runs(i)) = mean(z([r(k);c(runs(i))]));  % 详解: 调用函数：v(runs(i)) = mean(z([r(k);c(runs(i))]))
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  z(r) = v;  % 详解: 执行语句
  if ~isempty(rep),  % 详解: 条件判断：if (~isempty(rep),)
    z(r(runs)) = v(runs);  % 详解: 调用函数：z(r(runs)) = v(runs)
  end  % 详解: 执行语句

  x(c) = [];  % 详解: 执行语句
  y(c) = [];  % 详解: 执行语句
  z(c) = [];  % 详解: 执行语句
  xy(c,:) = [];  % 详解: 执行语句
  xy(:,c) = [];  % 详解: 执行语句
  d(c,:) = [];  % 详解: 执行语句
  d(:,c) = [];  % 详解: 执行语句
  
  ndp = sort([r;c]);  % 详解: 赋值：将 sort(...) 的结果保存到 ndp
  ndp(find(ndp(1:length(ndp)-1)==ndp(2:length(ndp)))) = [];  % 详解: 获取向量/矩阵尺寸

  warning(sprintf(['Averaged %d non-distinct points.\n' ...  % 详解: 执行语句
       '         Indices are: %s.'],length(ndp),num2str(ndp')))  % 详解: 执行语句  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句

g = (d.^2) .* (log(d)-1);  % 详解: 赋值：计算表达式并保存到 g
g(1:size(d,1)+1:prod(size(d))) = zeros(size(d,1),1);  % 详解: 调用函数：g(1:size(d,1)+1:prod(size(d))) = zeros(size(d,1),1)
weights = g \ z(:);  % 详解: 赋值：计算表达式并保存到 weights

[m,n] = size(xi);  % 详解: 获取向量/矩阵尺寸
zi = zeros(size(xi));  % 详解: 赋值：将 zeros(...) 的结果保存到 zi
jay = sqrt(-1);  % 详解: 赋值：将 sqrt(...) 的结果保存到 jay
xy = xy.';  % 赋值：设置变量 xy  % 详解: 赋值：计算表达式并保存到 xy  % 详解: 赋值：计算表达式并保存到 xy

for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
  for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
    d = abs(xi(i,j)+jay*yi(i,j) - xy);  % 详解: 赋值：将 abs(...) 的结果保存到 d
    mask = find(d == 0);  % 详解: 赋值：将 find(...) 的结果保存到 mask
    if length(mask)>0, d(mask) = ones(length(mask),1); end  % 详解: 条件判断：if (length(mask)>0, d(mask) = ones(length(mask),1); end)
    g = (d.^2) .* (log(d)-1);  % 详解: 赋值：计算表达式并保存到 g
    if length(mask)>0, g(mask) = zeros(length(mask),1); end  % 详解: 条件判断：if (length(mask)>0, g(mask) = zeros(length(mask),1); end)
    zi(i,j) = g * weights;  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

if nargout<=1,  % 详解: 条件判断：if (nargout<=1,)
  xi = zi;  % 详解: 赋值：计算表达式并保存到 xi
end  % 详解: 执行语句





