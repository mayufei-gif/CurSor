% 文件: QUAD2DG.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [int, tol1,k] = quad2dg(fun,xlow,xhigh,ylow,yhigh,tol,p1,p2,p3,p4,p5,p6,p7,p8,p9)  % 详解: 函数定义：quad2dg(fun,xlow,xhigh,ylow,yhigh,tol,p1,p2,p3,p4,p5,p6,p7,p8,p9), 返回：int, tol1,k

global bpx2 bpy2 wfxy2;  % 详解: 执行语句


if isempty(bpx2)| isempty(bpy2)| isempty(wfxy2)  % 详解: 条件判断：if (isempty(bpx2)| isempty(bpy2)| isempty(wfxy2))
  [bpx2,bpy2,wfxy2] = grule2d(2,2);  % 详解: 执行语句
end  % 详解: 执行语句


if exist('tol')~=1,  % 详解: 条件判断：if (exist('tol')~=1,)
  tol=1e-3;  % 详解: 赋值：计算表达式并保存到 tol
elseif isempty(tol),  % 详解: 条件判断：elseif (isempty(tol),)
  tol=1e-3;  % 详解: 赋值：计算表达式并保存到 tol
end  % 详解: 执行语句
[errorcode ,xlow,xhigh,ylow,yhigh]=distchck(4,xlow,xhigh,ylow,yhigh);  % 详解: 执行语句
if errorcode > 0  % 详解: 条件判断：if (errorcode > 0)
  error('Requires non-scalar arguments to match in size.');  % 详解: 调用函数：error('Requires non-scalar arguments to match in size.')
end  % 详解: 执行语句


[N M]=size(xlow);  % 详解: 获取向量/矩阵尺寸
xlow=xlow(:); xhigh=xhigh(:);ylow=ylow(:);yhigh=yhigh(:);  % 详解: 赋值：将 xlow(...) 的结果保存到 xlow
nk=N*M;  % 详解: 赋值：计算表达式并保存到 nk


qx=(xhigh-xlow)/2;  % 详解: 赋值：计算表达式并保存到 qx
px=(xhigh+xlow)/2;  % 详解: 赋值：计算表达式并保存到 px
x=permute(qx(:,ones(1,2), ones(1,2) ),[ 2 3 1]).*bpx2(:,:,ones(1,nk)) ...  % 详解: 赋值：将 permute(...) 的结果保存到 x
    +permute(px(:,ones(1,2), ones(1,2) ),[2 3 1]);  % 详解: 创建全 1 矩阵/数组
qy=(yhigh-ylow)/2;  % 详解: 赋值：计算表达式并保存到 qy
py=(yhigh+ylow)/2;  % 详解: 赋值：计算表达式并保存到 py
y=permute(qy(:,ones(1,2), ones(1,2) ),[ 2 3 1]).*bpy2(:,:,ones(1,nk)) ...  % 详解: 赋值：将 permute(...) 的结果保存到 y
    +permute(py(:,ones(1,2), ones(1,2) ),[2 3 1]);  % 详解: 创建全 1 矩阵/数组


if any(fun=='(') & any(fun=='y')& any(fun=='x'),  % 详解: 条件判断：if (any(fun=='(') & any(fun=='y')& any(fun=='x'),)
  exec_string=['fv=', fun, ';'];  % 详解: 赋值：计算表达式并保存到 exec_string
else  % 详解: 条件判断：else 分支
  exec_string=['fv=', fun, ' (x,y'];  % 详解: 赋值：计算表达式并保存到 exec_string
  num_parameters=nargin-6;  % 详解: 赋值：计算表达式并保存到 num_parameters
  for i=1:num_parameters,  % 详解: for 循环：迭代变量 i 遍历 1:num_parameters,
    exec_string=[exec_string,',p',int2str(i)];  % 详解: 赋值：计算表达式并保存到 exec_string
  end  % 详解: 执行语句
  exec_string=[exec_string,');'];  % 详解: 赋值：计算表达式并保存到 exec_string
end  % 详解: 执行语句
eval(exec_string);  % 详解: 调用函数：eval(exec_string)
int_old = squeeze(sum(sum(wfxy2(:,:,ones(1,nk)).*fv))).*qx.*qy;  % 详解: 赋值：将 squeeze(...) 的结果保存到 int_old
int=zeros(size(int_old));tol1=int;  % 详解: 赋值：将 zeros(...) 的结果保存到 int
k=1:nk;  % 详解: 赋值：计算表达式并保存到 k




converge='n';  % 详解: 赋值：计算表达式并保存到 converge
for i=1:8,  % 详解: for 循环：迭代变量 i 遍历 1:8,
  gnum=int2str(2^(i+1));  % 详解: 赋值：将 int2str(...) 的结果保存到 gnum
  eval(['global bpx',gnum,' wfxy',gnum,'  bpy',gnum,';']);  % 详解: 调用函数：eval(['global bpx',gnum,' wfxy',gnum,' bpy',gnum,';'])
  if isempty(eval(['bpx',gnum]))| isempty(eval(['bpy',gnum])) ,  % 详解: 条件判断：if (isempty(eval(['bpx',gnum]))| isempty(eval(['bpy',gnum])) ,)
    eval(['[bpx',gnum,',bpy',gnum,',wfxy',gnum,']=grule2d(',gnum,',', gnum,');']);  % 详解: 调用函数：eval(['[bpx',gnum,',bpy',gnum,',wfxy',gnum,']=grule2d(',gnum,',', gnum,');'])
  end  % 详解: 执行语句
  eval(['x=permute(qx(k,ones(1,',gnum,'), ones(1,',gnum,') ),[ 2 3 1]).*bpx',gnum,'(:,:,ones(1,nk)) +permute(px(k,ones(1,',gnum,'), ones(1,',gnum,') ),[2 3 1]);']);  % 详解: 调用函数：eval(['x=permute(qx(k,ones(1,',gnum,'), ones(1,',gnum,') ),[ 2 3 1]).*bpx',gnum,'(:,:,ones(1,nk)) +permute(px(k,ones(1,',gnum,'), ones(1,',gnum,') ),[2 3 1]);'])


  eval(['y=permute(qy(k,ones(1,',gnum,'), ones(1,',gnum,') ),[ 2 3 1]).*bpy',gnum,'(:,:,ones(1,nk)) +permute(py(k,ones(1,',gnum,'), ones(1,',gnum,') ),[2 3 1]);']);  % 详解: 调用函数：eval(['y=permute(qy(k,ones(1,',gnum,'), ones(1,',gnum,') ),[ 2 3 1]).*bpy',gnum,'(:,:,ones(1,nk)) +permute(py(k,ones(1,',gnum,'), ones(1,',gnum,') ),[2 3 1]);'])
    eval(exec_string);  % 详解: 调用函数：eval(exec_string)
    eval(['int(k) = squeeze(sum(sum((wfxy',gnum,'(:,:,ones(1,nk)).*fv)))).*qx(k).*qy(k);']);  % 详解: 调用函数：eval(['int(k) = squeeze(sum(sum((wfxy',gnum,'(:,:,ones(1,nk)).*fv)))).*qx(k).*qy(k);'])
    tol1(k)=abs(int_old(k)-int(k));  % 详解: 调用函数：tol1(k)=abs(int_old(k)-int(k))
    k=find(tol1 > abs(tol*int)|tol1 > abs(tol));  % 详解: 赋值：将 find(...) 的结果保存到 k
  
    if any(k),  % 详解: 条件判断：if (any(k),)
      nk=length(k);  % 详解: 赋值：将 length(...) 的结果保存到 nk
    else  % 详解: 条件判断：else 分支
      converge='y';  % 详解: 赋值：计算表达式并保存到 converge
      break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
    
     int_old(k)=int(k);  % 详解: 调用函数：int_old(k)=int(k)
end  % 详解: 执行语句
int=reshape(int,[N M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 int
if converge=='n',  % 详解: 条件判断：if (converge=='n',)
  disp('Integral did not converge--singularity likely')  % 详解: 调用函数：disp('Integral did not converge--singularity likely')
end  % 详解: 执行语句






