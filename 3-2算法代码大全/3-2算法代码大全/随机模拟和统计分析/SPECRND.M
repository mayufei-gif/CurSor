% 文件: SPECRND.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function r=specrnd(x,p,m,n)  % 详解: 执行语句
    

if nargin<4, n=1;end  % 详解: 条件判断：if (nargin<4, n=1;end)
if nargin<3, m=1;end  % 详解: 条件判断：if (nargin<3, m=1;end)
if nargin<2|isempty(p), p=1/length(x)*ones(size(x)); end;  % 详解: 条件判断：if (nargin<2|isempty(p), p=1/length(x)*ones(size(x)); end;)

[mx,nx]=size(x);[mp,np]=size(p);  % 详解: 获取向量/矩阵尺寸
sx=min(mx,nx);sp=min(mp,np);  % 详解: 赋值：将 min(...) 的结果保存到 sx
if (mx~=mp)|(nx~=np),  % 详解: 条件判断：if ((mx~=mp)|(nx~=np),)
   error('size must agree');  % 详解: 调用函数：error('size must agree')
end;  % 详解: 执行语句
if (sx~=1)&(sx~=1),error('should be vector'); end;  % 详解: 条件判断：if ((sx~=1)&(sx~=1),error('should be vector'); end;)
x=x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x
if any(x~=sort(x)), error('x should be acsend'); end;  % 详解: 条件判断：if (any(x~=sort(x)), error('x should be acsend'); end;)
if any((p<0)|(p>1)),error('p should be between 0 and 1'); end;  % 详解: 条件判断：if (any((p<0)|(p>1)),error('p should be between 0 and 1'); end;)
if abs(sum(p)-1)>1e-3,error('sum of p should be 1'); end;  % 详解: 条件判断：if (abs(sum(p)-1)>1e-3,error('sum of p should be 1'); end;)

r=zeros(m,n);lx=max(mx,nx);  % 详解: 赋值：将 zeros(...) 的结果保存到 r
s=rand(m,n);  % 详解: 赋值：将 rand(...) 的结果保存到 s
cp=cumsum(p);  % 详解: 赋值：将 cumsum(...) 的结果保存到 cp
k=find(s<cp(1));r(k)=x(1);  % 详解: 赋值：将 find(...) 的结果保存到 k
k=find(s>=cp(lx-1));r(k)=x(lx);  % 详解: 赋值：将 find(...) 的结果保存到 k
for i=2:(lx-1),  % 详解: for 循环：迭代变量 i 遍历 2:(lx-1),
   k=find((s>=cp(i-1))&(s<cp(i)));  % 详解: 赋值：将 find(...) 的结果保存到 k
   r(k)=x(i);  % 详解: 调用函数：r(k)=x(i)
end  % 详解: 执行语句





