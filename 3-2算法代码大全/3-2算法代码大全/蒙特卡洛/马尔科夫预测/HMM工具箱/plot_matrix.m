% 文件: plot_matrix.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function plot_matrix(G, bw)  % 详解: 函数定义：plot_matrix(G, bw)

if nargin < 2, bw = 0; end  % 详解: 条件判断：if (nargin < 2, bw = 0; end)

if 0  % 详解: 条件判断：if (0)
  imagesc(G)  % 详解: 调用函数：imagesc(G)
  grid on  % 详解: 执行语句
  n = length(G);  % 详解: 赋值：将 length(...) 的结果保存到 n
  
  set(gca,'xtick',1.5:1:n);  % 详解: 调用函数：set(gca,'xtick',1.5:1:n)
  set(gca,'ytick',1.5:1:n);  % 详解: 调用函数：set(gca,'ytick',1.5:1:n)
  
else  % 详解: 条件判断：else 分支
  imagesc(G);  % 详解: 调用函数：imagesc(G)
  if bw  % 详解: 条件判断：if (bw)
    colormap([1 1 1; 0 0 0]);  % 详解: 调用函数：colormap([1 1 1; 0 0 0])
  end  % 详解: 执行语句
  n = length(G);  % 详解: 赋值：将 length(...) 的结果保存到 n
  x = 1.5:1:n;  % 详解: 赋值：计算表达式并保存到 x
  x = [ x; x; repmat(nan,1,n-1) ];  % 详解: 赋值：计算表达式并保存到 x
  y = [ 0.5 n+0.5 nan ].';  % 赋值：设置变量 y  % 详解: 赋值：计算表达式并保存到 y  % 详解: 赋值：计算表达式并保存到 y
  y = repmat(y,1,n-1);  % 详解: 赋值：将 repmat(...) 的结果保存到 y
  x = x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x
  y = y(:);  % 详解: 赋值：将 y(...) 的结果保存到 y
  line(x,y,'linestyle',':','color','k');  % 详解: 调用函数：line(x,y,'linestyle',':','color','k')
  line(y,x,'linestyle',':','color','k');  % 详解: 调用函数：line(y,x,'linestyle',':','color','k')
  set(gca,'xtick',1:n)  % 详解: 调用函数：set(gca,'xtick',1:n)
  set(gca,'ytick',1:n)  % 详解: 调用函数：set(gca,'ytick',1:n)
end  % 详解: 执行语句






