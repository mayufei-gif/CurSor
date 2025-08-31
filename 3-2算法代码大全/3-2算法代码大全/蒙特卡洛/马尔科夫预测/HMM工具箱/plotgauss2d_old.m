% 文件: plotgauss2d_old.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h=plotgauss2d_old(mu, Sigma, plot_cross)  % 详解: 执行语句

if nargin < 3, plot_cross = 0; end  % 详解: 条件判断：if (nargin < 3, plot_cross = 0; end)
[V,D]=eig(Sigma);  % 详解: 执行语句
lam1 = D(1,1);  % 详解: 赋值：将 D(...) 的结果保存到 lam1
lam2 = D(2,2);  % 详解: 赋值：将 D(...) 的结果保存到 lam2
v1 = V(:,1);  % 详解: 赋值：将 V(...) 的结果保存到 v1
v2 = V(:,2);  % 详解: 赋值：将 V(...) 的结果保存到 v2
if v1(1)==0  % 详解: 条件判断：if (v1(1)==0)
  theta = 0;  % 详解: 赋值：计算表达式并保存到 theta
else  % 详解: 条件判断：else 分支
  theta = atan(v1(2)/v1(1));  % 详解: 赋值：将 atan(...) 的结果保存到 theta
end  % 详解: 执行语句
a = sqrt(lam1);  % 详解: 赋值：将 sqrt(...) 的结果保存到 a
b = sqrt(lam2);  % 详解: 赋值：将 sqrt(...) 的结果保存到 b
h=plot_ellipse(mu(1), mu(2), theta, a,b);  % 详解: 赋值：将 plot_ellipse(...) 的结果保存到 h

if plot_cross  % 详解: 条件判断：if (plot_cross)
  mu = mu(:);  % 详解: 赋值：将 mu(...) 的结果保存到 mu
  held = ishold;  % 详解: 赋值：计算表达式并保存到 held
  hold on  % 详解: 执行语句
  minor1 = mu-a*v1; minor2 = mu+a*v1;  % 详解: 赋值：计算表达式并保存到 minor1
  hminor = line([minor1(1) minor2(1)], [minor1(2) minor2(2)]);  % 详解: 赋值：将 line(...) 的结果保存到 hminor
  
  major1 = mu-b*v2; major2 = mu+b*v2;  % 详解: 赋值：计算表达式并保存到 major1
  hmajor = line([major1(1) major2(1)], [major1(2) major2(2)]);  % 详解: 赋值：将 line(...) 的结果保存到 hmajor
  if ~held  % 详解: 条件判断：if (~held)
    hold off  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句





