% 文件: plotgauss2d.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h=plotgauss2d(mu, Sigma)  % 详解: 执行语句

h = plotcov2(mu, Sigma);  % 详解: 赋值：将 plotcov2(...) 的结果保存到 h
return;  % 详解: 返回：从当前函数返回




function h = plotcov2(mu, Sigma, varargin)  % 详解: 执行语句

if size(Sigma) ~= [2 2], error('Sigma must be a 2 by 2 matrix'); end  % 详解: 条件判断：if (size(Sigma) ~= [2 2], error('Sigma must be a 2 by 2 matrix'); end)
if length(mu) ~= 2, error('mu must be a 2 by 1 vector'); end  % 详解: 条件判断：if (length(mu) ~= 2, error('mu must be a 2 by 1 vector'); end)

[p, ...  % 详解: 执行语句
 n, ...  % 详解: 执行语句
 plot_opts] = process_options(varargin, 'conf', 0.9, ...  % 详解: 执行语句
					'num-pts', 100);  % 详解: 执行语句
h = [];  % 详解: 赋值：计算表达式并保存到 h
holding = ishold;  % 详解: 赋值：计算表达式并保存到 holding
if (Sigma == zeros(2, 2))  % 详解: 条件判断：if ((Sigma == zeros(2, 2)))
  z = mu;  % 详解: 赋值：计算表达式并保存到 z
else  % 详解: 条件判断：else 分支
  k = conf2mahal(p, 2);  % 详解: 赋值：将 conf2mahal(...) 的结果保存到 k
  if (issparse(Sigma))  % 详解: 条件判断：if ((issparse(Sigma)))
    [V, D] = eigs(Sigma);  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    [V, D] = eig(Sigma);  % 详解: 执行语句
  end  % 详解: 执行语句
  t = linspace(0, 2*pi, n);  % 详解: 赋值：将 linspace(...) 的结果保存到 t
  u = [cos(t); sin(t)];  % 详解: 赋值：计算表达式并保存到 u
  w = (k * V * sqrt(D)) * u;  % 详解: 赋值：计算表达式并保存到 w
  z = repmat(mu, [1 n]) + w;  % 详解: 赋值：将 repmat(...) 的结果保存到 z
  L = k * sqrt(diag(D));  % 详解: 赋值：计算表达式并保存到 L
  h = plot([mu(1); mu(1) + L(1) * V(1, 1)], ...  % 详解: 赋值：将 plot(...) 的结果保存到 h
	   [mu(2); mu(2) + L(1) * V(2, 1)], plot_opts{:});  % 详解: 执行语句
  hold on;  % 详解: 执行语句
  h = [h; plot([mu(1); mu(1) + L(2) * V(1, 2)], ...  % 详解: 赋值：计算表达式并保存到 h
	       [mu(2); mu(2) + L(2) * V(2, 2)], plot_opts{:})];  % 详解: 执行语句
end  % 详解: 执行语句

h = [h; plot(z(1, :), z(2, :), plot_opts{:})];  % 详解: 赋值：计算表达式并保存到 h
if (~holding) hold off; end  % 详解: 条件判断：if ((~holding) hold off; end)




function m = conf2mahal(c, d)  % 详解: 执行语句

m = chi2inv(c, d);  % 详解: 赋值：将 chi2inv(...) 的结果保存到 m




