% 文件: plotcov2.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% PLOTCOV2 - Plots a covariance ellipse with major and minor axes
%            for a bivariate Gaussian distribution.  % 中文: 用于双变量高斯分布。
%
% Usage:  % 中文: CWM.M
%   h = plotcov2(mu, Sigma[, OPTIONS]);  % 中文: h = plotcov2（Mu，sigma [，options]）;
% 
% Inputs:
%   mu    - a 2 x 1 vector giving the mean of the distribution.  % 中文: MU -A 2 x 1矢量给出了分布的平均值。
%   Sigma - a 2 x 2 symmetric positive semi-definite matrix giving  % 中文: Sigma -A 2 x 2对称阳性半明确矩阵给予|||分布的协方差（或零矩阵）。 |||选项：||| 'conf' -  0和1之间的标量给予置信度|||间隔（即，椭圆形概率质量质量的比例|||）；默认值为0.9。
%           the covariance of the distribution (or the zero matrix).  % 中文: “数字”  - 用于绘制|||的点数椭圆;默认值为100。|||此功能还接受情节的选项。
%
% Options:  % 中文: H-图形手柄向椭圆边界的向量和|||它的主要和次要轴|||另请参阅：PlotCov3 |||计算椭圆形的Mahalanobis半径|||所需的概率质量。 |||协方差椭圆的主要和次级轴是由|||给出的协方差矩阵的特征向量。  它们的长度（|||带有单位Mahalanobis Radius的椭圆形）由|||给出相应特征值的平方根。
%   'conf'    - a scalar between 0 and 1 giving the confidence
%               interval (i.e., the fraction of probability mass to
%               be enclosed by the ellipse); default is 0.9.
%   'num-pts' - the number of points to be used to plot the
%               ellipse; default is 100.
%
% This function also accepts options for PLOT.
%
% Outputs:
%   h     - a vector of figure handles to the ellipse boundary and
%           its major and minor axes
%
% See also: PLOTCOV3

% Copyright (C) 2002 Mark A. Paskin
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by  % 中文: 根据|||发布的GNU通用公共许可条款的条款自由软件基金会；许可证的第2版，或||| （您可以选择）任何以后的版本。 |||该程序的分布是希望它将有用的，但是|||没有任何保修；甚至没有|||的隐含保证适合或适合特定目的的健身。  请参阅gnu |||通用公共许可证以获取更多详细信息。 |||您应该已经收到了GNU通用公共许可证的副本|||以及这个程序；如果没有，请写入免费软件||| Foundation，Inc。，59 Temple Place，Suite 330，马萨诸塞州波士顿02111-1307 |||美国。 |||琐碎的情况：|||首先，通过简单的最佳匹配来减少问题。  如果两个|||元素同意它们是最好的匹配，然后将它们匹配。 |||获取两组的（新）大小，u和v。||| mx = realmax; |||将亲和力矩阵为正方形|||运行匈牙利方法。  首先用|||替换无限值最大（或最小）有限值。
% the Free Software Foundation; either version 2 of the License, or  % 中文: fprintf（'跑步匈牙利\ n'）;
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
% USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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




