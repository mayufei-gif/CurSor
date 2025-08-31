% 文件: conf2mahal.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% CONF2MAHAL - Translates a confidence interval to a Mahalanobis  % 中文: a particular confidence interval uniquely determines
%              distance.  Consider a multivariate Gaussian  % 中文: an ellipsoid with a fixed Mahalanobis distance.
%              distribution of the form  % 中文: If X is an d dimensional Gaussian-distributed vector, |||。
%
%   p(x) = 1/sqrt((2 * pi)^d * det(C)) * exp((-1/2) * MD(x, m, inv(C)))
%
%              where MD(x, m, P) is the Mahalanobis distance from x
%              to m under P:
%
%                 MD(x, m, P) = (x - m) * P * (x - m)'
%
%              A particular Mahalanobis distance k identifies an
%              ellipsoid centered at the mean of the distribution.
%              The confidence interval associated with this ellipsoid
%              is the probability mass enclosed by it.  Similarly,
%              a particular confidence interval uniquely determines
%              an ellipsoid with a fixed Mahalanobis distance.
%
%              If X is an d dimensional Gaussian-distributed vector,
%              then the Mahalanobis distance of X is distributed
%              according to the Chi-squared distribution with d  % 中文: 根据d |||的卡方分布自由度。  因此，Mahalanobis距离为|||通过评估反累积||确定卡平方分布的分布函数|||达到置信价值。 |||用法：||| m = conf2mahal（c，d）;
%              degrees of freedom.  Thus, the Mahalanobis distance is  % 中文: C-置信区间||| D-高斯分布的尺寸|||输出：||| M-椭圆形的Mahalanobis半径封闭|||分布的概率质量的分数C |||另请参阅：Mahal2Conf
%              determined by evaluating the inverse cumulative  % 中文: labels01 =（labelspm+1）/2; ％地图-1-> 0， +1-> 1
%              distribution function of the chi squared distribution  % 中文: labelspm =（2*labels01）-1; ％地图0,1-> -1,1
%              up to the confidence value.  % 中文: cross_entropy计算两个离散概率之间的kullback-leibler差异。发行||| kl = cross_entropy（p，q，对称）|||如果对称= 1，我们计算对称版本。默认值：对称= 0; |||将我的代码与|||进行比较http://www.media.mit.edu/physics/publications/books/nmm/files/index.html
%
% Usage:  % 中文: CWM.M
% 
%   m = conf2mahal(c, d);  % 中文: （c）Neil Gershenfeld 9/1/97
%
% Inputs:
%
%   c    - the confidence interval  % 中文: 1D集群加权建模示例
%   d    - the number of dimensions of the Gaussian distribution
%
% Outputs:
%
%   m    - the Mahalanobis radius of the ellipsoid enclosing the
%          fraction c of the distribution's probability mass
%
% See also: MAHAL2CONF

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
function m = conf2mahal(c, d)  % 详解: 执行语句

m = chi2inv(c, d);  % 详解: 赋值：将 chi2inv(...) 的结果保存到 m



