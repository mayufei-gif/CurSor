% 文件: nonmaxsup.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% NONMAXSUP - Non-maximal Suppression
%
% Usage:  cim = nonmaxsup(im, radius)
%
% Arguments:   
%            im     - image to be processed.
%            radius - radius of region considered in non-maximal
%                     suppression (optional). Typical values to use might
%                     be 1-3.  Default is 1.
%
% Returns:
%            cim    - image with pixels that are not maximal within a 
%                     square neighborhood zeroed out.

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

function cim = nonmaxsup(m, radius)  % 详解: 执行语句
  if (nargin == 1) radius = 1; end  % 详解: 条件判断：if ((nargin == 1) radius = 1; end)
  sze = 2 * radius + 1;  % 详解: 赋值：计算表达式并保存到 sze
  mx = ordfilt2(m, sze^2, ones(sze));  % 详解: 赋值：将 ordfilt2(...) 的结果保存到 mx
  cim = sparse(m .* (m == mx));  % 详解: 赋值：将 sparse(...) 的结果保存到 cim






