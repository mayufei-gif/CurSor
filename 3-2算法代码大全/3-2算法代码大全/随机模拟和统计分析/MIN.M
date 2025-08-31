% 文件: MIN.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%列状数据最小值
%例如
%    A=[11 4 0.2;22 3 0.5;0 3 0.4];
%    min(A)
%
%MIN    Smallest component.
%   For vectors, MIN(X) is the smallest element in X. For matrices,
%   MIN(X) is a row vector containing the minimum element from each
%   column. For N-D arrays, MIN(X) operates along the first
%   non-singleton dimension.  % 中文: [y，i] = max（x）返回向量I中最大值的索引。|||如果沿第一个非单星维度的值包含更多|||比一个最大元素，第一个索引返回。
%
%   [Y,I] = MIN(X) returns the indices of the minimum values in vector I.
%   If the values along the first non-singleton dimension contain more  % 中文: [y，i] = max（x，[]，dim）沿dimension dim。 |||当复合物时，使用最大幅度（ABS（x））。  nan是|||计算最大值时被忽略。 |||示例：如果x = [2 8 4，则max（x，[]，1）为[7 8 9]，||| max（x，[]，2）为[8，max（x，5）为[5 8 5 |||另请参见最小，中间，卑鄙，排序。
%   than one minimal element, the index of the first one is returned.
%
%   MIN(X,Y) returns an array the same size as X and Y with the
%   smallest elements taken from X or Y. Either one can be a scalar.
%
%   [Y,I] = MIN(X,[],DIM) operates along the dimension DIM.  Suppose
%   X=[2 8 4;7 3 9] then min(X,[],1) is [2 3 4] and min(X,[],2) is [2,3].  % 中文: x = [2 8 4; 7 3 9]然后min（x，[]，1）为[2 3 4]，min（x，[]，2）为[2,3]。 |||当复合物时，使用幅度最小（ABS（x））。 nan被忽略|||计算最小值时。 |||示例：如果x = [2 8 4，则min（x，[]，1）为[2 3 4]，||| min（x，[]，2）为[2和min（x，5）是[2 5 4 |||另请参见Max，中位数，卑鄙，排序。
%
%   When complex, the magnitude MIN(ABS(X)) is used. NaN's are ignored  % 中文: $修订：5.10 $ $日期：1997/11/21 23:23:57 $ |||兰特均匀分布的随机数。
%   when computing the minimum.  % 中文: rand（n）是一个带有随机条目的n-n矩阵，从|||选择间隔的均匀分布（0.0,1.0）。
%
%   Example: If X = [2 8 4   then min(X,[],1) is [2 3 4],  % 中文: Rand（M，N）和Rand（[M，n]）是带有随机条目的M-BB-N矩阵。
%                    7 3 9]
%
%       min(X,[],2) is [2    and min(X,5) is [2 5 4  % 中文: Rand（M，N，P，...）或Rand（[M，N，P，...]）生成随机数组。 |||没有参数的兰德是一个标量，其值每次|||都会改变。被引用。  rand（size（a））的大小与A相同。|||兰德产生伪随机数字。  数字序列|||生成由发电机的状态确定。  由于matlab |||在启动时重置状态，生成的数字顺序将
%                       3],                   5 3 5].
%
%   See also MAX, MEDIAN, MEAN, SORT.

%   Copyright (c) 1984-98 by The MathWorks, Inc.  % 中文: Mathworks，Inc。版权（C）1984-98
%   $Revision: 5.10 $  $Date: 1997/11/21 23:23:57 $

%   Built-in function.


