% 文件: MAX.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%列状数据最大值
%例如
%    A=[11 4 0.2;22 3 0.5;0 3 0.4];
%    max(A)
%
%MAX    Largest component.
%   For vectors, MAX(X) is the largest element in X. For matrices,
%   MAX(X) is a row vector containing the maximum element from each
%   column. For N-D arrays, MAX(X) operates along the first  % 中文: 柱子。对于N-D数组，Max（X）沿第一个|||运行非辛格尔顿维度。
%   non-singleton dimension.  % 中文: [y，i] = max（x）返回向量I中最大值的索引。|||如果沿第一个非单星维度的值包含更多|||比一个最大元素，第一个索引返回。
%
%   [Y,I] = MAX(X) returns the indices of the maximum values in vector I.  % 中文: Max（X，Y）返回一个与x和y相同的数组，并返回|||最大的元素取自X或Y。任何一个都可以是标量。
%   If the values along the first non-singleton dimension contain more  % 中文: [y，i] = max（x，[]，dim）沿dimension dim。 |||当复合物时，使用最大幅度（ABS（x））。  nan是|||计算最大值时被忽略。 |||示例：如果x = [2 8 4，则max（x，[]，1）为[7 8 9]，||| max（x，[]，2）为[8，max（x，5）为[5 8 5 |||另请参见最小，中间，卑鄙，排序。
%   than one maximal element, the index of the first one is returned.  % 中文: $修订：5.11 $ $日期：1997/11/21 23:23:54 $ |||平均值（a）|||平均平均值或平均值。 |||对于向量，平均值（x）是x中元素的平均值。矩阵，平均（x）是一个行矢量，包含的平均值
%
%   MAX(X,Y) returns an array the same size as X and Y with the
%   largest elements taken from X or Y.  Either one can be a scalar.
%
%   [Y,I] = MAX(X,[],DIM) operates along the dimension DIM. 
%
%   When complex, the magnitude MAX(ABS(X)) is used.  NaN's are
%   ignored when computing the maximum.
%
%   Example: If X = [2 8 4   then max(X,[],1) is [7 8 9],
%                    7 3 9]
%
%       max(X,[],2) is [8    and max(X,5) is [5 8 5
%                       9],                   7 5 9].
%
%   See also MIN, MEDIAN, MEAN, SORT.

%   Copyright (c) 1984-98 by The MathWorks, Inc.  % 中文: Mathworks，Inc。版权（C）1984-98
%   $Revision: 5.11 $  $Date: 1997/11/21 23:23:54 $

%   Built-in function.


