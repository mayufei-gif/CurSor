% 文件: condgaussTrainObserved.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, Sigma] = mixgaussTrainObserved(obsData, hiddenData, nstates, varargin);  % 详解: 函数定义：mixgaussTrainObserved(obsData, hiddenData, nstates, varargin), 返回：mu, Sigma

[D numex] = size(obsData);  % 详解: 获取向量/矩阵尺寸
Y = zeros(D, nstates);  % 详解: 赋值：将 zeros(...) 的结果保存到 Y
YY = zeros(D,D,nstates);  % 详解: 赋值：将 zeros(...) 的结果保存到 YY
YTY = zeros(nstates,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 YTY
w = zeros(nstates, 1);  % 详解: 赋值：将 zeros(...) 的结果保存到 w
for q=1:nstates  % 详解: for 循环：迭代变量 q 遍历 1:nstates
  ndx = find(hiddenData==q);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  w(q) = length(ndx);  % 详解: 调用函数：w(q) = length(ndx)
  data = obsData(:,ndx);  % 详解: 赋值：将 obsData(...) 的结果保存到 data
  Y(:,q) = sum(data,2);  % 详解: 调用函数：Y(:,q) = sum(data,2)
  YY(:,:,q) = data*data';  % 调用函数：YY  % 详解: 执行语句  % 详解: 执行语句
  YTY(q) = sum(diag(data'*data));  % 统计：求和/均值/中位数  % 详解: 调用函数：YTY(q) = sum(diag(data'*data))  % 详解: 调用函数：YTY(q) = sum(diag(data'*data)); % 统计：求和/均值/中位数 % 详解: 调用函数：YTY(q) = sum(diag(data'*data))
end  % 详解: 执行语句
[mu, Sigma] = mixgauss_Mstep(w, Y, YY, YTY, varargin{:});  % 详解: 执行语句




