% 文件: parzen_fit_select_unif.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, N, pick] = parzen_fit_select_unif(data, labels, max_proto, varargin)  % 详解: 函数定义：parzen_fit_select_unif(data, labels, max_proto, varargin), 返回：mu, N, pick

nclasses = max(labels);  % 详解: 赋值：将 max(...) 的结果保存到 nclasses
[boundary, partition_names] = process_options(...  % 详解: 执行语句
    varargin, 'boundary', 0, 'partition_names', []);  % 详解: 执行语句

[D T] = size(data);  % 详解: 获取向量/矩阵尺寸
mu = zeros(D, 1, nclasses);  % 详解: 赋值：将 zeros(...) 的结果保存到 mu
mean_feat = mean(data,2);  % 详解: 赋值：将 mean(...) 的结果保存到 mean_feat
pick = cell(1,nclasses);  % 详解: 赋值：将 cell(...) 的结果保存到 pick
for c=1:nclasses  % 详解: for 循环：迭代变量 c 遍历 1:nclasses
  ndx = find(labels==c);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  if isempty(ndx)  % 详解: 条件判断：if (isempty(ndx))
    fprintf('no training images have label %d\n', c);  % 详解: 调用函数：fprintf('no training images have label %d\n', c)
    nviews = 1;  % 详解: 赋值：计算表达式并保存到 nviews
    mu(:,1,c) = mean_feat;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    foo = linspace(boundary+1, length(ndx-boundary), max_proto);  % 详解: 赋值：将 linspace(...) 的结果保存到 foo
    pick{c} = ndx(unique(floor(foo)));  % 详解: 执行语句
    nviews = length(pick{c});  % 详解: 赋值：将 length(...) 的结果保存到 nviews
    mu(:,1:nviews,c) = data(:, pick{c});  % 详解: 调用函数：mu(:,1:nviews,c) = data(:, pick{c})
  end  % 详解: 执行语句
  N(c) = nviews;  % 详解: 执行语句
end  % 详解: 执行语句




