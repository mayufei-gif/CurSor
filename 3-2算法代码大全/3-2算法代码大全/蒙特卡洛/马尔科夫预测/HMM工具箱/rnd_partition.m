% 文件: rnd_partition.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [train, test] = rnd_partition(data, train_percent);  % 详解: 函数定义：rnd_partition(data, train_percent), 返回：train, test

N = size(data, 2);  % 详解: 赋值：将 size(...) 的结果保存到 N
ndx = randperm(N);  % 详解: 赋值：将 randperm(...) 的结果保存到 ndx
k = ceil(N*train_percent);  % 详解: 赋值：将 ceil(...) 的结果保存到 k
train_ndx = ndx(1:k);  % 详解: 赋值：将 ndx(...) 的结果保存到 train_ndx
test_ndx = ndx(k+1:end);  % 详解: 赋值：将 ndx(...) 的结果保存到 test_ndx
train =  data(:, train_ndx);  % 详解: 赋值：将 data(...) 的结果保存到 train
test = data(:, test_ndx);  % 详解: 赋值：将 data(...) 的结果保存到 test




