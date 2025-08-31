% 文件: mhmmParzen_train_observed.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [initState, transmat, mu, Nproto, pick] = mhmmParzen_train_observed(obsData, hiddenData, ...  % 详解: 执行语句
						  nstates, maxNproto, varargin)  % 详解: 执行语句

[transmat, initState] = transmat_train_observed(...  % 详解: 执行语句
    hiddenData, nstates, varargin{:});  % 详解: 执行语句

if ~iscell(obsData)  % 详解: 条件判断：if (~iscell(obsData))
  [D T Nex] = size(obsData);  % 详解: 获取向量/矩阵尺寸
  obsData = reshape(obsData, D, T*Nex);  % 详解: 赋值：将 reshape(...) 的结果保存到 obsData
else  % 详解: 条件判断：else 分支
  obsData = cat(2, obsData{:});  % 详解: 赋值：将 cat(...) 的结果保存到 obsData
  hiddenData = cat(2, hiddenData{:});  % 详解: 赋值：将 cat(...) 的结果保存到 hiddenData
end  % 详解: 执行语句
[mu, Nproto, pick] = parzen_fit_select_unif(obsData, hiddenData(:), maxNproto);  % 详解: 执行语句

 




