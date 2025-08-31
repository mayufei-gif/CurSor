% 文件: fit_partitioned_model.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [model, partition_size] = fit_partitioned_model(...  % 详解: 执行语句
    inputs, outputs, selectors, sel_sizes, min_size, partition_names, fn_name, varargin)  % 详解: 执行语句


sel_ndx = subv2ind(sel_sizes, selectors');  % 赋值：设置变量 sel_ndx  % 详解: 赋值：将 subv2ind(...) 的结果保存到 sel_ndx  % 详解: 赋值：将 subv2ind(...) 的结果保存到 sel_ndx
Nmodels = prod(sel_sizes);  % 详解: 赋值：将 prod(...) 的结果保存到 Nmodels
model = cell(1, Nmodels);  % 详解: 赋值：将 cell(...) 的结果保存到 model
partition_size = zeros(1, Nmodels);  % 详解: 赋值：将 zeros(...) 的结果保存到 partition_size
for m=1:Nmodels  % 详解: for 循环：迭代变量 m 遍历 1:Nmodels
  ndx = find(sel_ndx==m);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  partition_size(m) = length(ndx);  % 详解: 调用函数：partition_size(m) = length(ndx)
  if ~isempty(partition_names)  % 详解: 条件判断：if (~isempty(partition_names))
    fprintf('partition %s has size %d, min size = %d\n', ...  % 详解: 打印/显示输出
	    partition_names{m}, partition_size(m), min_size);  % 详解: 执行语句
  end  % 详解: 执行语句
  if partition_size(m) >= min_size  % 详解: 条件判断：if (partition_size(m) >= min_size)
    if isempty(inputs)  % 详解: 条件判断：if (isempty(inputs))
      model{m} = feval(fn_name, outputs(:, ndx), varargin{:});  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      model{m} = feval(fn_name, inputs(:,ndx), outputs(:, ndx), varargin{:});  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




