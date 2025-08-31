% 文件: strmatch_substr.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function ndx = strmatch_substr(str, strs)  % 详解: 执行语句

ndx = [];  % 详解: 赋值：计算表达式并保存到 ndx
if ~iscell(str), str = {str}; end  % 详解: 条件判断：if (~iscell(str), str = {str}; end)
for j=1:length(str)  % 详解: for 循环：迭代变量 j 遍历 1:length(str)
  for i=1:length(strs)  % 详解: for 循环：迭代变量 i 遍历 1:length(strs)
    ind = findstr(strs{i}, str{j});  % 详解: 赋值：将 findstr(...) 的结果保存到 ind
    if ~isempty(ind)  % 详解: 条件判断：if (~isempty(ind))
      ndx = [ndx; i];  % 详解: 赋值：计算表达式并保存到 ndx
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




