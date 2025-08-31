% 文件: strmatch_multi.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [posns] = strmatch_multi(keys, strs)  % 详解: 函数定义：strmatch_multi(keys, strs), 返回：posns

if ~iscell(keys), keys = {keys}; end  % 详解: 条件判断：if (~iscell(keys), keys = {keys}; end)
nkeys = length(keys);  % 详解: 赋值：将 length(...) 的结果保存到 nkeys
posns = zeros(1, nkeys);  % 详解: 赋值：将 zeros(...) 的结果保存到 posns
if length(keys) < length(strs)  % 详解: 条件判断：if (length(keys) < length(strs))
  for i=1:nkeys  % 详解: for 循环：迭代变量 i 遍历 1:nkeys
    ndx = strcmp(keys{i}, strs);  % 详解: 赋值：将 strcmp(...) 的结果保存到 ndx
    pos = find(ndx);  % 详解: 赋值：将 find(...) 的结果保存到 pos
    if ~isempty(pos)  % 详解: 条件判断：if (~isempty(pos))
      posns(i) = pos(1);  % 详解: 调用函数：posns(i) = pos(1)
    end  % 详解: 执行语句
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  for s=1:length(strs)  % 详解: for 循环：迭代变量 s 遍历 1:length(strs)
    ndx = strcmp(strs{s}, keys);  % 详解: 赋值：将 strcmp(...) 的结果保存到 ndx
    ndx = find(ndx);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
    posns(ndx) = s;  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句





