% 文件: hash_del.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function ndx = hash_del(key, fname)  % 详解: 执行语句

ndx = [];  % 详解: 赋值：计算表达式并保存到 ndx

if ~exist(fname, 'file')  % 详解: 条件判断：if (~exist(fname, 'file'))
else  % 详解: 条件判断：else 分支
  load(fname, '-mat');  % 详解: 调用函数：load(fname, '-mat')
  Nentries = length(hashtable.key);  % 详解: 赋值：将 length(...) 的结果保存到 Nentries
  for i=1:Nentries  % 详解: for 循环：迭代变量 i 遍历 1:Nentries
    if isequal(hashtable.key{i}, key)  % 详解: 条件判断：if (isequal(hashtable.key{i}, key))
      ndx = [ndx i];  % 详解: 赋值：计算表达式并保存到 ndx
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  hashtable.key(ndx) = [];  % 详解: 执行语句
  hashtable.value(ndx) = [];  % 详解: 执行语句
  save(fname, 'hashtable', '-mat');  % 详解: 调用函数：save(fname, 'hashtable', '-mat')
end  % 详解: 执行语句





