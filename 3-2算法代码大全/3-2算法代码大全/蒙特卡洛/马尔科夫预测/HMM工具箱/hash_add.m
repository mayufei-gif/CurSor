% 文件: hash_add.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function hash_add(key, val, fname)  % 详解: 函数定义：hash_add(key, val, fname)

if ~exist(fname, 'file')  % 详解: 条件判断：if (~exist(fname, 'file'))
  hashtable.key{1} = key;  % 详解: 执行语句
  hashtable.value{1} = val;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  load(fname, '-mat');  % 详解: 调用函数：load(fname, '-mat')
  Nentries = length(hashtable.key);  % 详解: 赋值：将 length(...) 的结果保存到 Nentries
  hashtable.key{Nentries+1} = key;  % 详解: 执行语句
  hashtable.value{Nentries+1} = val;  % 详解: 执行语句
end  % 详解: 执行语句
save(fname, 'hashtable', '-mat');  % 详解: 调用函数：save(fname, 'hashtable', '-mat')




