% 文件: isemptycell.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function E = isemptycell(C)  % 详解: 执行语句

if 0  % 详解: 条件判断：if (0)
  E = cellfun('isempty', C);  % 详解: 赋值：将 cellfun(...) 的结果保存到 E
else  % 详解: 条件判断：else 分支
  E = zeros(size(C));  % 详解: 赋值：将 zeros(...) 的结果保存到 E
  for i=1:prod(size(C))  % 详解: for 循环：迭代变量 i 遍历 1:prod(size(C))
    E(i) = isempty(C{i});  % 详解: 调用函数：E(i) = isempty(C{i})
  end  % 详解: 执行语句
  E = logical(E);  % 详解: 赋值：将 logical(...) 的结果保存到 E
end  % 详解: 执行语句




