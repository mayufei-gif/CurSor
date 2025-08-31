% 文件: mysize.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function sz = mysize(M)  % 详解: 执行语句

if isvector(M)  % 详解: 条件判断：if (isvector(M))
  sz = length(M);  % 详解: 赋值：将 length(...) 的结果保存到 sz
else  % 详解: 条件判断：else 分支
  sz = size(M);  % 详解: 赋值：将 size(...) 的结果保存到 sz
end  % 详解: 执行语句




