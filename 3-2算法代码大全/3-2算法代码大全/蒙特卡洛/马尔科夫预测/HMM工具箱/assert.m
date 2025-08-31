% 文件: assert.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function assert(pred, str)  % 详解: 函数定义：assert(pred, str)

if nargin<2, str = ''; end  % 详解: 条件判断：if (nargin<2, str = ''; end)

if ~pred  % 详解: 条件判断：if (~pred)
  s = sprintf('assertion violated: %s', str);  % 详解: 赋值：将 sprintf(...) 的结果保存到 s
  error(s);  % 详解: 调用函数：error(s)
end  % 详解: 执行语句




