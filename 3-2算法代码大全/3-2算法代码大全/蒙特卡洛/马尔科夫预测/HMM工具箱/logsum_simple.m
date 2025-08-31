% 文件: logsum_simple.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function result = logsum(logv)  % 详解: 执行语句

len = length(logv);  % 详解: 赋值：将 length(...) 的结果保存到 len
if (len<2);  % 详解: 条件判断：if ((len<2);)
  error('Subroutine logsum cannot sum less than 2 terms.');  % 详解: 调用函数：error('Subroutine logsum cannot sum less than 2 terms.')
end;  % 详解: 执行语句

if (logv(2)<logv(1)),  % 详解: 条件判断：if ((logv(2)<logv(1)),)
  result = logv(1) + log( 1 + exp( logv(2)-logv(1) ) );  % 详解: 赋值：将 logv(...) 的结果保存到 result
else,  % 详解: 条件判断：else 分支
  result = logv(2) + log( 1 + exp( logv(1)-logv(2) ) );  % 详解: 赋值：将 logv(...) 的结果保存到 result
end;  % 详解: 执行语句

for (i=3:len),  % 详解: 执行语句
  term = logv(i);  % 详解: 赋值：将 logv(...) 的结果保存到 term
  if (result<term),  % 详解: 条件判断：if ((result<term),)
    result = term   + log( 1 + exp( result-term ) );  % 详解: 赋值：计算表达式并保存到 result
  else,  % 详解: 条件判断：else 分支
    result = result + log( 1 + exp( term-result ) );  % 详解: 赋值：计算表达式并保存到 result
  end;  % 详解: 执行语句
end  % 详解: 执行语句




