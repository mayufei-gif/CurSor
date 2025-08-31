% 文件: nchoose2.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function c = nchoose2(v, f)  % 详解: 执行语句


   nargs = nargin;  % 详解: 赋值：计算表达式并保存到 nargs
   if nargs < 1  % 详解: 条件判断：if (nargs < 1)
      error('Not enough input arguments.');  % 详解: 调用函数：error('Not enough input arguments.')
   elseif nargs == 1  % 详解: 条件判断：elseif (nargs == 1)
      v = v(:);  % 详解: 赋值：将 v(...) 的结果保存到 v
      n = length(v);  % 详解: 赋值：将 length(...) 的结果保存到 n
   elseif nargs == 2  % 详解: 条件判断：elseif (nargs == 2)
      n = v;  % 详解: 赋值：计算表达式并保存到 n
   else  % 详解: 条件判断：else 分支
      error('Too many input arguments.');  % 详解: 调用函数：error('Too many input arguments.')
   end  % 详解: 执行语句

   [ c(:,2), c(:,1) ] = find( tril( ones(n), -1 ) );  % 详解: 创建全 1 矩阵/数组

   if nargs == 1  % 详解: 条件判断：if (nargs == 1)
      c = v(c);  % 详解: 赋值：将 v(...) 的结果保存到 c
   end  % 详解: 执行语句




