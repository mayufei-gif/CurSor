% 文件: strsplit.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function parts = strsplit(splitstr, str, option)  % 详解: 执行语句


   nargsin = nargin;  % 详解: 赋值：计算表达式并保存到 nargsin
   error(nargchk(2, 3, nargsin));  % 详解: 调用函数：error(nargchk(2, 3, nargsin))
   if nargsin < 3  % 详解: 条件判断：if (nargsin < 3)
      option = 'omit';  % 详解: 赋值：计算表达式并保存到 option
   else  % 详解: 条件判断：else 分支
      option = lower(option);  % 详解: 赋值：将 lower(...) 的结果保存到 option
   end  % 详解: 执行语句

   splitlen = length(splitstr);  % 详解: 赋值：将 length(...) 的结果保存到 splitlen
   parts = {};  % 详解: 赋值：计算表达式并保存到 parts

   while 1  % 详解: while 循环：当 (1) 为真时迭代

      k = strfind(str, splitstr);  % 详解: 赋值：将 strfind(...) 的结果保存到 k
      if isempty(k)  % 详解: 条件判断：if (isempty(k))
         parts{end+1} = str;  % 详解: 执行语句
         break  % 详解: 跳出循环：break
      end  % 详解: 执行语句

      switch option  % 详解: 多分支选择：switch (option)
         case 'include'  % 详解: 分支：case 'include'
            parts(end+1:end+2) = {str(1:k(1)-1), splitstr};  % 详解: 执行语句
         case 'append'  % 详解: 分支：case 'append'
            parts{end+1} = str(1 : k(1)+splitlen-1);  % 详解: 执行语句
         case 'omit'  % 详解: 分支：case 'omit'
            parts{end+1} = str(1 : k(1)-1);  % 详解: 执行语句
         otherwise  % 详解: 默认分支：otherwise
            error(['Invalid option string -- ', option]);  % 详解: 调用函数：error(['Invalid option string -- ', option])
      end  % 详解: 执行语句


      str = str(k(1)+splitlen : end);  % 详解: 赋值：将 str(...) 的结果保存到 str

   end  % 详解: 执行语句




