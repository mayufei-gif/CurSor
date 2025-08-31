% 文件: INT.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function r = int(f,x,a,b)  % 详解: 执行语句


f = sym(f);  % 详解: 赋值：将 sym(...) 的结果保存到 f

if nargin <= 2  % 详解: 条件判断：if (nargin <= 2)
   if nargin < 2  % 详解: 条件判断：if (nargin < 2)
      x = findsym(f,1);  % 详解: 赋值：将 findsym(...) 的结果保存到 x
      if isempty(x), x = 'x'; end  % 详解: 条件判断：if (isempty(x), x = 'x'; end)
   else  % 详解: 条件判断：else 分支
      x = char(sym(x));  % 详解: 赋值：将 char(...) 的结果保存到 x
   end  % 详解: 执行语句
   r = maple('map','int',f,x);  % 详解: 赋值：将 maple(...) 的结果保存到 r
else  % 详解: 条件判断：else 分支
   if nargin < 4  % 详解: 条件判断：if (nargin < 4)
      b = a;  % 详解: 赋值：计算表达式并保存到 b
      a = x;  % 详解: 赋值：计算表达式并保存到 a
      x = findsym(f,1);  % 详解: 赋值：将 findsym(...) 的结果保存到 x
      if isempty(x), x = 'x'; end  % 详解: 条件判断：if (isempty(x), x = 'x'; end)
   end  % 详解: 执行语句
   x = sym(x);  % 详解: 赋值：将 sym(...) 的结果保存到 x
   b = sym(b);  % 详解: 赋值：将 sym(...) 的结果保存到 b
   a = sym(a);  % 详解: 赋值：将 sym(...) 的结果保存到 a
   r = maple('map','int',f,[x.s '=' a.s '..' b.s]);  % 详解: 赋值：将 maple(...) 的结果保存到 r
end  % 详解: 执行语句

if ~isempty(findstr('int(',char(r)))  % 详解: 条件判断：if (~isempty(findstr('int(',char(r))))
   for k = 1:prod(size(r));  % 详解: for 循环：迭代变量 k 遍历 1:prod(size(r));
      rc = char(r(k));  % 详解: 赋值：将 char(...) 的结果保存到 rc
      if length(rc) >= 4 & isequal(rc(1:4),'int(')  % 详解: 条件判断：if (length(rc) >= 4 & isequal(rc(1:4),'int('))
         warning('Explicit integral could not be found.')  % 详解: 调用函数：warning('Explicit integral could not be found.')
      end  % 详解: 执行语句
   end  % 详解: 执行语句
end  % 详解: 执行语句




