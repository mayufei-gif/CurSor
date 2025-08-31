% 文件: DSOLVE.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function varargout = dsolve(varargin)  % 详解: 执行语句


narg = nargin;  % 详解: 赋值：计算表达式并保存到 narg

x = 't';  % 详解: 赋值：计算表达式并保存到 x
if all(varargin{narg} ~= '='),  % 详解: 条件判断：if (all(varargin{narg} ~= '='),)
   x = varargin{narg}; narg = narg-1;  % 详解: 赋值：计算表达式并保存到 x
end;  % 详解: 执行语句

sys = varargin{1};  % 详解: 赋值：计算表达式并保存到 sys
for k = 2: narg  % 详解: for 循环：迭代变量 k 遍历 2: narg
   sys = [sys ', ' varargin{k}];;  % 详解: 赋值：计算表达式并保存到 sys
end  % 详解: 执行语句


var_set = '{';  % 详解: 赋值：计算表达式并保存到 var_set

d = [find(sys == 'D') length(sys)+1];  % 详解: 赋值：计算表达式并保存到 d

new_sys = sys(1:d-1);  % 详解: 赋值：将 sys(...) 的结果保存到 new_sys

for kd = 1:length(d)-1  % 详解: for 循环：迭代变量 kd 遍历 1:length(d)-1
   Dstr = sys(d(kd)+1:d(kd+1)-1);  % 详解: 赋值：将 sys(...) 的结果保存到 Dstr
   iletter = find(isletter(Dstr));  % 详解: 赋值：将 find(...) 的结果保存到 iletter

   if iletter(1)==1  % 详解: 条件判断：if (iletter(1)==1)
      new_sys = [new_sys '(D' Dstr(1:iletter(1)-1) ')' Dstr(iletter(1):end)];  % 详解: 赋值：计算表达式并保存到 new_sys
   else  % 详解: 条件判断：else 分支
      new_sys = [new_sys '(D@@' Dstr(1:iletter(1)-1) ')' Dstr(iletter(1):end)];  % 详解: 赋值：计算表达式并保存到 new_sys
   end  % 详解: 执行语句

   Dstr1 = Dstr(iletter(1):end);  % 详解: 赋值：将 Dstr(...) 的结果保存到 Dstr1
   ialphanum = find(~isletter(Dstr1) & (Dstr1 < '0' | Dstr1 > '9'));  % 详解: 赋值：将 find(...) 的结果保存到 ialphanum
   var_set = [var_set Dstr1(1:ialphanum(1)-1) ','];  % 详解: 赋值：计算表达式并保存到 var_set
end  % 详解: 执行语句

var_set(end) = '}';  % 详解: 执行语句
var_set = maple([var_set ' intersect ' var_set]);  % 详解: 赋值：将 maple(...) 的结果保存到 var_set


indx_ic = length(new_sys);  % 详解: 赋值：将 length(...) 的结果保存到 indx_ic
ic_str = [];  % 详解: 赋值：计算表达式并保存到 ic_str
eq_str = new_sys;  % 详解: 赋值：计算表达式并保存到 eq_str
var_str = '{';  % 详解: 赋值：计算表达式并保存到 var_str
vars = [',' var_set(2:end-1) ','];  % 详解: 赋值：计算表达式并保存到 vars
vars(find(vars==' '))=[];  % 详解: 执行语句
kommas = find(vars==',');  % 详解: 赋值：将 find(...) 的结果保存到 kommas

for k = 1: length(kommas)-1  % 详解: for 循环：迭代变量 k 遍历 1: length(kommas)-1
   v = vars(kommas(k)+1:kommas(k+1)-1);  % 详解: 赋值：将 vars(...) 的结果保存到 v

   var_str = [var_str v '(' x '),'];  % 详解: 赋值：计算表达式并保存到 var_str

   indx = findstr(eq_str, [v '(']);  % 详解: 赋值：将 findstr(...) 的结果保存到 indx
   if isempty(indx), indx = indx_ic; end  % 详解: 条件判断：if (isempty(indx), indx = indx_ic; end)
   indx_ic = min(indx_ic,indx(1));  % 详解: 赋值：将 min(...) 的结果保存到 indx_ic
   eq_str = new_sys(1:min(indx_ic));  % 详解: 赋值：将 new_sys(...) 的结果保存到 eq_str
end  % 详解: 执行语句

var_str(end) = '}';  % 详解: 执行语句

if indx_ic < length(new_sys)  % 详解: 条件判断：if (indx_ic < length(new_sys))
   last_comma = max(find(eq_str==','));  % 详解: 赋值：将 max(...) 的结果保存到 last_comma
   ic_str = new_sys(last_comma:end);  % 详解: 赋值：将 new_sys(...) 的结果保存到 ic_str
   eq_str = eq_str(1: last_comma-1);  % 详解: 赋值：将 eq_str(...) 的结果保存到 eq_str
end  % 详解: 执行语句

for j = 1:length(kommas)-1  % 详解: for 循环：迭代变量 j 遍历 1:length(kommas)-1
   v = vars(kommas(j)+1:kommas(j+1)-1);  % 详解: 赋值：将 vars(...) 的结果保存到 v
   m = length(v);  % 详解: 赋值：将 length(...) 的结果保存到 m
   e = length(eq_str);  % 详解: 赋值：将 length(...) 的结果保存到 e
   for k = fliplr(findstr(v,eq_str))  % 详解: for 循环：迭代变量 k 遍历 fliplr(findstr(v,eq_str))
      if k+m > e | ~isvarname(eq_str(k:k+m))  % 详解: 条件判断：if (k+m > e | ~isvarname(eq_str(k:k+m)))
         eq_str = [eq_str(1:k-1) '(' v ')(' x ')' eq_str(k+m:end)];  % 详解: 赋值：计算表达式并保存到 eq_str
      end  % 详解: 执行语句
   end  % 详解: 执行语句
end  % 详解: 执行语句

for j = 1:length(kommas)-1  % 详解: for 循环：迭代变量 j 遍历 1:length(kommas)-1
   v = vars(kommas(j)+1:kommas(j+1)-1);  % 详解: 赋值：将 vars(...) 的结果保存到 v
   m = length(v);  % 详解: 赋值：将 length(...) 的结果保存到 m
   e = length(ic_str);  % 详解: 赋值：将 length(...) 的结果保存到 e
   for k = fliplr(findstr(v,ic_str))  % 详解: for 循环：迭代变量 k 遍历 fliplr(findstr(v,ic_str))
      if k+m > e | ~isvarname(ic_str(k:k+m))  % 详解: 条件判断：if (k+m > e | ~isvarname(ic_str(k:k+m)))
         ic_str = [ic_str(1:k-1) '(' v ')' ic_str(k+m:end)];  % 详解: 赋值：计算表达式并保存到 ic_str
      end  % 详解: 执行语句
   end  % 详解: 执行语句
end  % 详解: 执行语句

[R,stat] = maple('dsolve', ...  % 详解: 执行语句
   ['convert({',eq_str,ic_str,'},fraction)'], var_str, 'explicit');  % 详解: 执行语句
if stat  % 详解: 条件判断：if (stat)
   error(R)  % 详解: 调用函数：error(R)
end  % 详解: 执行语句


if isempty(R) | ~isempty(findstr(R,'DESol'))  % 详解: 条件判断：if (isempty(R) | ~isempty(findstr(R,'DESol')))
   warning('Explicit solution could not be found.');  % 详解: 调用函数：warning('Explicit solution could not be found.')
   varargout = cell(1,nargout);  % 详解: 赋值：将 cell(...) 的结果保存到 varargout
   varargout{1} = sym([]);  % 详解: 执行语句
   return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


R(findstr(R,'_C')) = [];  % 详解: 执行语句


if R(1) ~= '{', R = ['{' R '}']; end  % 详解: 条件判断：if (R(1) ~= '{', R = ['{' R '}']; end)
vars(1) = '['; vars(end) = ']';  % 详解: 执行语句
vars = maple('sort',vars);  % 详解: 赋值：将 maple(...) 的结果保存到 vars
vars(1) = '{'; vars(end) = '}';  % 详解: 执行语句
nvars = sum(commas(vars))+1;  % 详解: 赋值：将 sum(...) 的结果保存到 nvars

if nvars == 1 & nargout <= 1  % 详解: 条件判断：if (nvars == 1 & nargout <= 1)


   S = sym([]);  % 详解: 赋值：将 sym(...) 的结果保存到 S
   c = find(commas(R) | R == '}');  % 详解: 赋值：将 find(...) 的结果保存到 c
   for p = find(R == '=')  % 详解: for 循环：迭代变量 p 遍历 find(R == '=')
      q = min(c(c>p));  % 详解: 赋值：将 min(...) 的结果保存到 q
      t = trim(R(p+1:q-1));  % 详解: 赋值：将 trim(...) 的结果保存到 t
      S = [S; sym(t)];  % 详解: 赋值：计算表达式并保存到 S
   end  % 详解: 执行语句
   varargout{1} = S;  % 详解: 执行语句

else  % 详解: 条件判断：else 分支


   c = [1 find(commas(vars)) length(vars)];  % 详解: 赋值：计算表达式并保存到 c
   S = [];  % 详解: 赋值：计算表达式并保存到 S
   for j = 1:nvars  % 详解: for 循环：迭代变量 j 遍历 1:nvars
      v = trim(vars(c(j)+1:c(j+1)-1));  % 详解: 赋值：将 trim(...) 的结果保存到 v
      S = setfield(S,v,[]);  % 详解: 赋值：将 setfield(...) 的结果保存到 S
   end  % 详解: 执行语句


   c = [1 find(commas(R) | R == '{' | R == '}') length(R)];  % 详解: 赋值：计算表达式并保存到 c
   for p = find(R == '=')  % 详解: for 循环：迭代变量 p 遍历 find(R == '=')
      q = max(c(c<p));  % 详解: 赋值：将 max(...) 的结果保存到 q
      v = trim(R(q+1:p-1));  % 详解: 赋值：将 trim(...) 的结果保存到 v
      v(findstr(v,'('):findstr(v,')')) = [];  % 详解: 执行语句
      q = min(c(c>p));  % 详解: 赋值：将 min(...) 的结果保存到 q
      t = trim(R(p+1:q-1));  % 详解: 赋值：将 trim(...) 的结果保存到 t
      S = setfield(S,v,[getfield(S,v); sym(t)]);  % 详解: 赋值：将 setfield(...) 的结果保存到 S
   end  % 详解: 执行语句
   
   if nargout <= 1  % 详解: 条件判断：if (nargout <= 1)

      varargout{1} = S;  % 详解: 执行语句

   elseif nargout == nvars  % 详解: 条件判断：elseif (nargout == nvars)

      v = fieldnames(S);  % 详解: 赋值：将 fieldnames(...) 的结果保存到 v
      for j = 1:nvars  % 详解: for 循环：迭代变量 j 遍历 1:nvars
         varargout{j} = getfield(S,v{j});  % 详解: 执行语句
      end  % 详解: 执行语句

   else  % 详解: 条件判断：else 分支
      error([int2str(nvars) ' variables does not match ' ...  % 详解: 执行语句
             int2str(nargout) ' outputs.'])  % 详解: 调用函数：int2str(nargout) ' outputs.'])
   end  % 详解: 执行语句
end  % 详解: 执行语句

function s = trim(s);  % 详解: 执行语句
while s(1) == ' ', s(1) = []; end  % 详解: while 循环：当 (s(1) == ' ', s(1) = []; end) 为真时迭代
while s(end) == ' ', s(end) = []; end  % 详解: while 循环：当 (s(end) == ' ', s(end) = []; end) 为真时迭代


function c = commas(s)  % 详解: 执行语句
p = cumsum((s == '(') - (s == ')'));  % 详解: 赋值：将 cumsum(...) 的结果保存到 p
c = (s == ',') & (p == 0);  % 详解: 赋值：计算表达式并保存到 c




