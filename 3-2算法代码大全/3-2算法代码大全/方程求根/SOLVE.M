% 文件: SOLVE.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function varargout = solve(varargin)  % 详解: 执行语句



maple('_EnvExplicit := true;');  % 详解: 调用函数：maple('_EnvExplicit := true;')


eqns = [];  % 详解: 赋值：计算表达式并保存到 eqns
vars = [];  % 详解: 赋值：计算表达式并保存到 vars
for k = 1:nargin  % 详解: for 循环：迭代变量 k 遍历 1:nargin
   v = varargin{k};  % 详解: 赋值：计算表达式并保存到 v
   if all(isletter(v) | ('0'<=v & v<='9') | v == '_' | v == ',')  % 详解: 条件判断：if (all(isletter(v) | ('0'<=v & v<='9') | v == '_' | v == ','))
      vars = [vars ',' v];  % 详解: 赋值：计算表达式并保存到 vars
   else  % 详解: 条件判断：else 分支
      [t,stat] = maple(v);  % 详解: 执行语句
      if stat  % 详解: 条件判断：if (stat)
         error(['''' v ''' is not a valid expression or equation.'])  % 详解: 调用函数：error(['''' v ''' is not a valid expression or equation.'])
      end  % 详解: 执行语句
      eqns = [eqns ',' v];  % 详解: 赋值：计算表达式并保存到 eqns
   end  % 详解: 执行语句
end  % 详解: 执行语句

if isempty(eqns)  % 详解: 条件判断：if (isempty(eqns))
   warning('List of equations is empty.')  % 详解: 调用函数：warning('List of equations is empty.')
   varargout = cell(1,nargout);  % 详解: 赋值：将 cell(...) 的结果保存到 varargout
   varargout{1} = sym([]);  % 详解: 执行语句
   return  % 详解: 返回：从当前函数返回
else  % 详解: 条件判断：else 分支
   eqns(1) = '{'; eqns(end+1) = '}';  % 详解: 执行语句
end  % 详解: 执行语句
neqns = sum(commas(eqns)) + 1;  % 详解: 赋值：将 sum(...) 的结果保存到 neqns


if isempty(vars)  % 详解: 条件判断：if (isempty(vars))
   vars = ['[' findsym(sym(eqns),neqns) ']'];  % 详解: 赋值：计算表达式并保存到 vars
else  % 详解: 条件判断：else 分支
   vars(1) = '['; vars(end+1) = ']';  % 详解: 执行语句
end  % 详解: 执行语句
v = vars;  % 详解: 赋值：计算表达式并保存到 v
[vars,stat] = maple('sort',v);  % 详解: 执行语句
if stat  % 详解: 条件判断：if (stat)
   error(['''' v ''' is not a valid variable list.'])  % 详解: 调用函数：error(['''' v ''' is not a valid variable list.'])
end  % 详解: 执行语句
vars(1) = '{'; vars(end) = '}';  % 详解: 执行语句
nvars = sum(commas(vars)) + 1;  % 详解: 赋值：将 sum(...) 的结果保存到 nvars

if neqns ~= nvars  % 详解: 条件判断：if (neqns ~= nvars)
   warning([int2str(neqns) ' equations in ' int2str(nvars) ' variables.'])  % 详解: 调用函数：warning([int2str(neqns) ' equations in ' int2str(nvars) ' variables.'])
end  % 详解: 执行语句


[R,stat] = maple('solve',eqns,vars);  % 详解: 执行语句


if (isempty(R) & (nvars == neqns))  % 详解: 条件判断：if ((isempty(R) & (nvars == neqns)))
   [R,stat] = maple('fsolve',eqns,vars);  % 详解: 执行语句
end  % 详解: 执行语句

if stat  % 详解: 条件判断：if (stat)
   error(R)  % 详解: 调用函数：error(R)
end  % 详解: 执行语句


if isempty(R) | findstr(R,'fsolve')  % 详解: 条件判断：if (isempty(R) | findstr(R,'fsolve'))
   warning('Explicit solution could not be found.');  % 详解: 调用函数：warning('Explicit solution could not be found.')
   varargout = cell(1,nargout);  % 详解: 赋值：将 cell(...) 的结果保存到 varargout
   varargout{1} = sym([]);  % 详解: 执行语句
   return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


while ~isempty(findstr(R,'RootOf'))  % 详解: while 循环：当 (~isempty(findstr(R,'RootOf'))) 为真时迭代
   k = min(findstr(R,'RootOf'));  % 详解: 赋值：将 min(...) 的结果保存到 k
   p = findstr(R,'{'); p = max(p(p<k));  % 详解: 赋值：将 findstr(...) 的结果保存到 p
   q = findstr(R,'}'); q = min(q(q>k));  % 详解: 赋值：将 findstr(...) 的结果保存到 q
   s = R(p:q);  % 详解: 赋值：将 R(...) 的结果保存到 s
   t = s(min(findstr(s,'RootOf'))+6:end);  % 详解: 赋值：将 s(...) 的结果保存到 t
   e = min(find(cumsum((t=='(')-(t==')'))==0));  % 详解: 赋值：将 min(...) 的结果保存到 e
   if isempty(find(commas(t(2:e-1))))  % 详解: 条件判断：if (isempty(find(commas(t(2:e-1)))))
      [s,stat] = maple('allvalues',s,'dependent');  % 详解: 执行语句
      if stat  % 详解: 条件判断：if (stat)
         error(s)  % 详解: 调用函数：error(s)
      end  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      s = maple('evalf',s);  % 详解: 赋值：将 maple(...) 的结果保存到 s
   end  % 详解: 执行语句
   R = [R(1:p-1) s R(q+1:end)];  % 详解: 赋值：计算表达式并保存到 R
end  % 详解: 执行语句


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




