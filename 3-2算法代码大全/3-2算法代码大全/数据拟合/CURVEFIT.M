% 文件: CURVEFIT.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,OPTIONS,CostFunction,JACOB] = curvefit(FUN,x0,XDATA,YDATA,OPTIONS,GRADFUN,varargin)  % 详解: 函数定义：curvefit(FUN,x0,XDATA,YDATA,OPTIONS,GRADFUN,varargin), 返回：x,OPTIONS,CostFunction,JACOB





if nargin < 4, error('curvefit requires four input arguments');end  % 详解: 条件判断：if (nargin < 4, error('curvefit requires four input arguments');end)
if nargin < 5, OPTIONS=[]; end  % 详解: 条件判断：if (nargin < 5, OPTIONS=[]; end)
if nargin < 6, GRADFUN=[]; end  % 详解: 条件判断：if (nargin < 6, GRADFUN=[]; end)

curvefitarg = 1;  % 详解: 赋值：计算表达式并保存到 curvefitarg
lenVarIn = length(varargin);  % 详解: 赋值：将 length(...) 的结果保存到 lenVarIn

if isempty(FUN)  % 详解: 条件判断：if (isempty(FUN))
   error('FUN must be a function name.');  % 详解: 调用函数：error('FUN must be a function name.')
elseif isa(FUN,'inline')  % 详解: 条件判断：elseif (isa(FUN,'inline'))
   error('FUN must be a function name, not an inline object. Use LEASTSQ instead.');  % 详解: 调用函数：error('FUN must be a function name, not an inline object. Use LEASTSQ instead.')
elseif isstr(FUN)  % 详解: 条件判断：elseif (isstr(FUN))
   fun0 = fcnchk(FUN);  % 详解: 赋值：将 fcnchk(...) 的结果保存到 fun0
   if isa(fun0,'inline')  % 详解: 条件判断：if (isa(fun0,'inline'))
      error('FUN must be a function name, not a string expression. Use LEASTSQ instead.');  % 详解: 调用函数：error('FUN must be a function name, not a string expression. Use LEASTSQ instead.')
   else  % 详解: 条件判断：else 分支
      fun1 = inline(FUN,lenVarIn+1);  % 详解: 赋值：将 inline(...) 的结果保存到 fun1
      fun1args = argnames(fun1);  % 详解: 赋值：将 argnames(...) 的结果保存到 fun1args
      arglist = sprintf('%s(', formula(fun1));  % 详解: 赋值：将 sprintf(...) 的结果保存到 arglist
      for k = 1:(nargin(fun1)-1);  % 详解: for 循环：迭代变量 k 遍历 1:(nargin(fun1)-1);
         arglist = sprintf('%s%s,', arglist, deblank(fun1args{k,:}));  % 详解: 赋值：将 sprintf(...) 的结果保存到 arglist
      end  % 详解: 执行语句
      arglist = sprintf('%s%s)',arglist, deblank(fun1args{nargin(fun1),:}));  % 详解: 赋值：将 sprintf(...) 的结果保存到 arglist
      funfcn = inline([arglist, ...  % 详解: 赋值：将 inline(...) 的结果保存到 funfcn
            '-P',int2str(lenVarIn+2)],lenVarIn+2);  % 详解: 执行语句
   end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
   error('FUN is an unrecognized data type.')  % 详解: 调用函数：error('FUN is an unrecognized data type.')
end  % 详解: 执行语句

if ~isempty(GRADFUN)  % 详解: 条件判断：if (~isempty(GRADFUN))
   if isa(GRADFUN,'inline')  % 详解: 条件判断：if (isa(GRADFUN,'inline'))
      error('GRADFUN must be a function name and not an inline object.');  % 详解: 调用函数：error('GRADFUN must be a function name and not an inline object.')
   elseif isstr(GRADFUN)  % 详解: 条件判断：elseif (isstr(GRADFUN))
      fun0 = fcnchk(GRADFUN);  % 详解: 赋值：将 fcnchk(...) 的结果保存到 fun0
      if isa(fun0,'inline')  % 详解: 条件判断：if (isa(fun0,'inline'))
         error('GRADFUN must be a function name, not a string expression.');  % 详解: 调用函数：error('GRADFUN must be a function name, not a string expression.')
      else  % 详解: 条件判断：else 分支
         fun1 = inline(GRADFUN,lenVarIn+1);  % 详解: 赋值：将 inline(...) 的结果保存到 fun1
         fun1args = argnames(fun1);  % 详解: 赋值：将 argnames(...) 的结果保存到 fun1args
         arglist = sprintf('%s(', formula(fun1));  % 详解: 赋值：将 sprintf(...) 的结果保存到 arglist
         for k = 1:(nargin(fun1)-1);  % 详解: for 循环：迭代变量 k 遍历 1:(nargin(fun1)-1);
            arglist = sprintf('%s%s,', arglist, deblank(fun1args{k,:}));  % 详解: 赋值：将 sprintf(...) 的结果保存到 arglist
         end  % 详解: 执行语句
         arglist = sprintf('%s%s)',arglist, deblank(fun1args{nargin(fun1),:}));  % 详解: 赋值：将 sprintf(...) 的结果保存到 arglist
         gradfcn = inline(arglist,lenVarIn+2);  % 详解: 赋值：将 inline(...) 的结果保存到 gradfcn
      end  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      error('GRADFUN is an unrecognized data type.')  % 详解: 调用函数：error('GRADFUN is an unrecognized data type.')
   end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
   gradfcn = [];  % 详解: 赋值：计算表达式并保存到 gradfcn
end  % 详解: 执行语句


try  % 详解: 异常处理：try 块开始
   f = feval(FUN,x0,XDATA,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
catch  % 详解: 异常处理：catch
   error(['Error in evaluating user supplied function: ',FUN])  % 详解: 调用函数：error(['Error in evaluating user supplied function: ',FUN])
end  % 详解: 执行语句

if ~isequal(size(f), size(YDATA))  % 详解: 条件判断：if (~isequal(size(f), size(YDATA)))
   error('Function value and YDATA sizes are incommensurate.')  % 详解: 调用函数：error('Function value and YDATA sizes are incommensurate.')
end  % 详解: 执行语句

[x,OPTIONS,CostFunction,JACOB] = ...  % 详解: 执行语句
   nlsqold(funfcn,x0,OPTIONS,gradfcn,XDATA,varargin{:},YDATA);  % 详解: 调用函数：nlsqold(funfcn,x0,OPTIONS,gradfcn,XDATA,varargin{:},YDATA)






