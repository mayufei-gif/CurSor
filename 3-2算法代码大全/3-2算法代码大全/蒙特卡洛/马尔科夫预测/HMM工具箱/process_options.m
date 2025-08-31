% 文件: process_options.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% PROCESS_OPTIONS - Processes options passed to a Matlab function.
%                   This function provides a simple means of
%                   parsing attribute-value options.  Each option is
%                   named by a unique string and is given a default
%                   value.
%
% Usage:  [var1, var2, ..., varn[, unused]] = ...
%           process_options(args, ...
%                           str1, def1, str2, def2, ..., strn, defn)
%
% Arguments:   
%            args            - a cell array of input arguments, such
%                              as that provided by VARARGIN.  Its contents
%                              should alternate between strings and
%                              values.
%            str1, ..., strn - Strings that are associated with a 
%                              particular variable  % 中文: 特定变量||| def1，...，defn-如果没有选项|||返回的默认值提供||| var1，...，varn-要分配给变量的值|||未使用 - 这些|||的可选单元格数组阵列未使用的弦值对； |||如果不提供，则|||每个|||将发出警告在缺乏比赛的arg中选项。 |||假设我们希望定义具有|||的MATLAB函数'func'所需的参数x和y，以及可选参数“ u”和“ v”。 |||定义|||函数y = func（x，y，varargin）||| [u，v] = process_options（varargin，'u'，0，'v'，1）; |||调用func（0，1，'v'，2）将分配0给x，1至y，0给u，以及2
%            def1, ..., defn - Default values returned if no option  % 中文: v。参数名称对情况不敏感；打电话||| func（0，1，'v'，2）具有相同的效果。  功能调用||| func（0，1，'u'，5，'z'，2）; |||将导致u具有值5和V的值1，但是|||将发出警告，即尚未使用“ Z”选项。  在
%                              is supplied
%
% Returns:
%            var1, ..., varn - values to be assigned to variables
%            unused          - an optional cell array of those 
%                              string-value pairs that were unused;
%                              if this is not supplied, then a
%                              warning will be issued for each
%                              option in args that lacked a match.
%
% Examples:
%
% Suppose we wish to define a Matlab function 'func' that has
% required parameters x and y, and optional arguments 'u' and 'v'.
% With the definition
%
%   function y = func(x, y, varargin)
%
%     [u, v] = process_options(varargin, 'u', 0, 'v', 1);
%
% calling func(0, 1, 'v', 2) will assign 0 to x, 1 to y, 0 to u, and 2
% to v.  The parameter names are insensitive to case; calling 
% func(0, 1, 'V', 2) has the same effect.  The function call
% 
%   func(0, 1, 'u', 5, 'z', 2);
%
% will result in u having the value 5 and v having value 1, but
% will issue a warning that the 'z' option has not been used.  On
% the other hand, if func is defined as  % 中文: 另一方面，如果将函数定义为||| [u，v，unused_args] = process_options（varargin，'u'，0，'v'，1）; |||然后呼叫函数（0，1，'u'，5，'z'，2）不会发出任何警告，||| unused_args将具有{'z'，2}的值。  此行为是|||对于具有调用其他功能的选项的功能|||有用有选项；所有选项都可以传递给外部功能，|||其未经处理的参数可以传递给内部函数。 |||检查输入参数的数量|||检查提供的输出参数的数量|||将输出设置为默认值|||现在处理所有参数||| s''未使用。'，args {i}））; |||分配未使用的参数|||创建一个由k的随机正定矩阵D大小为d（k默认为1）||| m = rand_psd（d，d2，k）默认值：d2 = d，k = 1
%
%   function y = func(x, y, varargin)
%
%     [u, v, unused_args] = process_options(varargin, 'u', 0, 'v', 1);  % 中文: a（i，:) = [x y w h]
%
% then the call func(0, 1, 'u', 5, 'z', 2) will yield no warning,  % 中文: b（j，:) = [x y w h] |||重叠（i，j）=交叉区域||| normoverlap（i，j）=重叠（i，j） / min（区域（i），区域（j））
% and unused_args will have the value {'z', 2}.  This behaviour is
% useful for functions with options that invoke other functions
% with options; all options can be passed to the outer function and
% its unprocessed arguments can be passed to the inner function.

% Copyright (C) 2002 Mark A. Paskin
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by  % 中文: 根据|||发布的GNU通用公共许可条款的条款自由软件基金会；许可证的第2版，或||| （您可以选择）任何以后的版本。 |||该程序的分布是希望它将有用的，但是|||没有任何保修；甚至没有|||的隐含保证适合或适合特定目的的健身。  请参阅gnu |||通用公共许可证以获取更多详细信息。 |||您应该已经收到了GNU通用公共许可证的副本|||以及这个程序；如果没有，请写入免费软件||| Foundation，Inc。，59 Temple Place，Suite 330，马萨诸塞州波士顿02111-1307 |||美国。 |||琐碎的情况：|||首先，通过简单的最佳匹配来减少问题。  如果两个|||元素同意它们是最好的匹配，然后将它们匹配。 |||获取两组的（新）大小，u和v。||| mx = realmax; |||将亲和力矩阵为正方形|||运行匈牙利方法。  首先用|||替换无限值最大（或最小）有限值。
% the Free Software Foundation; either version 2 of the License, or  % 中文: fprintf（'跑步匈牙利\ n'）;
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
% USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varargout] = process_options(args, varargin)  % 详解: 函数定义：process_options(args, varargin), 返回：varargout

n = length(varargin);  % 详解: 赋值：将 length(...) 的结果保存到 n
if (mod(n, 2))  % 详解: 条件判断：if ((mod(n, 2)))
  error('Each option must be a string/value pair.');  % 详解: 调用函数：error('Each option must be a string/value pair.')
end  % 详解: 执行语句

if (nargout < (n / 2))  % 详解: 条件判断：if ((nargout < (n / 2)))
  error('Insufficient number of output arguments given');  % 详解: 调用函数：error('Insufficient number of output arguments given')
elseif (nargout == (n / 2))  % 详解: 条件判断：elseif ((nargout == (n / 2)))
  warn = 1;  % 详解: 赋值：计算表达式并保存到 warn
  nout = n / 2;  % 详解: 赋值：计算表达式并保存到 nout
else  % 详解: 条件判断：else 分支
  warn = 0;  % 详解: 赋值：计算表达式并保存到 warn
  nout = n / 2 + 1;  % 详解: 赋值：计算表达式并保存到 nout
end  % 详解: 执行语句

varargout = cell(1, nout);  % 详解: 赋值：将 cell(...) 的结果保存到 varargout
for i=2:2:n  % 详解: for 循环：迭代变量 i 遍历 2:2:n
  varargout{i/2} = varargin{i};  % 详解: 执行语句
end  % 详解: 执行语句

nunused = 0;  % 详解: 赋值：计算表达式并保存到 nunused
for i=1:2:length(args)  % 详解: for 循环：迭代变量 i 遍历 1:2:length(args)
  found = 0;  % 详解: 赋值：计算表达式并保存到 found
  for j=1:2:n  % 详解: for 循环：迭代变量 j 遍历 1:2:n
    if strcmpi(args{i}, varargin{j})  % 详解: 条件判断：if (strcmpi(args{i}, varargin{j}))
      varargout{(j + 1)/2} = args{i + 1};  % 详解: 执行语句
      found = 1;  % 详解: 赋值：计算表达式并保存到 found
      break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  if (~found)  % 详解: 条件判断：if ((~found))
    if (warn)  % 详解: 条件判断：if ((warn))
      warning(sprintf('Option ''%s'' not used.', args{i}));  % 详解: 调用函数：warning(sprintf('Option ''%s'' not used.', args{i}))
      args{i}  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      nunused = nunused + 1;  % 详解: 赋值：计算表达式并保存到 nunused
      unused{2 * nunused - 1} = args{i};  % 详解: 执行语句
      unused{2 * nunused} = args{i + 1};  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

if (~warn)  % 详解: 条件判断：if ((~warn))
  if (nunused)  % 详解: 条件判断：if ((nunused))
    varargout{nout} = unused;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    varargout{nout} = cell(0);  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




