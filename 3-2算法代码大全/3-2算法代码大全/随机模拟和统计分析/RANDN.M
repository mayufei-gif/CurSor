% 文件: RANDN.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%R=randn(m,n) 生成标准正态分布的m行n列随机矩阵
%RANDN  Normally distributed random numbers.
%   RANDN(N) is an N-by-N matrix with random entries, chosen from  % 中文: randn（n）是一个带有随机条目的n-n by-n矩阵，从|||选择平均零和方差的正态分布。
%   a normal distribution with mean zero and variance one.  % 中文: Randn（M，N）和Randn（[M，N]）是带有随机条目的M-N矩阵。
%   RANDN(M,N) and RANDN([M,N]) are M-by-N matrices with random entries.  % 中文: Randn（M，N，P，...）或Randn（[M，N，P ...]）生成随机数组。 |||没有参数的兰德是一个标量，其值每次|||都会改变其值。被引用。  randn（size（a））大小与A相同。|||兰德（Randn）产生伪随机数字。  数字序列||| s = randn（'状态'）是一个2元素向量，包含当前状态|||正常发电机。  Randn（'State'，s）将状态重置为S.
%   RANDN(M,N,P,...) or RANDN([M,N,P...]) generate random arrays.  % 中文: Randn（“状态”，0）将发电机重置为其初始状态。
%   RANDN with no arguments is a scalar whose value changes each time it  % 中文: randn（'state'，j），用于整数J，将发电机重置为其J-th State。
%   is referenced.  RANDN(SIZE(A)) is the same size as A.    % 中文: Randn（'状态'，总和（100*时钟））每次将其重置为不同的状态。
%
%   RANDN produces pseudo-random numbers.  The sequence of numbers  % 中文: Randn（'Seed'，0）和Randn（'Seed'，J）导致MATLAB 4 GENERATOR |||被使用。
%   generated is determined by the state of the generator.  Since MATLAB
%   resets the state at start-up, the sequence of numbers generated will
%   be the same unless the state is changed.  % 中文: 除非国家更改，否则要保持相同。
%
%   S = RANDN('state') is a 2-element vector containing the current state  % 中文: Randn（“种子”）返回MATLAB 4正常发生器的当前种子。
%   of the normal generator.  RANDN('state',S) resets the state to S.  % 中文: Randn（'State'，j）和Randn（“状态”，s）导致MATLAB 5 GENERATOR |||另请参见Rand，Sprand，Sprandn，Randperm。
%   RANDN('state',0) resets the generator to its initial state.
%   RANDN('state',J), for integer J, resets the generator to its J-th state.
%   RANDN('state',sum(100*clock)) resets it to a different state each time.
%
%   MATLAB Version 4.x used random number generators with a single seed.
%   RANDN('seed',0) and RANDN('seed',J) cause the MATLAB 4 generator
%   to be used.
%   RANDN('seed') returns the current seed of the MATLAB 4 normal generator.
%   RANDN('state',J) and RANDN('state',S) cause the MATLAB 5 generator
%   to be used. 
%
%   See also RAND, SPRAND, SPRANDN, RANDPERM.

%   Copyright (c) 1984-98 by The MathWorks, Inc.  % 中文: Mathworks，Inc。版权（C）1984-98
%   $Revision: 5.9 $  $Date: 1997/11/21 23:30:11 $  % 中文: $修订：5.9 $ $日期：1997/11/21 23:30:11 $
%   Built-in function.


