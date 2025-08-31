% 文件: RAND.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%R=rand(m,n) 生成(0,1)上均匀分布的m行n列随机矩阵
%RAND   Uniformly distributed random numbers.
%   RAND(N) is an N-by-N matrix with random entries, chosen from
%   a uniform distribution on the interval (0.0,1.0).
%   RAND(M,N) and RAND([M,N]) are M-by-N matrices with random entries.
%   RAND(M,N,P,...) or RAND([M,N,P,...]) generate random arrays.
%   RAND with no arguments is a scalar whose value changes each time it
%   is referenced.  RAND(SIZE(A)) is the same size as A.  
%
%   RAND produces pseudo-random numbers.  The sequence of numbers
%   generated is determined by the state of the generator.  Since MATLAB
%   resets the state at start-up, the sequence of numbers generated will
%   be the same unless the state is changed.  % 中文: 除非国家更改，否则要保持相同。
%
%   S = RAND('state') is a 35-element vector containing the current state  % 中文: s = rand（'状态'）是一个35个元素向量，包含当前状态|||统一发电机。  兰德（“状态”，s）将状态重置为S. |||兰德（“状态”，0）将发电机重置为其初始状态。 |||兰德（'State'，j），用于整数J，将发电机重置为其J-th State。
%   of the uniform generator.  RAND('state',S) resets the state to S.  % 中文: Rand（'状态'，总和（100*时钟））每次将其重置为不同的状态。 |||该发电机可以生成|||中的所有浮点数闭合间隔[2^（-53），1-2^（-53）]。  从理论上讲，它可以生成|||在重复自我之前，超过2^1492值。
%   RAND('state',0) resets the generator to its initial state.  % 中文: MATLAB版本4.x使用带有单个种子的随机数生成器。 |||兰德（'seed'，0）和兰德（'seed'，j）导致使用MATLAB 4发电机。 |||兰德（“种子”）返回MATLAB 4均匀发生器的当前种子。 |||兰德（“状态”，j）和兰德（“状态”，s）导致使用MATLAB 5发电机。 |||另请参见Randn，Sprand，Sprandn，Randperm。
%   RAND('state',J), for integer J, resets the generator to its J-th state.  % 中文: $修订：5.8 $ $日期：1997/11/21 23:30:10 $
%   RAND('state',sum(100*clock)) resets it to a different state each time.  % 中文: L.J.Hu 1999年8月17日||| Randn正式分布随机数。
%
%   This generator can generate all the floating point numbers in the 
%   closed interval [2^(-53), 1-2^(-53)].  Theoretically, it can generate
%   over 2^1492 values before repeating itself.
%
%   MATLAB Version 4.x used random number generators with a single seed.
%   RAND('seed',0) and RAND('seed',J) cause the MATLAB 4 generator to be used.
%   RAND('seed') returns the current seed of the MATLAB 4 uniform generator.
%   RAND('state',J) and RAND('state',S) cause the MATLAB 5 generator to be used.
%
%   See also RANDN, SPRAND, SPRANDN, RANDPERM.

%   Copyright (c) 1984-98 by The MathWorks, Inc.  % 中文: Mathworks，Inc。版权（C）1984-98
%   $Revision: 5.8 $  $Date: 1997/11/21 23:30:10 $
%   Built-in function.


