% 文件: cwr_test.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Verify that my code gives the same results as the 1D example at
% http://www.media.mit.edu/physics/publications/books/nmm/files/cwm.m

seed = 0;  % 详解: 赋值：计算表达式并保存到 seed
rand('state', seed);  % 详解: 调用函数：rand('state', seed)
randn('state', seed);  % 详解: 调用函数：randn('state', seed)
x = (-10:10)';  % 赋值：设置变量 x  % 详解: 赋值：计算表达式并保存到 x  % 详解: 赋值：计算表达式并保存到 x
y = double(x > 0);  % 详解: 赋值：将 double(...) 的结果保存到 y
npts = length(x);  % 详解: 赋值：将 length(...) 的结果保存到 npts
plot(x,y,'+')  % 详解: 调用函数：plot(x,y,'+')

nclusters = 4;  % 详解: 赋值：计算表达式并保存到 nclusters
nplot = 100;  % 详解: 赋值：计算表达式并保存到 nplot
xplot = 24*(1:nplot)'/nplot - 12;  % 赋值：设置变量 xplot  % 详解: 赋值：计算表达式并保存到 xplot  % 详解: 赋值：计算表达式并保存到 xplot

mux = 20*rand(1,nclusters) - 10;  % 详解: 赋值：计算表达式并保存到 mux
muy = zeros(1,nclusters);  % 详解: 赋值：将 zeros(...) 的结果保存到 muy
varx = ones(1,nclusters);  % 详解: 赋值：将 ones(...) 的结果保存到 varx
vary = ones(1,nclusters);  % 详解: 赋值：将 ones(...) 的结果保存到 vary
pc = 1/nclusters * ones(1,nclusters);  % 详解: 赋值：计算表达式并保存到 pc


I = repmat(eye(1,1), [1 1 nclusters]);  % 详解: 赋值：将 repmat(...) 的结果保存到 I
O = repmat(zeros(1,1), [1 1 nclusters]);  % 详解: 赋值：将 repmat(...) 的结果保存到 O
X = x(:)';  % 赋值：设置变量 X  % 详解: 赋值：将 x(...) 的结果保存到 X  % 详解: 赋值：将 x(...) 的结果保存到 X
Y = y(:)';  % 赋值：设置变量 Y  % 详解: 赋值：将 y(...) 的结果保存到 Y  % 详解: 赋值：将 y(...) 的结果保存到 Y



cwr = cwr_em(X, Y, nclusters, 'muX', mux, 'muY', muy,  'SigmaX', I, 'cov_typeX', 'spherical', 'SigmaY', I, 'cov_typeY', 'spherical', 'priorC', pc, 'weightsY', O,  'create_init_params', 0, 'clamp_weights', 1, 'max_iter', 1);  % 详解: 赋值：将 cwr_em(...) 的结果保存到 cwr



px = exp(-(kron(x,ones(1,nclusters)) ...  % 详解: 赋值：将 exp(...) 的结果保存到 px
	   - kron(ones(npts,1),mux)).^2 ...  % 详解: 创建全 1 矩阵/数组
	 ./ (2*kron(ones(npts,1),varx))) ...  % 详解: 创建全 1 矩阵/数组
     ./ sqrt(2*pi*kron(ones(npts,1),varx));  % 详解: 创建全 1 矩阵/数组
py = exp(-(kron(y,ones(1,nclusters)) ...  % 详解: 赋值：将 exp(...) 的结果保存到 py
	   - kron(ones(npts,1),muy)).^2 ...  % 详解: 创建全 1 矩阵/数组
	 ./ (2*kron(ones(npts,1),vary))) ...  % 详解: 创建全 1 矩阵/数组
     ./ sqrt(2*pi*kron(ones(npts,1),vary));  % 详解: 创建全 1 矩阵/数组
p = px .* py .* kron(ones(npts,1),pc);  % 详解: 赋值：计算表达式并保存到 p
pp = p ./ kron(sum(p,2),ones(1,nclusters));  % 详解: 赋值：计算表达式并保存到 pp

eps = 0.01;  % 详解: 赋值：计算表达式并保存到 eps
pc2 = sum(pp)/npts;  % 详解: 赋值：将 sum(...) 的结果保存到 pc2

mux2 = sum(kron(x,ones(1,nclusters)) .* pp) ...  % 详解: 赋值：将 sum(...) 的结果保存到 mux2
      ./ (npts*pc2);  % 详解: 执行语句
varx2 = eps + sum((kron(x,ones(1,nclusters)) ...  % 详解: 赋值：计算表达式并保存到 varx2
		  - kron(ones(npts,1),mux2)).^2 .* pp) ...  % 详解: 创建全 1 矩阵/数组
       ./ (npts*pc2);  % 详解: 执行语句
muy2 = sum(kron(y,ones(1,nclusters)) .* pp) ...  % 详解: 赋值：将 sum(...) 的结果保存到 muy2
      ./ (npts*pc2);  % 详解: 执行语句
vary2 = eps + sum((kron(y,ones(1,nclusters)) ...  % 详解: 赋值：计算表达式并保存到 vary2
		  - kron(ones(npts,1),muy2)).^2 .* pp) ...  % 详解: 创建全 1 矩阵/数组
       ./ (npts*pc2);  % 详解: 执行语句


denom = (npts*pc2);  % 详解: 赋值：计算表达式并保存到 denom

cwr_mux = cwr.muX;  % 详解: 赋值：计算表达式并保存到 cwr_mux
assert(approxeq(mux2, cwr_mux))  % 详解: 调用函数：assert(approxeq(mux2, cwr_mux))
cwr_SigmaX = squeeze(cwr.SigmaX)';  % 赋值：设置变量 cwr_SigmaX  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaX  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaX
assert(approxeq(varx2, cwr_SigmaX))  % 详解: 调用函数：assert(approxeq(varx2, cwr_SigmaX))

cwr_muy = cwr.muY;  % 详解: 赋值：计算表达式并保存到 cwr_muy
assert(approxeq(muy2, cwr_muy))  % 详解: 调用函数：assert(approxeq(muy2, cwr_muy))
cwr_SigmaY = squeeze(cwr.SigmaY)';  % 赋值：设置变量 cwr_SigmaY  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaY  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaY
assert(approxeq(vary2, cwr_SigmaY))  % 详解: 调用函数：assert(approxeq(vary2, cwr_SigmaY))






