% 文件: cwr_demo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Compare my code with
% http://www.media.mit.edu/physics/publications/books/nmm/files/index.html
%
% cwm.m
% (c) Neil Gershenfeld  9/1/97
% 1D Cluster-Weighted Modeling example
%
clear all  % 详解: 执行语句
figure;  % 详解: 执行语句
seed = 0;  % 详解: 赋值：计算表达式并保存到 seed
rand('state', seed);  % 详解: 调用函数：rand('state', seed)
randn('state', seed);  % 详解: 调用函数：randn('state', seed)
x = (-10:10)';  % 赋值：设置变量 x  % 详解: 赋值：计算表达式并保存到 x  % 详解: 赋值：计算表达式并保存到 x
y = (x > 0);  % 详解: 赋值：计算表达式并保存到 y
npts = length(x);  % 详解: 赋值：将 length(...) 的结果保存到 npts
plot(x,y,'+')  % 详解: 调用函数：plot(x,y,'+')
xlabel('x')  % 详解: 调用函数：xlabel('x')
ylabel('y')  % 详解: 调用函数：ylabel('y')
nclusters = 4;  % 详解: 赋值：计算表达式并保存到 nclusters
nplot = 100;  % 详解: 赋值：计算表达式并保存到 nplot
xplot = 24*(1:nplot)'/nplot - 12;  % 赋值：设置变量 xplot  % 详解: 赋值：计算表达式并保存到 xplot  % 详解: 赋值：计算表达式并保存到 xplot

mux = 20*rand(1,nclusters) - 10;  % 详解: 赋值：计算表达式并保存到 mux
muy = zeros(1,nclusters);  % 详解: 赋值：将 zeros(...) 的结果保存到 muy
varx = ones(1,nclusters);  % 详解: 赋值：将 ones(...) 的结果保存到 varx
vary = ones(1,nclusters);  % 详解: 赋值：将 ones(...) 的结果保存到 vary
pc = 1/nclusters * ones(1,nclusters);  % 详解: 赋值：计算表达式并保存到 pc
niterations = 5;  % 详解: 赋值：计算表达式并保存到 niterations
eps = 0.01;  % 详解: 赋值：计算表达式并保存到 eps
  

I = repmat(eye(1,1), [1 1 nclusters]);  % 详解: 赋值：将 repmat(...) 的结果保存到 I
O = repmat(zeros(1,1), [1 1 nclusters]);  % 详解: 赋值：将 repmat(...) 的结果保存到 O
X = x(:)';  % 赋值：设置变量 X  % 详解: 赋值：将 x(...) 的结果保存到 X  % 详解: 赋值：将 x(...) 的结果保存到 X
Y = y(:)';  % 赋值：设置变量 Y  % 详解: 赋值：将 y(...) 的结果保存到 Y  % 详解: 赋值：将 y(...) 的结果保存到 Y

cwr = cwr_em(X, Y, nclusters, 'muX', mux, 'muY', muy,  'SigmaX', I, ...  % 详解: 赋值：将 cwr_em(...) 的结果保存到 cwr
	     'cov_typeX', 'spherical', 'SigmaY', I, 'cov_typeY', 'spherical', ...  % 详解: 执行语句
	     'priorC', pc, 'weightsY', O,  'create_init_params', 0, ...  % 详解: 执行语句
	     'clamp_weights', 1, 'max_iter', niterations, ...  % 详解: 执行语句
	     'cov_priorX', eps*ones(1,1,nclusters), ...  % 详解: 创建全 1 矩阵/数组
	     'cov_priorY', eps*ones(1,1,nclusters));  % 详解: 创建全 1 矩阵/数组


for step = 1:niterations  % 详解: for 循环：迭代变量 step 遍历 1:niterations
    pplot = exp(-(kron(xplot,ones(1,nclusters)) ...  % 详解: 赋值：将 exp(...) 的结果保存到 pplot
		  - kron(ones(nplot,1),mux)).^2 ...  % 详解: 创建全 1 矩阵/数组
		./ (2*kron(ones(nplot,1),varx))) ...  % 详解: 创建全 1 矩阵/数组
	    ./ sqrt(2*pi*kron(ones(nplot,1),varx)) ...  % 详解: 创建全 1 矩阵/数组
	    .* kron(ones(nplot,1),pc);  % 详解: 创建全 1 矩阵/数组
    plot(xplot,pplot,'k');  % 详解: 调用函数：plot(xplot,pplot,'k')
    pause(0);  % 详解: 调用函数：pause(0)
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
    pc = sum(pp)/npts;  % 详解: 赋值：将 sum(...) 的结果保存到 pc
    yfit = sum(kron(ones(npts,1),muy) .* p,2) ...  % 详解: 赋值：将 sum(...) 的结果保存到 yfit
	   ./ sum(p,2);  % 详解: 统计：求和/均值/中位数
    mux = sum(kron(x,ones(1,nclusters)) .* pp) ...  % 详解: 赋值：将 sum(...) 的结果保存到 mux
	  ./ (npts*pc);  % 详解: 执行语句
    varx = eps + sum((kron(x,ones(1,nclusters)) ...  % 详解: 赋值：计算表达式并保存到 varx
		      - kron(ones(npts,1),mux)).^2 .* pp) ...  % 详解: 创建全 1 矩阵/数组
	   ./ (npts*pc);  % 详解: 执行语句
    muy = sum(kron(y,ones(1,nclusters)) .* pp) ...  % 详解: 赋值：将 sum(...) 的结果保存到 muy
	  ./ (npts*pc);  % 详解: 执行语句
    vary = eps + sum((kron(y,ones(1,nclusters)) ...  % 详解: 赋值：计算表达式并保存到 vary
		      - kron(ones(npts,1),muy)).^2 .* pp) ...  % 详解: 创建全 1 矩阵/数组
	   ./ (npts*pc);  % 详解: 执行语句
end  % 详解: 执行语句


cwr_pc = cwr.priorC';  % 赋值：设置变量 cwr_pc  % 详解: 赋值：计算表达式并保存到 cwr_pc  % 详解: 赋值：计算表达式并保存到 cwr_pc
assert(approxeq(cwr_pc, pc))  % 详解: 调用函数：assert(approxeq(cwr_pc, pc))
cwr_mux = cwr.muX;  % 详解: 赋值：计算表达式并保存到 cwr_mux
assert(approxeq(mux, cwr_mux))  % 详解: 调用函数：assert(approxeq(mux, cwr_mux))
cwr_SigmaX = squeeze(cwr.SigmaX)';  % 赋值：设置变量 cwr_SigmaX  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaX  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaX
assert(approxeq(varx, cwr_SigmaX))  % 详解: 调用函数：assert(approxeq(varx, cwr_SigmaX))
cwr_muy = cwr.muY;  % 详解: 赋值：计算表达式并保存到 cwr_muy
assert(approxeq(muy, cwr_muy))  % 详解: 调用函数：assert(approxeq(muy, cwr_muy))
cwr_SigmaY = squeeze(cwr.SigmaY)';  % 赋值：设置变量 cwr_SigmaY  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaY  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_SigmaY
assert(approxeq(vary, cwr_SigmaY))  % 详解: 调用函数：assert(approxeq(vary, cwr_SigmaY))



X = xplot(:)';  % 赋值：设置变量 X  % 详解: 赋值：将 xplot(...) 的结果保存到 X  % 详解: 赋值：将 xplot(...) 的结果保存到 X
[cwr_mu, Sigma, post] = cwr_predict(cwr, X);  % 详解: 执行语句
cwr_ystd = squeeze(Sigma)';  % 赋值：设置变量 cwr_ystd  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_ystd  % 详解: 赋值：将 squeeze(...) 的结果保存到 cwr_ystd

pplot = exp(-(kron(xplot,ones(1,nclusters)) ...  % 详解: 赋值：将 exp(...) 的结果保存到 pplot
   - kron(ones(nplot,1),mux)).^2 ...  % 详解: 创建全 1 矩阵/数组
   ./ (2*kron(ones(nplot,1),varx))) ...  % 详解: 创建全 1 矩阵/数组
   ./ sqrt(2*pi*kron(ones(nplot,1),varx)) ...  % 详解: 创建全 1 矩阵/数组
   .* kron(ones(nplot,1),pc);  % 详解: 创建全 1 矩阵/数组
yplot = sum(kron(ones(nplot,1),muy) .* pplot,2) ...  % 详解: 赋值：将 sum(...) 的结果保存到 yplot
   ./ sum(pplot,2);  % 详解: 统计：求和/均值/中位数
ystdplot = sum(kron(ones(nplot,1),(muy.^2+vary)) .* pplot,2) ...  % 详解: 赋值：将 sum(...) 的结果保存到 ystdplot
   ./ sum(pplot,2) - yplot.^2;  % 详解: 统计：求和/均值/中位数


assert(approxeq(yplot(:)', cwr_mu(:)'))  % 详解: 调用函数：assert(approxeq(yplot(:)', cwr_mu(:)'))
assert(approxeq(ystdplot, cwr_ystd))  % 详解: 调用函数：assert(approxeq(ystdplot, cwr_ystd))
assert(approxeq(pplot ./ repmat(sum(pplot,2), 1, nclusters),post') )  % 统计：求和/均值/中位数  % 详解: 调用函数：assert(approxeq(pplot ./ repmat(sum(pplot,2), 1, nclusters),post'))  % 详解: 调用函数：assert(approxeq(pplot ./ repmat(sum(pplot,2), 1, nclusters),post') ) % 统计：求和/均值/中位数 % 详解: 调用函数：assert(approxeq(pplot ./ repmat(sum(pplot,2), 1, nclusters),post'))

plot(xplot,yplot,'k');  % 详解: 调用函数：plot(xplot,yplot,'k')
hold on  % 详解: 执行语句
plot(xplot,yplot+ystdplot,'k--');  % 详解: 调用函数：plot(xplot,yplot+ystdplot,'k--')
plot(xplot,yplot-ystdplot,'k--');  % 详解: 调用函数：plot(xplot,yplot-ystdplot,'k--')
plot(x,y,'k+');  % 详解: 调用函数：plot(x,y,'k+')
axis([-12 12 -1 1.1]);  % 详解: 调用函数：axis([-12 12 -1 1.1])
plot(xplot,.8*pplot/max(max(pplot))-1,'k')  % 详解: 调用函数：plot(xplot,.8*pplot/max(max(pplot))-1,'k')
hold off  % 详解: 执行语句





