% 文件: cwr_em.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function cwr  = cwr_em(X, Y, nc, varargin)  % 详解: 执行语句

[max_iter, thresh, cov_typeX, cov_typeY, clamp_weights, ...  % 详解: 执行语句
 muX, muY, SigmaX, SigmaY, weightsY, priorC, create_init_params, ...  % 详解: 执行语句
cov_priorX, cov_priorY, verbose, regress, clamp_covX, clamp_covY] = process_options(...  % 详解: 执行语句
    varargin, 'max_iter', 10, 'thresh', 1e-2, 'cov_typeX', 'full', ...  % 详解: 执行语句
     'cov_typeY', 'full', 'clamp_weights', 0, ...  % 详解: 执行语句
     'muX', [], 'muY', [], 'SigmaX', [], 'SigmaY', [], 'weightsY', [], 'priorC', [], ...  % 详解: 执行语句
     'create_init_params', 1, 'cov_priorX', [], 'cov_priorY', [], 'verbose', 0, ...  % 详解: 执行语句
    'regress', 1, 'clamp_covX', 0, 'clamp_covY', 0);  % 详解: 执行语句
     
[nx N] = size(X);  % 详解: 获取向量/矩阵尺寸
[ny N2] = size(Y);  % 详解: 获取向量/矩阵尺寸
if N ~= N2  % 详解: 条件判断：if (N ~= N2)
  error(sprintf('nsamples X (%d) ~= nsamples Y (%d)', N, N2));  % 详解: 调用函数：error(sprintf('nsamples X (%d) ~= nsamples Y (%d)', N, N2))
end  % 详解: 执行语句
if (N < nx) & regress  % 详解: 条件判断：if ((N < nx) & regress)
  fprintf('cwr_em warning: dim X = %d, nsamples X = %d\n', nx, N);  % 详解: 调用函数：fprintf('cwr_em warning: dim X = %d, nsamples X = %d\n', nx, N)
end  % 详解: 执行语句
if (N < ny)  % 详解: 条件判断：if ((N < ny))
  fprintf('cwr_em warning: dim Y = %d, nsamples Y = %d\n', ny, N);  % 详解: 调用函数：fprintf('cwr_em warning: dim Y = %d, nsamples Y = %d\n', ny, N)
end  % 详解: 执行语句
if (nc > N)  % 详解: 条件判断：if ((nc > N))
  error(sprintf('cwr_em: more centers (%d) than data', nc))  % 详解: 调用函数：error(sprintf('cwr_em: more centers (%d) than data', nc))
end  % 详解: 执行语句

if nc==1  % 详解: 条件判断：if (nc==1)
  w = 1/N;  % 详解: 赋值：计算表达式并保存到 w
  WYbig = Y*w;  % 详解: 赋值：计算表达式并保存到 WYbig
  WYY = WYbig * Y';   % 赋值：设置变量 WYY  % 详解: 赋值：计算表达式并保存到 WYY  % 详解: 赋值：计算表达式并保存到 WYY
  WY = sum(WYbig, 2);  % 详解: 赋值：将 sum(...) 的结果保存到 WY
  WYTY = sum(diag(WYbig' * Y));  % 统计：求和/均值/中位数  % 详解: 赋值：将 sum(...) 的结果保存到 WYTY  % 详解: 赋值：将 sum(...) 的结果保存到 WYTY
  cwr.priorC = 1;  % 详解: 赋值：计算表达式并保存到 cwr.priorC
  cwr.SigmaX = [];  % 详解: 赋值：计算表达式并保存到 cwr.SigmaX
  if ~regress  % 详解: 条件判断：if (~regress)
    cwr.weightsY = [];  % 详解: 赋值：计算表达式并保存到 cwr.weightsY
    [cwr.muY, cwr.SigmaY] = ...  % 详解: 执行语句
	mixgauss_Mstep(1, WY, WYY, WYTY, ...  % 详解: 执行语句
		       'cov_type', cov_typeY, 'cov_prior', cov_priorY);  % 详解: 执行语句
    assert(approxeq(cwr.muY, mean(Y')))  % 统计：求和/均值/中位数  % 详解: 调用函数：assert(approxeq(cwr.muY, mean(Y')))  % 详解: 调用函数：assert(approxeq(cwr.muY, mean(Y'))) % 统计：求和/均值/中位数 % 详解: 调用函数：assert(approxeq(cwr.muY, mean(Y')))
    assert(approxeq(cwr.SigmaY, cov(Y') + 0.01*eye(ny)))  % 创建单位矩阵  % 详解: 调用函数：assert(approxeq(cwr.SigmaY, cov(Y') + 0.01*eye(ny)))  % 详解: 调用函数：assert(approxeq(cwr.SigmaY, cov(Y') + 0.01*eye(ny))) % 创建单位矩阵 % 详解: 调用函数：assert(approxeq(cwr.SigmaY, cov(Y') + 0.01*eye(ny)))
  else  % 详解: 条件判断：else 分支
    WXbig = X*w;  % 详解: 赋值：计算表达式并保存到 WXbig
    WXX = WXbig * X';  % 赋值：设置变量 WXX  % 详解: 赋值：计算表达式并保存到 WXX  % 详解: 赋值：计算表达式并保存到 WXX
   WX = sum(WXbig, 2);  % 详解: 赋值：将 sum(...) 的结果保存到 WX
    WXTX = sum(diag(WXbig' * X));  % 统计：求和/均值/中位数  % 详解: 赋值：将 sum(...) 的结果保存到 WXTX  % 详解: 赋值：将 sum(...) 的结果保存到 WXTX
    WXY = WXbig * Y';  % 赋值：设置变量 WXY  % 详解: 赋值：计算表达式并保存到 WXY  % 详解: 赋值：计算表达式并保存到 WXY
    [cwr.muY, cwr.SigmaY, cwr.weightsY] = ...  % 详解: 执行语句
	clg_Mstep(1, WY, WYY, WYTY, WX, WXX, WXY, ...  % 详解: 执行语句
		  'cov_type', cov_typeY, 'cov_prior', cov_priorY);  % 详解: 执行语句
  end  % 详解: 执行语句
  if clamp_covY, cwr.SigmaY = SigmaY; end  % 详解: 条件判断：if (clamp_covY, cwr.SigmaY = SigmaY; end)
  if clamp_weights,  cwr.weightsY = weightsY; end  % 详解: 条件判断：if (clamp_weights,  cwr.weightsY = weightsY; end)
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


if create_init_params  % 详解: 条件判断：if (create_init_params)
  [cwr.muX, cwr.SigmaX] = mixgauss_init(nc, X, cov_typeX);  % 详解: 执行语句
  [cwr.muY, cwr.SigmaY] = mixgauss_init(nc, Y, cov_typeY);  % 详解: 执行语句
  cwr.weightsY = zeros(ny, nx, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 cwr.weightsY
  cwr.priorC = normalize(ones(nc,1));  % 详解: 赋值：将 normalize(...) 的结果保存到 cwr.priorC
else  % 详解: 条件判断：else 分支
  cwr.muX = muX;  cwr.muY = muY; cwr.SigmaX = SigmaX; cwr.SigmaY = SigmaY;  % 详解: 赋值：计算表达式并保存到 cwr.muX
  cwr.weightsY = weightsY; cwr.priorC = priorC;  % 详解: 赋值：计算表达式并保存到 cwr.weightsY
end  % 详解: 执行语句


if clamp_covY, cwr.SigmaY = SigmaY; end  % 详解: 条件判断：if (clamp_covY, cwr.SigmaY = SigmaY; end)
if clamp_covX,  cwr.SigmaX = SigmaX; end  % 详解: 条件判断：if (clamp_covX,  cwr.SigmaX = SigmaX; end)
if clamp_weights,  cwr.weightsY = weightsY; end  % 详解: 条件判断：if (clamp_weights,  cwr.weightsY = weightsY; end)

previous_loglik = -inf;  % 详解: 赋值：计算表达式并保存到 previous_loglik
num_iter = 1;  % 详解: 赋值：计算表达式并保存到 num_iter
converged = 0;  % 详解: 赋值：计算表达式并保存到 converged

while (num_iter <= max_iter) & ~converged  % 详解: while 循环：当 ((num_iter <= max_iter) & ~converged) 为真时迭代

  
  [likXandY, likYgivenX, post] = cwr_prob(cwr, X, Y);  % 详解: 执行语句
  loglik = sum(log(likXandY));  % 详解: 赋值：将 sum(...) 的结果保存到 loglik
  w = sum(post,2);  % 详解: 赋值：将 sum(...) 的结果保存到 w
  WYY = zeros(ny, ny, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WYY
  WY = zeros(ny, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WY
  WYTY = zeros(nc,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 WYTY
  
  WXX = zeros(nx, nx, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WXX
  WX = zeros(nx, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WX
  WXTX = zeros(nc, 1);  % 详解: 赋值：将 zeros(...) 的结果保存到 WXTX
  WXY = zeros(nx,ny,nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WXY
  for c=1:nc  % 详解: for 循环：迭代变量 c 遍历 1:nc
    weights = repmat(post(c,:), ny, 1);  % 详解: 赋值：将 repmat(...) 的结果保存到 weights
    WYbig = Y .* weights;  % 详解: 赋值：计算表达式并保存到 WYbig
    WYY(:,:,c) = WYbig * Y';   % 调用函数：WYY  % 详解: 执行语句  % 详解: 执行语句
    WY(:,c) = sum(WYbig, 2);  % 详解: 调用函数：WY(:,c) = sum(WYbig, 2)
    WYTY(c) = sum(diag(WYbig' * Y));  % 统计：求和/均值/中位数  % 详解: 调用函数：WYTY(c) = sum(diag(WYbig' * Y))  % 详解: 调用函数：WYTY(c) = sum(diag(WYbig' * Y)); % 统计：求和/均值/中位数 % 详解: 调用函数：WYTY(c) = sum(diag(WYbig' * Y))

    weights = repmat(post(c,:), nx, 1);  % 详解: 赋值：将 repmat(...) 的结果保存到 weights
    WXbig = X .* weights;  % 详解: 赋值：计算表达式并保存到 WXbig
    WXX(:,:,c) = WXbig * X';  % 调用函数：WXX  % 详解: 执行语句  % 详解: 执行语句
    WX(:,c) = sum(WXbig, 2);  % 详解: 调用函数：WX(:,c) = sum(WXbig, 2)
    WXTX(c) = sum(diag(WXbig' * X));  % 统计：求和/均值/中位数  % 详解: 调用函数：WXTX(c) = sum(diag(WXbig' * X))  % 详解: 调用函数：WXTX(c) = sum(diag(WXbig' * X)); % 统计：求和/均值/中位数 % 详解: 调用函数：WXTX(c) = sum(diag(WXbig' * X))
    WXY(:,:,c) = WXbig * Y';  % 调用函数：WXY  % 详解: 执行语句  % 详解: 执行语句
  end  % 详解: 执行语句

  [cwr.muX, cwr.SigmaX] = mixgauss_Mstep(w, WX, WXX, WXTX, ...  % 详解: 执行语句
			    'cov_type', cov_typeX, 'cov_prior', cov_priorX);  % 详解: 执行语句
  for c=1:nc  % 详解: for 循环：迭代变量 c 遍历 1:nc
    assert(is_psd(cwr.SigmaX(:,:,c)))  % 详解: 调用函数：assert(is_psd(cwr.SigmaX(:,:,c)))
  end  % 详解: 执行语句
  
  if clamp_weights  % 详解: 条件判断：if (clamp_weights)
    W = cwr.weightsY;  % 详解: 赋值：计算表达式并保存到 W
  else  % 详解: 条件判断：else 分支
    W = [];  % 详解: 赋值：计算表达式并保存到 W
  end  % 详解: 执行语句
  [cwr.muY, cwr.SigmaY, cwr.weightsY] = ...  % 详解: 执行语句
      clg_Mstep(w, WY, WYY, WYTY, WX, WXX, WXY, ...  % 详解: 执行语句
		'cov_type', cov_typeY, 'clamped_weights', W, ...  % 详解: 执行语句
		'cov_prior', cov_priorY);  % 详解: 执行语句

  cwr.priorC = normalize(w);  % 详解: 赋值：将 normalize(...) 的结果保存到 cwr.priorC

  for c=1:nc  % 详解: for 循环：迭代变量 c 遍历 1:nc
    assert(is_psd(cwr.SigmaY(:,:,c)))  % 详解: 调用函数：assert(is_psd(cwr.SigmaY(:,:,c)))
  end  % 详解: 执行语句

  if clamp_covY, cwr.SigmaY = SigmaY; end  % 详解: 条件判断：if (clamp_covY, cwr.SigmaY = SigmaY; end)
  if clamp_covX,  cwr.SigmaX = SigmaX; end  % 详解: 条件判断：if (clamp_covX,  cwr.SigmaX = SigmaX; end)
  if clamp_weights,  cwr.weightsY = weightsY; end  % 详解: 条件判断：if (clamp_weights,  cwr.weightsY = weightsY; end)

  if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end  % 详解: 条件判断：if (verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end)
  num_iter =  num_iter + 1;  % 详解: 赋值：计算表达式并保存到 num_iter
  converged = em_converged(loglik, previous_loglik, thresh);  % 详解: 赋值：将 em_converged(...) 的结果保存到 converged
  previous_loglik = loglik;  % 详解: 赋值：计算表达式并保存到 previous_loglik
  
end  % 详解: 执行语句





