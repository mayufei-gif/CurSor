% 文件: cwr_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function  [likXandY, likYgivenX, post] = cwr_prob(cwr, X, Y);  % 详解: 函数定义：cwr_prob(cwr, X, Y), 返回：likXandY, likYgivenX, post

[nx N] = size(X);  % 详解: 获取向量/矩阵尺寸
nc = length(cwr.priorC);  % 详解: 赋值：将 length(...) 的结果保存到 nc

if nc == 1  % 详解: 条件判断：if (nc == 1)
  [mu, Sigma] = cwr_predict(cwr, X);  % 详解: 执行语句
  likY = gaussian_prob(Y, mu, Sigma);  % 详解: 赋值：将 gaussian_prob(...) 的结果保存到 likY
  likXandY = likY;  % 详解: 赋值：计算表达式并保存到 likXandY
  likYgivenX = likY;  % 详解: 赋值：计算表达式并保存到 likYgivenX
  post = ones(1,N);  % 详解: 赋值：将 ones(...) 的结果保存到 post
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


likY = clg_prob(X, Y, cwr.muY, cwr.SigmaY, cwr.weightsY);  % 详解: 赋值：将 clg_prob(...) 的结果保存到 likY

[junk, likX] = mixgauss_prob(X, cwr.muX, cwr.SigmaX);  % 详解: 执行语句
likX = squeeze(likX);  % 详解: 赋值：将 squeeze(...) 的结果保存到 likX

prior = repmat(cwr.priorC(:), 1, N);  % 详解: 赋值：将 repmat(...) 的结果保存到 prior

post = likX .* likY .* prior;  % 详解: 赋值：计算表达式并保存到 post
likXandY = sum(post, 1);  % 详解: 赋值：将 sum(...) 的结果保存到 likXandY
post = post ./ repmat(likXandY, nc, 1);  % 详解: 赋值：计算表达式并保存到 post

likX = sum(likX .* prior, 1);  % 详解: 赋值：将 sum(...) 的结果保存到 likX
likYgivenX = likXandY ./ likX;  % 详解: 赋值：计算表达式并保存到 likYgivenX




