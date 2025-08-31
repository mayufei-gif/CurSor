% 文件: mixgauss_classifier_train.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function mixgauss = mixgauss_classifier_train(trainFeatures, trainLabels, nc, varargin)  % 详解: 执行语句

[testFeatures, testLabels, max_iter, thresh, cov_type, mu, Sigma, priorC, method, ...  % 详解: 执行语句
 cov_prior, verbose, prune_thresh] = process_options(...  % 详解: 执行语句
    varargin, 'testFeatures', [], 'testLabels', [], ...  % 详解: 执行语句
     'max_iter', 10, 'thresh', 0.01, 'cov_type', 'diag', ...  % 详解: 执行语句
    'mu', [], 'Sigma', [], 'priorC', [], 'method', 'kmeans', ...  % 详解: 执行语句
    'cov_prior', [], 'verbose', 0, 'prune_thresh', 0);  % 详解: 执行语句

Nclasses = 2;  % 详解: 赋值：计算表达式并保存到 Nclasses

pos = find(trainLabels == 1);  % 详解: 赋值：将 find(...) 的结果保存到 pos
neg = find(trainLabels == 0);  % 详解: 赋值：将 find(...) 的结果保存到 neg

if verbose, fprintf('fitting pos\n'); end  % 详解: 条件判断：if (verbose, fprintf('fitting pos\n'); end)
[mixgauss.pos.mu, mixgauss.pos.Sigma, mixgauss.pos.prior] = ...  % 详解: 执行语句
    mixgauss_em(trainFeatures(:, pos), nc, varargin{:});  % 详解: 调用函数：mixgauss_em(trainFeatures(:, pos), nc, varargin{:})

if verbose, fprintf('fitting neg\n'); end  % 详解: 条件判断：if (verbose, fprintf('fitting neg\n'); end)
[mixgauss.neg.mu, mixgauss.neg.Sigma, mixgauss.neg.prior] = ...  % 详解: 执行语句
    mixgauss_em(trainFeatures(:, neg), nc, varargin{:});  % 详解: 调用函数：mixgauss_em(trainFeatures(:, neg), nc, varargin{:})


if ~isempty(priorC)  % 详解: 条件判断：if (~isempty(priorC))
  mixgauss.priorC = priorC;  % 详解: 赋值：计算表达式并保存到 mixgauss.priorC
else  % 详解: 条件判断：else 分支
  mixgauss.priorC = normalize([length(pos) length(neg)]);  % 详解: 赋值：将 normalize(...) 的结果保存到 mixgauss.priorC
end  % 详解: 执行语句




