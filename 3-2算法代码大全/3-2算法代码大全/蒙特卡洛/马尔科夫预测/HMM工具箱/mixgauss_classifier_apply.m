% 文件: mixgauss_classifier_apply.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [classHatTest, probPos] = mixgauss_classifier_apply(mixgauss, testFeatures)  % 详解: 函数定义：mixgauss_classifier_apply(mixgauss, testFeatures), 返回：classHatTest, probPos

Bpos = mixgauss_prob(testFeatures, mixgauss.pos.mu, mixgauss.pos.Sigma, mixgauss.pos.prior);  % 详解: 赋值：将 mixgauss_prob(...) 的结果保存到 Bpos
Bneg = mixgauss_prob(testFeatures, mixgauss.neg.mu, mixgauss.neg.Sigma,  mixgauss.neg.prior);  % 详解: 赋值：将 mixgauss_prob(...) 的结果保存到 Bneg
prior_pos = mixgauss.priorC(1);  % 详解: 赋值：计算表达式并保存到 prior_pos
prior_neg = mixgauss.priorC(2);  % 详解: 赋值：计算表达式并保存到 prior_neg
post = normalize([Bpos * prior_pos; Bneg * prior_neg], 1);  % 详解: 赋值：将 normalize(...) 的结果保存到 post
probPos = post(1,:)';  % 赋值：设置变量 probPos  % 详解: 赋值：将 post(...) 的结果保存到 probPos  % 详解: 赋值：将 post(...) 的结果保存到 probPos
[junk, classHatTest] = max(post);  % 详解: 统计：最大/最小值
classHatTest(find(classHatTest==2))=0;  % 详解: 执行语句
classHatTest = classHatTest(:);  % 详解: 赋值：将 classHatTest(...) 的结果保存到 classHatTest




