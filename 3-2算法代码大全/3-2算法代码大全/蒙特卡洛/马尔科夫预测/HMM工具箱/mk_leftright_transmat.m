% 文件: mk_leftright_transmat.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function transmat = mk_leftright_transmat(Q, p)  % 详解: 执行语句

transmat = p*diag(ones(Q,1)) + (1-p)*diag(ones(Q-1,1),1);  % 详解: 赋值：计算表达式并保存到 transmat
transmat(Q,Q)=1;  % 详解: 执行语句




