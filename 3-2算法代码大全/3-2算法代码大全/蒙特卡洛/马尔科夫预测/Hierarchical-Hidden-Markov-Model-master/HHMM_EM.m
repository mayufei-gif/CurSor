% 文件: HHMM_EM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [PI_new, A_new, B_new P] = HHMM_EM(q, seq, A, PI, B, alph, Palt)  % 详解: 函数定义：HHMM_EM(q, seq, A, PI, B, alph, Palt), 返回：PI_new, A_new, B_new P




    A = log(A);  % 详解: 赋值：将 log(...) 的结果保存到 A
    B = log(B);  % 详解: 赋值：将 log(...) 的结果保存到 B
    PI = log(PI);  % 详解: 赋值：将 log(...) 的结果保存到 PI
    
    [alpha beta ] = expectationAlphaBetaLog(A, PI, q, B, seq);  % 详解: 执行语句
    P = sum(exp(alpha(1,length(seq),2,:,1,1)));  % 详解: 赋值：将 sum(...) 的结果保存到 P
    
    [eta_in eta_out ] = expectationEtaLog(A, PI, q, alpha, beta, seq);  % 详解: 执行语句
    
    [xi chi gamma_in gamma_out ] = expectationXiChiLog(A, PI, q, eta_in, eta_out, alpha, beta, seq);  % 详解: 执行语句

    [PI_new A_new B_new] = estimationLog(q, xi, chi, gamma_in, gamma_out, seq, B, alph);  % 详解: 执行语句
end  % 详解: 执行语句




