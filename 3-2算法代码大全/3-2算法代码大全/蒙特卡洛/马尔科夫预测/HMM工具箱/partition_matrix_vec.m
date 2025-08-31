% 文件: partition_matrix_vec.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [m1, m2, K11, K12, K21, K22] = partition_matrix_vec(m, K, n1, n2, bs)  % 详解: 函数定义：partition_matrix_vec(m, K, n1, n2, bs), 返回：m1, m2, K11, K12, K21, K22

dom = myunion(n1, n2);  % 详解: 赋值：将 myunion(...) 的结果保存到 dom
n1i = block(find_equiv_posns(n1, dom), bs(dom));  % 详解: 赋值：将 block(...) 的结果保存到 n1i
n2i = block(find_equiv_posns(n2, dom), bs(dom));  % 详解: 赋值：将 block(...) 的结果保存到 n2i
m1 = m(n1i);  % 详解: 赋值：将 m(...) 的结果保存到 m1
m2 = m(n2i);  % 详解: 赋值：将 m(...) 的结果保存到 m2
K11 = K(n1i, n1i);  % 详解: 赋值：将 K(...) 的结果保存到 K11
K12 = K(n1i, n2i);  % 详解: 赋值：将 K(...) 的结果保存到 K12
K21 = K(n2i, n1i);  % 详解: 赋值：将 K(...) 的结果保存到 K21
K22 = K(n2i, n2i);  % 详解: 赋值：将 K(...) 的结果保存到 K22




