% 文件: plotgauss1d.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h = plotgauss1d(mu, sigma2)  % 详解: 执行语句

sigma = sqrt(sigma2);  % 详解: 赋值：将 sqrt(...) 的结果保存到 sigma
x = linspace(mu-3*sigma, mu+3*sigma, 100);  % 详解: 赋值：将 linspace(...) 的结果保存到 x
p = gaussian_prob(x, mu, sigma2);  % 详解: 赋值：将 gaussian_prob(...) 的结果保存到 p
h = plot(x, p, '-');  % 详解: 赋值：将 plot(...) 的结果保存到 h




