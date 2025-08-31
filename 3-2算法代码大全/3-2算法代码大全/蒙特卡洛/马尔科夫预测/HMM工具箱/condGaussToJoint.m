% 文件: condGaussToJoint.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [muXY, SigmaXY] = condGaussToJoint(muX, SigmaX, muY, SigmaY, WYgivenX)  % 详解: 函数定义：condGaussToJoint(muX, SigmaX, muY, SigmaY, WYgivenX), 返回：muXY, SigmaXY



dx = length(muX);  % 详解: 赋值：将 length(...) 的结果保存到 dx
dy = length(muY);  % 详解: 赋值：将 length(...) 的结果保存到 dy
muXY = [muX(:); WYgivenX*muX(:) + muY];  % 详解: 赋值：计算表达式并保存到 muXY

W = [zeros(dx,dx) WYgivenX';  % 创建零矩阵/数组  % 详解: 赋值：计算表达式并保存到 W  % 详解: 赋值：计算表达式并保存到 W
     zeros(dy,dx) zeros(dy,dy)];  % 详解: 创建零矩阵/数组
D = [SigmaX       zeros(dx,dy);  % 详解: 赋值：计算表达式并保存到 D
     zeros(dy,dx) SigmaY];  % 详解: 创建零矩阵/数组

U = inv(eye(size(W)) - W')';  % 详解: 赋值：将 inv(...) 的结果保存到 U
SigmaXY = U' * D * U;  % 赋值：设置变量 SigmaXY  % 详解: 赋值：计算表达式并保存到 SigmaXY  % 详解: 赋值：计算表达式并保存到 SigmaXY






