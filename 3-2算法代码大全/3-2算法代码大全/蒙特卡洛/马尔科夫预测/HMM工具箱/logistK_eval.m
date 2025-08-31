% 文件: logistK_eval.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [post,lik,lli] = logistK_eval(beta,x,y)  % 详解: 函数定义：logistK_eval(beta,x,y), 返回：post,lik,lli


error(nargchk(2,3,nargin));  % 详解: 调用函数：error(nargchk(2,3,nargin))

if size(beta,1) ~= size(x,1),  % 详解: 条件判断：if (size(beta,1) ~= size(x,1),)
  error('Inputs beta,x not the same height.');  % 详解: 调用函数：error('Inputs beta,x not the same height.')
end  % 详解: 执行语句
if nargin > 3 & size(y,2) ~= size(x,2),  % 详解: 条件判断：if (nargin > 3 & size(y,2) ~= size(x,2),)
  error('Inputs x,y not the same length.');  % 详解: 调用函数：error('Inputs x,y not the same length.')
end  % 详解: 执行语句

[d,k] = size(beta);  % 详解: 获取向量/矩阵尺寸
[d,n] = size(x);  % 详解: 获取向量/矩阵尺寸

post = zeros(k,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 post
bx = zeros(k,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 bx
for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
  bx(j,:) = beta(:,j)'*x;   % 调用函数：bx  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句
for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
  post(j,:) = 1 ./ sum(exp(bx - repmat(bx(j,:),k,1)),1);  % 详解: 调用函数：post(j,:) = 1 ./ sum(exp(bx - repmat(bx(j,:),k,1)),1)
end  % 详解: 执行语句
clear bx;  % 详解: 执行语句

if nargout > 1,  % 详解: 条件判断：if (nargout > 1,)
  y = y ./ repmat(sum(y,1),k,1);  % 详解: 赋值：计算表达式并保存到 y
  lik = prod(post.^y,1);  % 详解: 赋值：将 prod(...) 的结果保存到 lik
end  % 详解: 执行语句

if nargout > 2,  % 详解: 条件判断：if (nargout > 2,)
  lli = sum(log(lik+eps));  % 详解: 赋值：将 sum(...) 的结果保存到 lli
end;  % 详解: 执行语句





