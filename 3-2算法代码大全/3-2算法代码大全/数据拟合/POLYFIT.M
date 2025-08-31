% 文件: POLYFIT.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [p,S] = polyfit(x,y,n)  % 详解: 函数定义：polyfit(x,y,n), 返回：p,S




if ~isequal(size(x),size(y))  % 详解: 条件判断：if (~isequal(size(x),size(y)))
    error('X and Y vectors must be the same size.')  % 详解: 调用函数：error('X and Y vectors must be the same size.')
end  % 详解: 执行语句

x = x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x
y = y(:);  % 详解: 赋值：将 y(...) 的结果保存到 y

V(:,n+1) = ones(length(x),1);  % 详解: 调用函数：V(:,n+1) = ones(length(x),1)
for j = n:-1:1  % 详解: for 循环：迭代变量 j 遍历 n:-1:1
    V(:,j) = x.*V(:,j+1);  % 详解: 调用函数：V(:,j) = x.*V(:,j+1)
end  % 详解: 执行语句

[Q,R] = qr(V,0);  % 详解: 执行语句
p = R\(Q'*y);    % Same as p = V\y;  % 中文: x = [1 2 4 5]; y = [1 3 4 2];  % 详解: 赋值：计算表达式并保存到 p  % 详解: 赋值：计算表达式并保存到 p
r = y - V*p;  % 详解: 赋值：计算表达式并保存到 r
p = p.';          % Polynomial coefficients are row vectors by convention.  % 中文: p =样条（x，y）;  % 详解: 赋值：计算表达式并保存到 p  % 详解: 赋值：计算表达式并保存到 p


S.R = R;  % 详解: 赋值：计算表达式并保存到 S.R
S.df = length(y) - (n+1);  % 详解: 赋值：将 length(...) 的结果保存到 S.df
S.normr = norm(r);  % 详解: 赋值：将 norm(...) 的结果保存到 S.normr




