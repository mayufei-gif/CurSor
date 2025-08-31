% 文件: mkPolyFvec.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = mkPolyFvec(x)  % 详解: 执行语句

fvec = x;  % 详解: 赋值：计算表达式并保存到 fvec
fvecSq = x.*x;  % 详解: 赋值：计算表达式并保存到 fvecSq
[D N] = size(x);  % 详解: 获取向量/矩阵尺寸
fvecCross = zeros(D*(D-1)/2, N);  % 详解: 赋值：将 zeros(...) 的结果保存到 fvecCross
i = 1;  % 详解: 赋值：计算表达式并保存到 i
for d=1:D  % 详解: for 循环：迭代变量 d 遍历 1:D
  for d2=d+1:D  % 详解: for 循环：迭代变量 d2 遍历 d+1:D
    fvecCross(i,:) = x(d,:) .* x(d2,:);  % 详解: 调用函数：fvecCross(i,:) = x(d,:) .* x(d2,:)
    i = i + 1;  % 详解: 赋值：计算表达式并保存到 i
  end  % 详解: 执行语句
end  % 详解: 执行语句
p = [fvec; fvecSq; fvecCross];  % 详解: 赋值：计算表达式并保存到 p




