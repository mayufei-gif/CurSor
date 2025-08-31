% 文件: sqdist.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function m = sqdist(p, q, A)  % 详解: 执行语句


[d, pn] = size(p);  % 详解: 获取向量/矩阵尺寸
[d, qn] = size(q);  % 详解: 获取向量/矩阵尺寸

if nargin == 2  % 详解: 条件判断：if (nargin == 2)
  
  pmag = sum(p .* p, 1);  % 详解: 赋值：将 sum(...) 的结果保存到 pmag
  qmag = sum(q .* q, 1);  % 详解: 赋值：将 sum(...) 的结果保存到 qmag
  m = repmat(qmag, pn, 1) + repmat(pmag', 1, qn) - 2*p'*q;  % 详解: 赋值：将 repmat(...) 的结果保存到 m
  
else  % 详解: 条件判断：else 分支

  if isempty(A) | isempty(p)  % 详解: 条件判断：if (isempty(A) | isempty(p))
    error('sqdist: empty matrices');  % 详解: 调用函数：error('sqdist: empty matrices')
  end  % 详解: 执行语句
  Ap = A*p;  % 详解: 赋值：计算表达式并保存到 Ap
  Aq = A*q;  % 详解: 赋值：计算表达式并保存到 Aq
  pmag = sum(p .* Ap, 1);  % 详解: 赋值：将 sum(...) 的结果保存到 pmag
  qmag = sum(q .* Aq, 1);  % 详解: 赋值：将 sum(...) 的结果保存到 qmag
  m = repmat(qmag, pn, 1) + repmat(pmag', 1, qn) - 2*p'*Aq;  % 详解: 赋值：将 repmat(...) 的结果保存到 m
  
end  % 详解: 执行语句




