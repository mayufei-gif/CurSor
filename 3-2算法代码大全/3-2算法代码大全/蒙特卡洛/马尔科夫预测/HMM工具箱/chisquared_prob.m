% 文件: chisquared_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function P = chisquared_prob(X2,v)  % 详解: 执行语句




versn_str=version; eval(['versn=' versn_str(1) ';']);  % 详解: 赋值：计算表达式并保存到 versn_str
if versn<=3,  % 详解: 条件判断：if (versn<=3,)
 P = gamma(v/2, X2/2);  % 详解: 赋值：将 gamma(...) 的结果保存到 P
else  % 详解: 条件判断：else 分支
 P = gammainc(X2/2, v/2);  % 详解: 赋值：将 gammainc(...) 的结果保存到 P
end  % 详解: 执行语句




