% 文件: INTERP.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function yi=interp(x,y,xi,method)  % 详解: 执行语句


x=x(:);y=y(:);  % 详解: 赋值：将 x(...) 的结果保存到 x
z=[x,y];  % 详解: 赋值：计算表达式并保存到 z
z=sortrows(z);  % 详解: 赋值：将 sortrows(...) 的结果保存到 z
yi=interp1(z(:,1),z(:,2),xi,method);  % 详解: 赋值：将 interp1(...) 的结果保存到 yi






