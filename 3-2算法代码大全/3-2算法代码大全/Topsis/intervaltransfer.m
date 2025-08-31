% 文件: intervaltransfer.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x2 =intervaltransfer(qujian,lb,ub,x)  % 详解: 执行语句
x2=(1-(qujian(1)-x)./(qujian(1)-lb)).*(x>=lb&x<qujian(1))+(x>=qujian(1)&x<=qujian(2))+(1-(x-qujian(2))./(ub-qujian(2))).*(x>qujian(2)&x<=ub);  % 详解: 赋值：计算表达式并保存到 x2

end  % 详解: 执行语句





