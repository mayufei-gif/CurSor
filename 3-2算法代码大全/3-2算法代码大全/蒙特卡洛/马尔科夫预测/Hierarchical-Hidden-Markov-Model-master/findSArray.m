% 文件: findSArray.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function sArr = findSArray(q,d,i)  % 详解: 执行语句
    if i == 1  % 详解: 条件判断：if (i == 1)
        sArr = 1:cumsum(q(d,1:i,2));  % 详解: 赋值：计算表达式并保存到 sArr
    else  % 详解: 条件判断：else 分支
        start = cumsum(q(d,1:(i-1),2));  % 详解: 赋值：将 cumsum(...) 的结果保存到 start
        ende = cumsum(q(d,1:i,2));  % 详解: 赋值：将 cumsum(...) 的结果保存到 ende
        sArr = start(end)+1:ende(end);  % 详解: 赋值：将 start(...) 的结果保存到 sArr
    end  % 详解: 执行语句
end  % 详解: 执行语句



