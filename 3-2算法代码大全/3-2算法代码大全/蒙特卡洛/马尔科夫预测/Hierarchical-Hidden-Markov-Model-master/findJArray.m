% 文件: findJArray.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function jArr = findJArray(q,d,i)  % 详解: 执行语句
    jArr = find(q(d,:,1)==2);  % 详解: 赋值：将 find(...) 的结果保存到 jArr
    preJ = find(jArr<i);  % 详解: 赋值：将 find(...) 的结果保存到 preJ
    if isempty(preJ)==1  % 详解: 条件判断：if (isempty(preJ)==1)
        preJ(1) = 0;  % 详解: 执行语句
        postJ= find(jArr>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 postJ
        jArr = (preJ(end))+1:jArr(postJ);  % 详解: 赋值：计算表达式并保存到 jArr
    else  % 详解: 条件判断：else 分支
        postJ= find(jArr>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 postJ
        jArr = jArr(preJ(end))+1:jArr(postJ);  % 详解: 赋值：将 jArr(...) 的结果保存到 jArr
    end  % 详解: 执行语句
end  % 详解: 执行语句



