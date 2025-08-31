% 文件: compresstable2matrix.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function A=compresstable2matrix(b)  % 详解: 执行语句
    [n ~]=size(b);  % 详解: 获取向量/矩阵尺寸
    m=max(b(:));  % 详解: 赋值：将 max(...) 的结果保存到 m
    A=zeros(m,m);  % 详解: 赋值：将 zeros(...) 的结果保存到 A

    for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
        A(b(i,1),b(i,2))=1;  % 详解: 执行语句
        A(b(i,2),b(i,1))=1;  % 详解: 执行语句
    end  % 详解: 执行语句

end  % 详解: 执行语句



