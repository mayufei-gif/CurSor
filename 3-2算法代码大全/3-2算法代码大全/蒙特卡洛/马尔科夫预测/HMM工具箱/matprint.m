% 文件: matprint.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% MATPRINT - prints a matrix with specified format string
%
% Usage: matprint(a, fmt, fid)
%
%                 a   - Matrix to be printed.
%                 fmt - C style format string to use for each value.
%                 fid - Optional file id.
%
% Eg. matprint(a,'%3.1f') will print each entry to 1 decimal place

% Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk @ csse uwa edu au
% http://www.csse.uwa.edu.au/~pk
%
% March 2002  % 中文: 2002年3月|||为矩阵的每一行构建格式字符串，由|||组成编号规范的数字的“ cols”副本|||添加线供稿|||打印矩阵的转置，因为||| FPRINTF沿矩阵的列运行。

function matprint(a, fmt, fid)  % 详解: 函数定义：matprint(a, fmt, fid)
    
    if nargin < 3  % 详解: 条件判断：if (nargin < 3)
	fid = 1;  % 详解: 赋值：计算表达式并保存到 fid
    end  % 详解: 执行语句
    
    [rows,cols] = size(a);  % 详解: 获取向量/矩阵尺寸
    
    fmtstr = [];  % 详解: 赋值：计算表达式并保存到 fmtstr
    for c = 1:cols  % 详解: for 循环：迭代变量 c 遍历 1:cols
      fmtstr = [fmtstr, ' ', fmt];  % 详解: 赋值：计算表达式并保存到 fmtstr
    end  % 详解: 执行语句
    fmtstr = [fmtstr '\n'];  % 详解: 赋值：计算表达式并保存到 fmtstr
    
    fprintf(fid, fmtstr, a');  % Print the transpose of the matrix because  % 详解: 调用函数：fprintf(fid, fmtstr, a')  % 详解: 调用函数：fprintf(fid, fmtstr, a'); % Print the transpose of the matrix because % 详解: 调用函数：fprintf(fid, fmtstr, a')



