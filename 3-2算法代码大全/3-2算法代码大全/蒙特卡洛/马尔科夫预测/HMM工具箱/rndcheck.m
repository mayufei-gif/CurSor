% 文件: rndcheck.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [errorcode, rows, columns] = rndcheck(nargs,nparms,arg1,arg2,arg3,arg4,arg5)  % 详解: 函数定义：rndcheck(nargs,nparms,arg1,arg2,arg3,arg4,arg5), 返回：errorcode, rows, columns


sizeinfo = nargs - nparms;  % 详解: 赋值：计算表达式并保存到 sizeinfo
errorcode = 0;  % 详解: 赋值：计算表达式并保存到 errorcode

if nparms == 3  % 详解: 条件判断：if (nparms == 3)
    [r1 c1] = size(arg1);  % 详解: 获取向量/矩阵尺寸
    [r2 c2] = size(arg2);  % 详解: 获取向量/矩阵尺寸
    [r3 c3] = size(arg3);  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句

if nparms == 2  % 详解: 条件判断：if (nparms == 2)
    [r1 c1] = size(arg1);  % 详解: 获取向量/矩阵尺寸
    [r2 c2] = size(arg2);  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句

if sizeinfo == 0  % 详解: 条件判断：if (sizeinfo == 0)
    if nparms == 1  % 详解: 条件判断：if (nparms == 1)
        [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
    end  % 详解: 执行语句
    
    if nparms == 2  % 详解: 条件判断：if (nparms == 2)
        scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
        scalararg2 = (prod(size(arg2)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg2
        if ~scalararg1 & ~scalararg2  % 详解: 条件判断：if (~scalararg1 & ~scalararg2)
            if r1 ~= r2 | c1 ~= c2  % 详解: 条件判断：if (r1 ~= r2 | c1 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
        elseif ~scalararg2  % 详解: 条件判断：elseif (~scalararg2)
            [rows columns] = size(arg2);  % 详解: 获取向量/矩阵尺寸
        else  % 详解: 条件判断：else 分支
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    
    if nparms == 3  % 详解: 条件判断：if (nparms == 3)
        scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
        scalararg2 = (prod(size(arg2)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg2
        scalararg3 = (prod(size(arg3)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg3

        if ~scalararg1 & ~scalararg2  % 详解: 条件判断：if (~scalararg1 & ~scalararg2)
            if r1 ~= r2 | c1 ~= c2  % 详解: 条件判断：if (r1 ~= r2 | c1 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        if ~scalararg1 & ~scalararg3  % 详解: 条件判断：if (~scalararg1 & ~scalararg3)
            if r1 ~= r3 | c1 ~= c3  % 详解: 条件判断：if (r1 ~= r3 | c1 ~= c3)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        if ~scalararg3 & ~scalararg2  % 详解: 条件判断：if (~scalararg3 & ~scalararg2)
            if r3 ~= r2 | c3 ~= c2  % 详解: 条件判断：if (r3 ~= r2 | c3 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
            if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
                [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
            elseif ~scalararg2  % 详解: 条件判断：elseif (~scalararg2)
            [rows columns] = size(arg2);  % 详解: 获取向量/矩阵尺寸
            else  % 详解: 条件判断：else 分支
                [rows columns] = size(arg3);  % 详解: 获取向量/矩阵尺寸
            end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

if sizeinfo == 1  % 详解: 条件判断：if (sizeinfo == 1)
    scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
    if nparms == 1  % 详解: 条件判断：if (nparms == 1)
        if prod(size(arg2)) ~= 2  % 详解: 条件判断：if (prod(size(arg2)) ~= 2)
            errorcode = 2;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
        if  ~scalararg1 & arg2 ~= size(arg1)  % 详解: 条件判断：if (~scalararg1 & arg2 ~= size(arg1))
            errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
        if (arg2(1) < 0 | arg2(2) < 0 | arg2(1) ~= round(arg2(1)) | arg2(2) ~= round(arg2(2))),  % 详解: 条件判断：if ((arg2(1) < 0 | arg2(2) < 0 | arg2(1) ~= round(arg2(1)) | arg2(2) ~= round(arg2(2))),)
            errorcode = 4;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
        rows    = arg2(1);  % 详解: 赋值：将 arg2(...) 的结果保存到 rows
        columns = arg2(2);  % 详解: 赋值：将 arg2(...) 的结果保存到 columns
    end  % 详解: 执行语句
    
    if nparms == 2  % 详解: 条件判断：if (nparms == 2)
        if prod(size(arg3)) ~= 2  % 详解: 条件判断：if (prod(size(arg3)) ~= 2)
            errorcode = 2;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
        scalararg2 = (prod(size(arg2)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg2
        if ~scalararg1 & ~scalararg2  % 详解: 条件判断：if (~scalararg1 & ~scalararg2)
            if r1 ~= r2 | c1 ~= c2  % 详解: 条件判断：if (r1 ~= r2 | c1 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        if (arg3(1) < 0 | arg3(2) < 0 | arg3(1) ~= round(arg3(1)) | arg3(2) ~= round(arg3(2))),  % 详解: 条件判断：if ((arg3(1) < 0 | arg3(2) < 0 | arg3(1) ~= round(arg3(1)) | arg3(2) ~= round(arg3(2))),)
            errorcode = 4;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
        if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
            if any(arg3 ~= size(arg1))  % 详解: 条件判断：if (any(arg3 ~= size(arg1)))
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
        elseif ~scalararg2  % 详解: 条件判断：elseif (~scalararg2)
            if any(arg3 ~= size(arg2))  % 详解: 条件判断：if (any(arg3 ~= size(arg2)))
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            [rows columns] = size(arg2);  % 详解: 获取向量/矩阵尺寸
        else  % 详解: 条件判断：else 分支
            rows    = arg3(1);  % 详解: 赋值：将 arg3(...) 的结果保存到 rows
            columns = arg3(2);  % 详解: 赋值：将 arg3(...) 的结果保存到 columns
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    
    if nparms == 3  % 详解: 条件判断：if (nparms == 3)
        if prod(size(arg4)) ~= 2  % 详解: 条件判断：if (prod(size(arg4)) ~= 2)
            errorcode = 2;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
        scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
        scalararg2 = (prod(size(arg2)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg2
        scalararg3 = (prod(size(arg3)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg3

        if (arg4(1) < 0 | arg4(2) < 0 | arg4(1) ~= round(arg4(1)) | arg4(2) ~= round(arg4(2))),  % 详解: 条件判断：if ((arg4(1) < 0 | arg4(2) < 0 | arg4(1) ~= round(arg4(1)) | arg4(2) ~= round(arg4(2))),)
            errorcode = 4;  % 详解: 赋值：计算表达式并保存到 errorcode
            return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句

        if ~scalararg1 & ~scalararg2  % 详解: 条件判断：if (~scalararg1 & ~scalararg2)
            if r1 ~= r2 | c1 ~= c2  % 详解: 条件判断：if (r1 ~= r2 | c1 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        if ~scalararg1 & ~scalararg3  % 详解: 条件判断：if (~scalararg1 & ~scalararg3)
            if r1 ~= r3 | c1 ~= c3  % 详解: 条件判断：if (r1 ~= r3 | c1 ~= c3)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        if ~scalararg3 & ~scalararg2  % 详解: 条件判断：if (~scalararg3 & ~scalararg2)
            if r3 ~= r2 | c3 ~= c2  % 详解: 条件判断：if (r3 ~= r2 | c3 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
            if any(arg4 ~= size(arg1))  % 详解: 条件判断：if (any(arg4 ~= size(arg1)))
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
        elseif ~scalararg2  % 详解: 条件判断：elseif (~scalararg2)
            if any(arg4 ~= size(arg2))  % 详解: 条件判断：if (any(arg4 ~= size(arg2)))
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            [rows columns] = size(arg2);  % 详解: 获取向量/矩阵尺寸
        elseif ~scalararg3  % 详解: 条件判断：elseif (~scalararg3)
            if any(arg4 ~= size(arg3))  % 详解: 条件判断：if (any(arg4 ~= size(arg3)))
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            [rows columns] = size(arg3);  % 详解: 获取向量/矩阵尺寸
        else  % 详解: 条件判断：else 分支
            rows    = arg4(1);  % 详解: 赋值：将 arg4(...) 的结果保存到 rows
            columns = arg4(2);  % 详解: 赋值：将 arg4(...) 的结果保存到 columns
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

if sizeinfo == 2  % 详解: 条件判断：if (sizeinfo == 2)
    if nparms == 1  % 详解: 条件判断：if (nparms == 1)
        scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
        if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
            if rows ~= arg2 | columns ~= arg3  % 详解: 条件判断：if (rows ~= arg2 | columns ~= arg3)
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    if (arg2 < 0 | arg3 < 0 | arg2 ~= round(arg2) | arg3 ~= round(arg3)),  % 详解: 条件判断：if ((arg2 < 0 | arg3 < 0 | arg2 ~= round(arg2) | arg3 ~= round(arg3)),)
        errorcode = 4;  % 详解: 赋值：计算表达式并保存到 errorcode
        return;  % 详解: 返回：从当前函数返回
    end  % 详解: 执行语句
        rows = arg2;  % 详解: 赋值：计算表达式并保存到 rows
        columns = arg3;  % 详解: 赋值：计算表达式并保存到 columns
    end  % 详解: 执行语句
    
    if nparms == 2  % 详解: 条件判断：if (nparms == 2)
        scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
        scalararg2 = (prod(size(arg2)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg2
        if ~scalararg1 & ~scalararg2  % 详解: 条件判断：if (~scalararg1 & ~scalararg2)
            if r1 ~= r2 | c1 ~= c2  % 详解: 条件判断：if (r1 ~= r2 | c1 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
            if rows ~= arg3 | columns ~= arg4  % 详解: 条件判断：if (rows ~= arg3 | columns ~= arg4)
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        elseif ~scalararg2  % 详解: 条件判断：elseif (~scalararg2)
            [rows columns] = size(arg2);  % 详解: 获取向量/矩阵尺寸
            if rows ~= arg3 | columns ~= arg4  % 详解: 条件判断：if (rows ~= arg3 | columns ~= arg4)
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            if (arg3 < 0 | arg4 < 0 | arg3 ~= round(arg3) | arg4 ~= round(arg4)),  % 详解: 条件判断：if ((arg3 < 0 | arg4 < 0 | arg3 ~= round(arg3) | arg4 ~= round(arg4)),)
                errorcode = 4;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            rows = arg3;  % 详解: 赋值：计算表达式并保存到 rows
            columns = arg4;  % 详解: 赋值：计算表达式并保存到 columns
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    
    if nparms == 3  % 详解: 条件判断：if (nparms == 3)
        scalararg1 = (prod(size(arg1)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg1
        scalararg2 = (prod(size(arg2)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg2
        scalararg3 = (prod(size(arg3)) == 1);  % 详解: 赋值：计算表达式并保存到 scalararg3

        if ~scalararg1 & ~scalararg2  % 详解: 条件判断：if (~scalararg1 & ~scalararg2)
            if r1 ~= r2 | c1 ~= c2  % 详解: 条件判断：if (r1 ~= r2 | c1 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        if ~scalararg1 & ~scalararg3  % 详解: 条件判断：if (~scalararg1 & ~scalararg3)
            if r1 ~= r3 | c1 ~= c3  % 详解: 条件判断：if (r1 ~= r3 | c1 ~= c3)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        if ~scalararg3 & ~scalararg2  % 详解: 条件判断：if (~scalararg3 & ~scalararg2)
            if r3 ~= r2 | c3 ~= c2  % 详解: 条件判断：if (r3 ~= r2 | c3 ~= c2)
                errorcode = 1;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        
        if ~scalararg1  % 详解: 条件判断：if (~scalararg1)
            [rows columns] = size(arg1);  % 详解: 获取向量/矩阵尺寸
            if rows ~= arg4 | columns ~= arg5  % 详解: 条件判断：if (rows ~= arg4 | columns ~= arg5)
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        elseif ~scalararg2  % 详解: 条件判断：elseif (~scalararg2)
            [rows columns] = size(arg2);  % 详解: 获取向量/矩阵尺寸
            if rows ~= arg4 | columns ~= arg5  % 详解: 条件判断：if (rows ~= arg4 | columns ~= arg5)
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        elseif ~scalararg3  % 详解: 条件判断：elseif (~scalararg3)
            [rows columns] = size(arg3);  % 详解: 获取向量/矩阵尺寸
            if rows ~= arg4 | columns ~= arg5  % 详解: 条件判断：if (rows ~= arg4 | columns ~= arg5)
                errorcode = 3;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            if (arg4 < 0 | arg5 < 0 | arg4 ~= round(arg4) | arg5 ~= round(arg5)),  % 详解: 条件判断：if ((arg4 < 0 | arg5 < 0 | arg4 ~= round(arg4) | arg5 ~= round(arg5)),)
                errorcode = 4;  % 详解: 赋值：计算表达式并保存到 errorcode
                return;  % 详解: 返回：从当前函数返回
            end  % 详解: 执行语句
            rows    = arg4;  % 详解: 赋值：计算表达式并保存到 rows
            columns = arg5;  % 详解: 赋值：计算表达式并保存到 columns
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句




