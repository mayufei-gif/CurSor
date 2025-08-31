% 文件: Edmonds.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [Matching,Cost] = Edmonds(a)  % 详解: 函数定义：Edmonds(a), 返回：Matching,Cost
Matching = zeros(size(a));  % 详解: 赋值：将 zeros(...) 的结果保存到 Matching
    num_y = sum(~isinf(a),1);  % 详解: 赋值：将 sum(...) 的结果保存到 num_y
    num_x = sum(~isinf(a),2);  % 详解: 赋值：将 sum(...) 的结果保存到 num_x
    x_con = find(num_x~=0);  % 详解: 赋值：将 find(...) 的结果保存到 x_con
    y_con = find(num_y~=0);  % 详解: 赋值：将 find(...) 的结果保存到 y_con
    P_size = max(length(x_con),length(y_con));  % 详解: 赋值：将 max(...) 的结果保存到 P_size
    P_cond = zeros(P_size);  % 详解: 赋值：将 zeros(...) 的结果保存到 P_cond
    P_cond(1:length(x_con),1:length(y_con)) = a(x_con,y_con);  % 详解: 调用函数：P_cond(1:length(x_con),1:length(y_con)) = a(x_con,y_con)
    if isempty(P_cond)  % 详解: 条件判断：if (isempty(P_cond))
      Cost = 0;  % 详解: 赋值：计算表达式并保存到 Cost
      return  % 详解: 返回：从当前函数返回
    end  % 详解: 执行语句
      Edge = P_cond;  % 详解: 赋值：计算表达式并保存到 Edge
      Edge(P_cond~=Inf) = 0;  % 详解: 执行语句
        cnum = min_line_cover(Edge);  % 详解: 赋值：将 min_line_cover(...) 的结果保存到 cnum
         Pmax = max(max(P_cond(P_cond~=Inf)));  % 详解: 赋值：将 max(...) 的结果保存到 Pmax
      P_size = length(P_cond)+cnum;  % 详解: 赋值：将 length(...) 的结果保存到 P_size
      P_cond = ones(P_size)*Pmax;  % 详解: 赋值：将 ones(...) 的结果保存到 P_cond
      P_cond(1:length(x_con),1:length(y_con)) = a(x_con,y_con);  % 详解: 调用函数：P_cond(1:length(x_con),1:length(y_con)) = a(x_con,y_con)
  exit_flag = 1;  % 详解: 赋值：计算表达式并保存到 exit_flag
  stepnum = 1;  % 详解: 赋值：计算表达式并保存到 stepnum
  while exit_flag  % 详解: while 循环：当 (exit_flag) 为真时迭代
    switch stepnum  % 详解: 多分支选择：switch (stepnum)
      case 1  % 详解: 分支：case 1
        [P_cond,stepnum] = step1(P_cond);  % 详解: 执行语句
      case 2  % 详解: 分支：case 2
        [r_cov,c_cov,M,stepnum] = step2(P_cond);  % 详解: 执行语句
      case 3  % 详解: 分支：case 3
        [c_cov,stepnum] = step3(M,P_size);  % 详解: 执行语句
      case 4  % 详解: 分支：case 4
        [M,r_cov,c_cov,Z_r,Z_c,stepnum] = step4(P_cond,r_cov,c_cov,M);  % 详解: 执行语句
      case 5  % 详解: 分支：case 5
        [M,r_cov,c_cov,stepnum] = step5(M,Z_r,Z_c,r_cov,c_cov);  % 详解: 执行语句
      case 6  % 详解: 分支：case 6
        [P_cond,stepnum] = step6(P_cond,r_cov,c_cov);  % 详解: 执行语句
      case 7  % 详解: 分支：case 7
        exit_flag = 0;  % 详解: 赋值：计算表达式并保存到 exit_flag
    end  % 详解: 执行语句
  end  % 详解: 执行语句
Matching(x_con,y_con) = M(1:length(x_con),1:length(y_con));  % 详解: 调用函数：Matching(x_con,y_con) = M(1:length(x_con),1:length(y_con))
Cost = sum(sum(a(Matching==1)));  % 详解: 赋值：将 sum(...) 的结果保存到 Cost
function [P_cond,stepnum] = step1(P_cond)  % 详解: 函数定义：step1(P_cond), 返回：P_cond,stepnum
  P_size = length(P_cond);  % 详解: 赋值：将 length(...) 的结果保存到 P_size
  for ii = 1:P_size  % 详解: for 循环：迭代变量 ii 遍历 1:P_size
    rmin = min(P_cond(ii,:));  % 详解: 赋值：将 min(...) 的结果保存到 rmin
    P_cond(ii,:) = P_cond(ii,:)-rmin;  % 详解: 执行语句
  end  % 详解: 执行语句
  stepnum = 2;  % 详解: 赋值：计算表达式并保存到 stepnum
function [r_cov,c_cov,M,stepnum] = step2(P_cond)  % 详解: 函数定义：step2(P_cond), 返回：r_cov,c_cov,M,stepnum
  P_size = length(P_cond);  % 详解: 赋值：将 length(...) 的结果保存到 P_size
  r_cov = zeros(P_size,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 r_cov
  c_cov = zeros(P_size,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 c_cov
  M = zeros(P_size);  % 详解: 赋值：将 zeros(...) 的结果保存到 M
  for ii = 1:P_size  % 详解: for 循环：迭代变量 ii 遍历 1:P_size
    for jj = 1:P_size  % 详解: for 循环：迭代变量 jj 遍历 1:P_size
      if P_cond(ii,jj) == 0 && r_cov(ii) == 0 && c_cov(jj) == 0  % 详解: 条件判断：if (P_cond(ii,jj) == 0 && r_cov(ii) == 0 && c_cov(jj) == 0)
        M(ii,jj) = 1;  % 详解: 执行语句
        r_cov(ii) = 1;  % 详解: 执行语句
        c_cov(jj) = 1;  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  r_cov = zeros(P_size,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 r_cov
  c_cov = zeros(P_size,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 c_cov
  stepnum = 3;  % 详解: 赋值：计算表达式并保存到 stepnum
  function [c_cov,stepnum] = step3(M,P_size)  % 详解: 函数定义：step3(M,P_size), 返回：c_cov,stepnum
  c_cov = sum(M,1);  % 详解: 赋值：将 sum(...) 的结果保存到 c_cov
  if sum(c_cov) == P_size  % 详解: 条件判断：if (sum(c_cov) == P_size)
    stepnum = 7;  % 详解: 赋值：计算表达式并保存到 stepnum
  else  % 详解: 条件判断：else 分支
    stepnum = 4;  % 详解: 赋值：计算表达式并保存到 stepnum
  end  % 详解: 执行语句
function [M,r_cov,c_cov,Z_r,Z_c,stepnum] = step4(P_cond,r_cov,c_cov,M)  % 详解: 函数定义：step4(P_cond,r_cov,c_cov,M), 返回：M,r_cov,c_cov,Z_r,Z_c,stepnum
P_size = length(P_cond);  % 详解: 赋值：将 length(...) 的结果保存到 P_size
zflag = 1;  % 详解: 赋值：计算表达式并保存到 zflag
while zflag  % 详解: while 循环：当 (zflag) 为真时迭代
       row = 0; col = 0; exit_flag = 1;  % 详解: 赋值：计算表达式并保存到 row
      ii = 1; jj = 1;  % 详解: 赋值：计算表达式并保存到 ii
      while exit_flag  % 详解: while 循环：当 (exit_flag) 为真时迭代
          if P_cond(ii,jj) == 0 && r_cov(ii) == 0 && c_cov(jj) == 0  % 详解: 条件判断：if (P_cond(ii,jj) == 0 && r_cov(ii) == 0 && c_cov(jj) == 0)
            row = ii;  % 详解: 赋值：计算表达式并保存到 row
            col = jj;  % 详解: 赋值：计算表达式并保存到 col
            exit_flag = 0;  % 详解: 赋值：计算表达式并保存到 exit_flag
          end  % 详解: 执行语句
          jj = jj + 1;  % 详解: 赋值：计算表达式并保存到 jj
          if jj > P_size; jj = 1; ii = ii+1; end  % 详解: 条件判断：if (jj > P_size; jj = 1; ii = ii+1; end)
          if ii > P_size; exit_flag = 0; end  % 详解: 条件判断：if (ii > P_size; exit_flag = 0; end)
      end  % 详解: 执行语句
       if row == 0  % 详解: 条件判断：if (row == 0)
        stepnum = 6;  % 详解: 赋值：计算表达式并保存到 stepnum
        zflag = 0;  % 详解: 赋值：计算表达式并保存到 zflag
        Z_r = 0;  % 详解: 赋值：计算表达式并保存到 Z_r
        Z_c = 0;  % 详解: 赋值：计算表达式并保存到 Z_c
      else  % 详解: 条件判断：else 分支
          M(row,col) = 2;  % 详解: 执行语句
                if sum(find(M(row,:)==1)) ~= 0  % 详解: 条件判断：if (sum(find(M(row,:)==1)) ~= 0)
            r_cov(row) = 1;  % 详解: 执行语句
            zcol = find(M(row,:)==1);  % 详解: 赋值：将 find(...) 的结果保存到 zcol
            c_cov(zcol) = 0;  % 详解: 执行语句
          else  % 详解: 条件判断：else 分支
            stepnum = 5;  % 详解: 赋值：计算表达式并保存到 stepnum
             zflag = 0;  % 详解: 赋值：计算表达式并保存到 zflag
            Z_r = row;  % 详解: 赋值：计算表达式并保存到 Z_r
            Z_c = col;  % 详解: 赋值：计算表达式并保存到 Z_c
          end  % 详解: 执行语句
      end  % 详解: 执行语句
end  % 详解: 执行语句
function [M,r_cov,c_cov,stepnum] = step5(M,Z_r,Z_c,r_cov,c_cov)  % 详解: 函数定义：step5(M,Z_r,Z_c,r_cov,c_cov), 返回：M,r_cov,c_cov,stepnum
  zflag = 1;  % 详解: 赋值：计算表达式并保存到 zflag
  ii = 1;  % 详解: 赋值：计算表达式并保存到 ii
  while zflag  % 详解: while 循环：当 (zflag) 为真时迭代
     rindex = find(M(:,Z_c(ii))==1);  % 详解: 赋值：将 find(...) 的结果保存到 rindex
    if rindex > 0  % 详解: 条件判断：if (rindex > 0)
          ii = ii+1;  % 详解: 赋值：计算表达式并保存到 ii
       Z_r(ii,1) = rindex;  % 详解: 执行语句
        Z_c(ii,1) = Z_c(ii-1);  % 详解: 调用函数：Z_c(ii,1) = Z_c(ii-1)
    else  % 详解: 条件判断：else 分支
      zflag = 0;  % 详解: 赋值：计算表达式并保存到 zflag
    end  % 详解: 执行语句
      if zflag == 1;  % 详解: 条件判断：if (zflag == 1;)
         cindex = find(M(Z_r(ii),:)==2);  % 详解: 赋值：将 find(...) 的结果保存到 cindex
      ii = ii+1;  % 详解: 赋值：计算表达式并保存到 ii
      Z_r(ii,1) = Z_r(ii-1);  % 详解: 调用函数：Z_r(ii,1) = Z_r(ii-1)
      Z_c(ii,1) = cindex;  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
   for ii = 1:length(Z_r)  % 详解: for 循环：迭代变量 ii 遍历 1:length(Z_r)
    if M(Z_r(ii),Z_c(ii)) == 1  % 详解: 条件判断：if (M(Z_r(ii),Z_c(ii)) == 1)
      M(Z_r(ii),Z_c(ii)) = 0;  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      M(Z_r(ii),Z_c(ii)) = 1;  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
   r_cov = r_cov.*0;  % 详解: 赋值：计算表达式并保存到 r_cov
  c_cov = c_cov.*0;  % 详解: 赋值：计算表达式并保存到 c_cov
  M(M==2) = 0;  % 详解: 执行语句
stepnum = 3;  % 详解: 赋值：计算表达式并保存到 stepnum
function [P_cond,stepnum] = step6(P_cond,r_cov,c_cov)  % 详解: 函数定义：step6(P_cond,r_cov,c_cov), 返回：P_cond,stepnum
a = find(r_cov == 0);  % 详解: 赋值：将 find(...) 的结果保存到 a
b = find(c_cov == 0);  % 详解: 赋值：将 find(...) 的结果保存到 b
minval = min(min(P_cond(a,b)));  % 详解: 赋值：将 min(...) 的结果保存到 minval
P_cond(find(r_cov == 1),:) = P_cond(find(r_cov == 1),:) + minval;  % 详解: 执行语句
P_cond(:,find(c_cov == 0)) = P_cond(:,find(c_cov == 0)) - minval;  % 详解: 执行语句
stepnum = 4;  % 详解: 赋值：计算表达式并保存到 stepnum
function cnum = min_line_cover(Edge)  % 详解: 执行语句
    [r_cov,c_cov,M,stepnum] = step2(Edge);  % 详解: 执行语句
     [c_cov,stepnum] = step3(M,length(Edge));  % 详解: 获取向量/矩阵尺寸
     [M,r_cov,c_cov,Z_r,Z_c,stepnum] = step4(Edge,r_cov,c_cov,M);  % 详解: 执行语句
     cnum = length(Edge)-sum(r_cov)-sum(c_cov);  % 详解: 赋值：将 length(...) 的结果保存到 cnum




