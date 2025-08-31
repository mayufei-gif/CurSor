% 文件: subplot2.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function subplot2(nrows, ncols, i, j)  % 详解: 函数定义：subplot2(nrows, ncols, i, j)


sz = [nrows ncols];  % 详解: 赋值：计算表达式并保存到 sz
k = sub2ind(sz(end:-1:1), j, i);  % 详解: 赋值：将 sub2ind(...) 的结果保存到 k
subplot(nrows, ncols, k);  % 详解: 调用函数：subplot(nrows, ncols, k)

if 0  % 详解: 条件判断：if (0)
  ncols_plot = ceil(sqrt(Nplots));  % 详解: 赋值：将 ceil(...) 的结果保存到 ncols_plot
  nrows_plot = ceil(Nplots/ncols_plot);  % 详解: 赋值：将 ceil(...) 的结果保存到 nrows_plot
  Nplots = nrows_plot*ncols_plot;  % 详解: 赋值：计算表达式并保存到 Nplots
  for p=1:Nplots  % 详解: for 循环：迭代变量 p 遍历 1:Nplots
    subplot(nrows_plot, ncols_plot, p);  % 详解: 调用函数：subplot(nrows_plot, ncols_plot, p)
  end  % 详解: 执行语句
end  % 详解: 执行语句




