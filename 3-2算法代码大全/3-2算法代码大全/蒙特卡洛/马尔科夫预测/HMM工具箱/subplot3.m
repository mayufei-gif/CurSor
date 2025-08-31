% 文件: subplot3.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function fignum = subplot3(nrows, ncols, fignumBase, plotnumBase)  % 详解: 执行语句

nplotsPerFig = nrows*ncols;  % 详解: 赋值：计算表达式并保存到 nplotsPerFig
fignum = fignumBase + div(plotnumBase-1, nplotsPerFig);  % 详解: 赋值：计算表达式并保存到 fignum
plotnum = wrap(plotnumBase, nplotsPerFig);  % 详解: 赋值：将 wrap(...) 的结果保存到 plotnum
figure(fignum);  % 详解: 调用函数：figure(fignum)
if plotnum==1, clf; end  % 详解: 条件判断：if (plotnum==1, clf; end)
subplot(nrows, ncols, plotnum);  % 详解: 调用函数：subplot(nrows, ncols, plotnum)






