% 文件: installC_KPMtools.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

mex ind2subv.c  % 详解: 执行语句
mex subv2ind.c  % 详解: 执行语句
mex normalise.c  % 详解: 执行语句
mex -c mexutil.c  % 详解: 执行语句
if ~isunix  % 详解: 条件判断：if (~isunix)
  mex repmatC.c mexutil.obj  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  mex repmatC.c mexutil.o  % 详解: 执行语句
end  % 详解: 执行语句




