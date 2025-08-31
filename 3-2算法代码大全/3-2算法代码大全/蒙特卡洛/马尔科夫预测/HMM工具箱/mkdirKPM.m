% 文件: mkdirKPM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function mkdirKPM(dname)  % 详解: 函数定义：mkdirKPM(dname)

[pathstr, name, ext, versn] = fileparts(dname);  % 详解: 执行语句
if ~isempty(ext)  % 详解: 条件判断：if (~isempty(ext))
  [pathstr, name, ext, versn] = fileparts(pathstr);  % 详解: 执行语句
  name = sprintf('%s%s', name, ext);  % 详解: 赋值：将 sprintf(...) 的结果保存到 name
end  % 详解: 执行语句

dname = fullfile(pathstr, name);  % 详解: 赋值：将 fullfile(...) 的结果保存到 dname
if ~exist(dname, 'dir')  % 详解: 条件判断：if (~exist(dname, 'dir'))
  mkdir(pathstr, name)  % 详解: 调用函数：mkdir(pathstr, name)
else  % 详解: 条件判断：else 分支
end  % 详解: 执行语句




