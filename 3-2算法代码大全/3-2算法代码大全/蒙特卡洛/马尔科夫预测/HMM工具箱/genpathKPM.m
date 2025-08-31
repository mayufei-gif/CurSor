% 文件: genpathKPM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = genpathKPM(d)  % 详解: 执行语句

if nargin==0,  % 详解: 条件判断：if (nargin==0,)
  p = genpath(fullfile(matlabroot,'toolbox'));  % 详解: 赋值：将 genpath(...) 的结果保存到 p
  if length(p) > 1, p(end) = []; end  % 详解: 条件判断：if (length(p) > 1, p(end) = []; end)
  return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

methodsep = '@';  % 详解: 赋值：计算表达式并保存到 methodsep
p = '';  % 详解: 赋值：计算表达式并保存到 p

files = dir(d);  % 详解: 赋值：将 dir(...) 的结果保存到 files
if isempty(files)  % 详解: 条件判断：if (isempty(files))
  return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

p = [p d pathsep];  % 详解: 赋值：计算表达式并保存到 p

isdir = logical(cat(1,files.isdir));  % 详解: 赋值：将 logical(...) 的结果保存到 isdir
dirs = files(isdir);  % 详解: 赋值：将 files(...) 的结果保存到 dirs

for i=1:length(dirs)  % 详解: for 循环：迭代变量 i 遍历 1:length(dirs)
   dirname = dirs(i).name;  % 详解: 赋值：将 dirs(...) 的结果保存到 dirname
   if    ~strcmp( dirname,'.')         & ...  % 详解: 条件判断：if (~strcmp( dirname,'.')         & ...)
         ~strcmp( dirname,'..')        & ...  % 详解: 执行语句
         ~strncmp( dirname,methodsep,1)& ...  % 详解: 执行语句
         ~strcmp( dirname,'private') & ...  % 详解: 执行语句
	 ~strcmp( dirname, 'old') & ...  % 详解: 执行语句
	 ~strcmp( dirname, 'Old') & ...  % 详解: 执行语句
     	 ~strcmp( dirname, 'CVS')  % 详解: 执行语句
      p = [p genpathKPM(fullfile(d,dirname))];  % 详解: 赋值：计算表达式并保存到 p
   end  % 详解: 执行语句
end  % 详解: 执行语句





