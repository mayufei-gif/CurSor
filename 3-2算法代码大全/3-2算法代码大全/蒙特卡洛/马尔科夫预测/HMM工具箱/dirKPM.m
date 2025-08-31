% 文件: dirKPM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function filenames = dirKPM(dirname, ext, varargin)  % 详解: 执行语句

if nargin < 1, dirname = '.'; end  % 详解: 条件判断：if (nargin < 1, dirname = '.'; end)

if nargin < 2, ext = ''; end  % 详解: 条件判断：if (nargin < 2, ext = ''; end)

[fileType, prepend, doSort, doRecurse, MAXDEPTH, DEPTH] = process_options(...  % 详解: 执行语句
	varargin, 'fileType', 'all', 'prepend', 0, 'doSort', 1, 'doRecurse', 0,...  % 详解: 执行语句
	'MAXDEPTH', 3, 'DEPTH', 0);  % 详解: 执行语句

tmp = dir(fullfile(dirname, ext));  % 详解: 赋值：将 dir(...) 的结果保存到 tmp
[filenames I] = setdiff({tmp.name}, {'.', '..'});  % 详解: 执行语句
tmp = tmp(I);  % 详解: 赋值：将 tmp(...) 的结果保存到 tmp

if doRecurse && sum([tmp.isdir])>0 && DEPTH<MAXDEPTH  % 详解: 条件判断：if (doRecurse && sum([tmp.isdir])>0 && DEPTH<MAXDEPTH)
	for fi=1:length(tmp)  % 详解: for 循环：迭代变量 fi 遍历 1:length(tmp)
		subDirFilenames = {};  % 详解: 赋值：计算表达式并保存到 subDirFilenames

		if tmp(fi).isdir  % 详解: 条件判断：if (tmp(fi).isdir)
			varargin = change_option( varargin, 'prepend', false );  % 详解: 赋值：将 change_option(...) 的结果保存到 varargin
			varargin = change_option( varargin, 'doSort', false );  % 详解: 赋值：将 change_option(...) 的结果保存到 varargin
			varargin = change_option( varargin, 'DEPTH', DEPTH+1 );  % 详解: 赋值：将 change_option(...) 的结果保存到 varargin
			subDirFilenames = dirKPM( fullfile(dirname,tmp(fi).name), ext, varargin{:} );  % 详解: 赋值：将 dirKPM(...) 的结果保存到 subDirFilenames

			for sdfi=1:length(subDirFilenames)  % 详解: for 循环：迭代变量 sdfi 遍历 1:length(subDirFilenames)
				subDirFilenames{sdfi} = fullfile(tmp(fi).name, subDirFilenames{sdfi});  % 详解: 执行语句
			end  % 详解: 执行语句
		end  % 详解: 执行语句


		nfilenames = length(filenames);  % 详解: 赋值：将 length(...) 的结果保存到 nfilenames
		if length(subDirFilenames)>0  % 详解: 条件判断：if (length(subDirFilenames)>0)
			filenames(nfilenames+1:nfilenames+length(subDirFilenames)) = subDirFilenames;  % 详解: 获取向量/矩阵尺寸
		end  % 详解: 执行语句
	end  % 详解: 执行语句
end  % 详解: 执行语句

nfiles = length(filenames);  % 详解: 赋值：将 length(...) 的结果保存到 nfiles
if nfiles==0 return; end  % 详解: 条件判断：if (nfiles==0 return; end)

switch fileType  % 详解: 多分支选择：switch (fileType)
	case 'image',  % 详解: 分支：case 'image',
		for fi=1:nfiles  % 详解: for 循环：迭代变量 fi 遍历 1:nfiles
			good(fi) = isImage(filenames{fi});  % 详解: 调用函数：good(fi) = isImage(filenames{fi})
		end  % 详解: 执行语句
		filenames = filenames(find(good));  % 详解: 赋值：将 filenames(...) 的结果保存到 filenames
	case 'all',  % 详解: 分支：case 'all',
	otherwise  % 详解: 默认分支：otherwise
		error(sprintf('unrecognized file type %s', fileType));  % 详解: 调用函数：error(sprintf('unrecognized file type %s', fileType))
end  % 详解: 执行语句

if doSort  % 详解: 条件判断：if (doSort)

	filenames=sort(filenames);  % 详解: 赋值：将 sort(...) 的结果保存到 filenames
	
end  % 详解: 执行语句


if prepend  % 详解: 条件判断：if (prepend)
	nfiles = length(filenames);  % 详解: 赋值：将 length(...) 的结果保存到 nfiles
	for fi=1:nfiles  % 详解: for 循环：迭代变量 fi 遍历 1:nfiles
		filenames{fi} = fullfile(dirname, filenames{fi});  % 详解: 执行语句
	end  % 详解: 执行语句
end  % 详解: 执行语句





