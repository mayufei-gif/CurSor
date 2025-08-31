% 文件: zipload.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%ZIPLOAD   Load compressed data file created with ZIPSAVE
%
%	[data] = zipload( filename )
%	filename: string variable that contains the name of the 
%	compressed file (do not include '.zip' extension)
%	Use only with files created with 'zipsave'
%	pkzip25.exe has to be in the matlab path. This file is a compression utility 
%	made by Pkware, Inc. It can be dowloaded from:	http://www.pkware.com
%	Or directly from ftp://ftp.pkware.com/pk250c32.exe, for the Windows 95/NT version.
%	This function was tested using 'PKZIP 2.50 Command Line for Windows 9x/NT'
%	It is important to use version 2.5 of the utility. Otherwise the command line below
%	has to be changed to include the proper options of the compression utility you 
%	wish to use.
%	This function was tested in MATLAB Version 5.3 under Windows NT.
%	Fernando A. Brucher - May/25/1999
%	
%	Example:
%		[loadedData] = zipload('testfile');
%--------------------------------------------------------------------

function [data] = zipload( filename )  % 详解: 函数定义：zipload(filename), 返回：data


eval( ['!pkzip25 -extract -silent -over=all ', filename, '.zip'] )  % 详解: 调用函数：eval(['!pkzip25 -extract -silent -over=all ', filename, '.zip'])



try  % 详解: 异常处理：try 块开始
   tmpStruc = load( filename );  % 详解: 赋值：将 load(...) 的结果保存到 tmpStruc
   data = tmpStruc.data;  % 详解: 赋值：计算表达式并保存到 data
catch, return, end  % 详解: 异常处理：catch 捕获变量 , return, end



delete( [filename,'.mat'] )  % 详解: 调用函数：delete([filename,'.mat'])






