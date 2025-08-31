% 文件: zipsave.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%ZIPSAVE   Save data in compressed format
%
%	zipsave( filename, data )
%	filename: string variable that contains the name of the resulting 
%	compressed file (do not include '.zip' extension)
%	pkzip25.exe has to be in the matlab path. This file is a compression utility 
%	made by Pkware, Inc. It can be dowloaded from: http://www.pkware.com
%	This function was tested using 'PKZIP 2.50 Command Line for Windows 9x/NT'
%	It is important to use version 2.5 of the utility. Otherwise the command line below
%	has to be changed to include the proper options of the compression utility you 
%	wish to use.
%	This function was tested in MATLAB Version 5.3 under Windows NT.
%	Fernando A. Brucher - May/25/1999
%	
%	Example:
%		testData = [1 2 3; 4 5 6; 7 8 9];
%		zipsave('testfile', testData);
%
% Modified by Kevin Murphy, 26 Feb 2004, to use winzip
%------------------------------------------------------------------------

function zipsave( filename, data )  % 详解: 函数定义：zipsave(filename, data)


eval( ['save ''', filename, ''' data'] )  % 详解: 调用函数：eval(['save ''', filename, ''' data'])



eval( ['!zip ', filename, '.zip ', filename,'.mat'] )  % 详解: 调用函数：eval(['!zip ', filename, '.zip ', filename,'.mat'])


delete( [filename,'.mat'] )  % 详解: 调用函数：delete([filename,'.mat'])






