% 文件: asort.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%[ANR,SNR,STR]	=  ASORT(INP,'OPT',...);
% S		=  ASORT(INP,'OPT',...);
%		   to sort alphanumeric strings numerically if
%		   they contain one properly formatted number
%		   otherwise, ascii dictionary sorting is applied
%
% INP	unsorted input:
%	- a char array
%	- a cell array of strings
% OPT	options
%  -s	- sorting option
%	  '-s','ascend'					[def]
%	  '-s','descend'
%  -st	- force output form S				[def: nargout dependent]
%  -t	- replace matching template(s) with one space
%	  prior to sorting
%	  '-t','template'
%	  '-t',{'template1','template2',...}
%  -w	- remove space(s) prior to sorting
%
%	  NOTE	-t/-w options are processed in the
%		      order that they appear in
%		      the command line
%
%  -v	- verbose output				[def: quiet]
%  -d	- debug mode
%	  save additional output in S
%	  .c:	lex parser input
%	  .t:	lex parser table  % 中文: .t：Lex Parser表||| .N：LEX解析器输出||| .D：数字从.N |||读取ANR数字排序的字母数字字符串[例如，'F.-1.5e+2x.x']
%	  .n:	lex parser output  % 中文: - 包含一个可以通过|||读取的数字<strread> | <SSCANF>
%	  .d:	numbers read from .n  % 中文: snr ascii dict排序字母数字||| http://www.mathworks.com/matlabcentral/fileexchange/loadfile.do?objectID=7212#
%
% ANR	numerically sorted alphanumeric strings		[eg, 'f.-1.5e+2x.x']  % 中文: - 包含多个数字[例如'F.-1.5e +2.x']
%	- contain one number that can be read by  % 中文: - 包含不完整的|模棱两可的数字[例如，'F.-1.5e+2.x']
%	  <strread> | <sscanf>  % 中文: str ascii dict排序字符串||| - 不包含数字[例如，'a test']
% SNR	ascii dict  sorted alphanumeric strings  % 中文: S结构带有字段||| .anr
% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=7212#  % 中文: .srn
%
%	- contain more than one number			[eg, 'f.-1.5e +2.x']  % 中文: .str |||创建：||| US 03-MAR-2002 |||修改：||| US 30-MAR-2005 11:57:07 / TMW R14.SP2
%	- contain incomplete|ambiguous numbers		[eg, 'f.-1.5e+2.x']  % 中文: - 常见参数/选项|||排序选项|||模板|||输出模式：结构|||删除模板|||调试模式|||冗长输出|||删除空间|||规格号|||规格字符|||小数点|||指数||| -1d/％-1d] \ n'，nargout，3））;
% STR	ascii dict  sorted strings  % 中文: [ins，inx] = sortrows（inp）;
%	- contain no numbers				[eg, 'a test']
%
% S	structure with fields
%	.anr
%	.srn
%	.str

% created:
%	us	03-Mar-2002
% modified:
%	us	30-Mar-2005 11:57:07 	/ TMW R14.sp2

%--------------------------------------------------------------------------------
function	varargout=asort(inp,varargin)  % 详解: 执行语句

varargout(1:nargout)={[]};  % 详解: 执行语句
if	~nargin  % 详解: 条件判断：if (~nargin)
	help(mfilename);  % 详解: 调用函数：help(mfilename)
	return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

n=[];  % 详解: 赋值：计算表达式并保存到 n
ds=[];  % 详解: 赋值：计算表达式并保存到 ds
anr={};  % 详解: 赋值：计算表达式并保存到 anr
snr={};  % 详解: 赋值：计算表达式并保存到 snr
str={};  % 详解: 赋值：计算表达式并保存到 str
smod='ascend';  % 详解: 赋值：计算表达式并保存到 smod
tmpl={};  % 详解: 赋值：计算表达式并保存到 tmpl
sflg=false;  % 详解: 赋值：计算表达式并保存到 sflg
tflg=false;  % 详解: 赋值：计算表达式并保存到 tflg
dflg=false;  % 详解: 赋值：计算表达式并保存到 dflg
vflg=false;  % 详解: 赋值：计算表达式并保存到 vflg
wflg=false;  % 详解: 赋值：计算表达式并保存到 wflg

if	nargin > 1  % 详解: 条件判断：if (nargin > 1)
	ix=find(strcmp('-s',varargin));  % 详解: 赋值：将 find(...) 的结果保存到 ix
	if	~isempty(ix) && nargin > ix(end)+1  % 详解: 条件判断：if (~isempty(ix) && nargin > ix(end)+1)
		smod=varargin{ix(end)+1};  % 详解: 赋值：计算表达式并保存到 smod
	end  % 详解: 执行语句
	ix=find(strcmp('-t',varargin));  % 详解: 赋值：将 find(...) 的结果保存到 ix
	if	~isempty(ix) && nargin > ix(end)+1  % 详解: 条件判断：if (~isempty(ix) && nargin > ix(end)+1)
		tflg=ix(end);  % 详解: 赋值：将 ix(...) 的结果保存到 tflg
		tmpl=varargin{ix(end)+1};  % 详解: 赋值：计算表达式并保存到 tmpl
	end  % 详解: 执行语句
	if	find(strcmp('-d',varargin));  % 详解: 条件判断：if (find(strcmp('-d',varargin));)
		dflg=true;  % 详解: 赋值：计算表达式并保存到 dflg
	end  % 详解: 执行语句
	if	find(strcmp('-st',varargin));  % 详解: 条件判断：if (find(strcmp('-st',varargin));)
		sflg=true;  % 详解: 赋值：计算表达式并保存到 sflg
	end  % 详解: 执行语句
	if	find(strcmp('-v',varargin));  % 详解: 条件判断：if (find(strcmp('-v',varargin));)
		vflg=true;  % 详解: 赋值：计算表达式并保存到 vflg
	end  % 详解: 执行语句
	ix=find(strcmp('-w',varargin));  % 详解: 赋值：将 find(...) 的结果保存到 ix
	if	~isempty(ix)  % 详解: 条件判断：if (~isempty(ix))
		wflg=ix(end);  % 详解: 赋值：将 ix(...) 的结果保存到 wflg
	end  % 详解: 执行语句
end  % 详解: 执行语句
ntmpl={  % 详解: 赋值：计算表达式并保存到 ntmpl
	' inf '  % 详解: 执行语句
	'+inf '  % 详解: 执行语句
	'-inf '  % 详解: 执行语句
	' nan '  % 详解: 执行语句
	'+nan '  % 详解: 执行语句
	'-nan '  % 详解: 执行语句
	};  % 详解: 执行语句
ctmpl={  % 详解: 赋值：计算表达式并保存到 ctmpl
	'.'  % 详解: 执行语句
	'd'  % 详解: 执行语句
	'e'  % 详解: 执行语句
	};  % 详解: 执行语句

if	nargout <= 3  % 详解: 条件判断：if (nargout <= 3)
	varargout{1}=inp;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
	disp(sprintf('ASORT> too many output args [%-1d/%-1d]\n',nargout,3));  % 详解: 调用函数：disp(sprintf('ASORT> too many output args [%-1d/%-1d]\n',nargout,3))
	help(mfilename);  % 详解: 调用函数：help(mfilename)
	return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句
if	isempty(inp)  % 详解: 条件判断：if (isempty(inp))
	disp(sprintf('ASORT> input is empty'));  % 详解: 调用函数：disp(sprintf('ASORT> input is empty'))
	return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

ti=clock;  % 详解: 赋值：计算表达式并保存到 ti
winp=whos('inp');  % 详解: 赋值：将 whos(...) 的结果保存到 winp
switch	winp.class  % 详解: 多分支选择：switch (winp.class)
	case	'cell'  % 详解: 分支：case 'cell'
		if	~iscellstr(inp)  % 详解: 条件判断：if (~iscellstr(inp))
			disp(sprintf('ASORT> cell is not an array of strings'));  % 详解: 调用函数：disp(sprintf('ASORT> cell is not an array of strings'))
			return;  % 详解: 返回：从当前函数返回
		end  % 详解: 执行语句
		inp=inp(:);  % 详解: 赋值：将 inp(...) 的结果保存到 inp
		[ins,inx]=sort(inp);  % 详解: 执行语句
	case	'char'  % 详解: 分支：case 'char'
		inp=cstr(inp);  % 详解: 赋值：将 cstr(...) 的结果保存到 inp
	otherwise  % 详解: 默认分支：otherwise
		disp(sprintf('ASORT> does not sort input of class <%s>',winp.class));  % 详解: 调用函数：disp(sprintf('ASORT> does not sort input of class <%s>',winp.class))
		return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

inp=inp(:);  % 详解: 赋值：将 inp(...) 的结果保存到 inp
inp=setinp(inp,tmpl,[tflg wflg]);  % 详解: 赋值：将 setinp(...) 的结果保存到 inp
[ins,inx]=sort(inp);  % 详解: 执行语句
if	strcmp(smod,'descend')  % 详解: 条件判断：if (strcmp(smod,'descend'))
	ins=ins(end:-1:1,:);  % 详解: 赋值：将 ins(...) 的结果保存到 ins
	inx=inx(end:-1:1);  % 详解: 赋值：将 inx(...) 的结果保存到 inx
end  % 详解: 执行语句
ins=inp(inx);  % 详解: 赋值：将 inp(...) 的结果保存到 ins
c=lower(char(ins));  % 详解: 赋值：将 lower(...) 的结果保存到 c
wins=whos('c');  % 详解: 赋值：将 whos(...) 的结果保存到 wins
[cr,cc]=size(c);  % 详解: 获取向量/矩阵尺寸

c=[' '*ones(cr,2) c ' '*ones(cr,2)];  % 详解: 赋值：计算表达式并保存到 c

t=(c>='0'&c<='9');  % 详解: 赋值：计算表达式并保存到 t
t=t|c=='-';  % 详解: 赋值：计算表达式并保存到 t
t=t|c=='+';  % 详解: 赋值：计算表达式并保存到 t
[tr,tc]=size(t);  % 详解: 获取向量/矩阵尺寸
ix1=	 t(:,1:end-2) & ...  % 详解: 赋值：将 t(...) 的结果保存到 ix1
	~isletter(c(:,1:end-2)) & ...  % 详解: 执行语句
	c(:,2:end-1)=='.';  % 详解: 执行语句
t(:,2:end-1)=t(:,2:end-1)|ix1;  % 详解: 执行语句
ix1=	(t(:,3:end) & ...  % 详解: 赋值：计算表达式并保存到 ix1
	(~isletter(c(:,3:end)) & ...  % 详解: 执行语句
	~isletter(c(:,1:end-2))) | ...  % 详解: 执行语句
	(c(:,3:end)=='e' | ...  % 详解: 执行语句
	c(:,3:end)=='d')) & ...  % 详解: 执行语句
	c(:,2:end-1)=='.';  % 详解: 执行语句
t(:,2:end-1)=t(:,2:end-1)|ix1;  % 详解: 执行语句
t(c=='-')=false;  % 详解: 执行语句
t(c=='+')=false;  % 详解: 执行语句
ix1=	 t(:,3:end) & ...  % 详解: 赋值：将 t(...) 的结果保存到 ix1
	(c(:,2:end-1)=='-' | ...  % 详解: 执行语句
	c(:,2:end-1)=='+');  % 详解: 调用函数：c(:,2:end-1)=='+')
t(:,2:end-1)=t(:,2:end-1)|ix1;  % 详解: 执行语句
ix1=	 t(:,1:end-2) & ...  % 详解: 赋值：将 t(...) 的结果保存到 ix1
	(c(:,2:end-1)=='e' | ...  % 详解: 执行语句
	c(:,2:end-1)=='d');  % 详解: 调用函数：c(:,2:end-1)=='d')
t(:,2:end-1)=t(:,2:end-1)|ix1;  % 详解: 执行语句
c=reshape(c.',1,[]);  % 赋值：设置变量 c  % 详解: 赋值：将 reshape(...) 的结果保存到 c  % 详解: 赋值：将 reshape(...) 的结果保存到 c
t=t';  % 赋值：设置变量 t  % 详解: 赋值：计算表达式并保存到 t  % 详解: 赋值：计算表达式并保存到 t
ic=[];  % 详解: 赋值：计算表达式并保存到 ic
for	j=1:numel(ntmpl)  % 详解: for 循环：迭代变量 j 遍历 1:numel(ntmpl)
	ic=[ic,strfind(c,ntmpl{j})];  % 详解: 赋值：计算表达式并保存到 ic
end  % 详解: 执行语句
ic=sort(ic);  % 详解: 赋值：将 sort(...) 的结果保存到 ic
for	i=1:numel(ic)  % 详解: for 循环：迭代变量 i 遍历 1:numel(ic)
	ix=ic(i)+0:ic(i)+4;  % 详解: 赋值：将 ic(...) 的结果保存到 ix
	t(ix)=true;  % 详解: 执行语句
end  % 详解: 执行语句
t=t';  % 赋值：设置变量 t  % 详解: 赋值：计算表达式并保存到 t  % 详解: 赋值：计算表达式并保存到 t
c=reshape(c.',[tc,tr]).';  % 详解: 赋值：将 reshape(...) 的结果保存到 c
t(c==' ')=false;  % 详解: 执行语句

il=~any(t,2);  % 详解: 赋值：计算表达式并保存到 il
ib=strfind(reshape(t.',1,[]),[0 1]);  % 赋值：设置变量 ib  % 详解: 赋值：将 strfind(...) 的结果保存到 ib  % 详解: 赋值：将 strfind(...) 的结果保存到 ib
if	~isempty(ib)  % 详解: 条件判断：if (~isempty(ib))
	ixe=cell(3,1);  % 详解: 赋值：将 cell(...) 的结果保存到 ixe
	n=reshape(char(t.*c).',1,[]);  % 赋值：设置变量 n  % 详解: 赋值：将 reshape(...) 的结果保存到 n  % 详解: 赋值：将 reshape(...) 的结果保存到 n
	for	i=1:numel(ctmpl)  % 详解: for 循环：迭代变量 i 遍历 1:numel(ctmpl)
		id=strfind(n,ctmpl{i});  % 详解: 赋值：将 strfind(...) 的结果保存到 id
		if	~isempty(id)  % 详解: 条件判断：if (~isempty(id))
			[dum,dum,ixu{i},ixe{i}]=dupinx(id,tc);  % 详解: 执行语句
		end  % 详解: 执行语句
	end  % 详解: 执行语句
	in=false(tr,1);  % 详解: 赋值：将 false(...) 的结果保存到 in
	im=in;  % 详解: 赋值：计算表达式并保存到 im
	id=sort(...  % 详解: 赋值：将 sort(...) 的结果保存到 id
		[find(n>='0' & n<='9'),...  % 详解: 执行语句
		strfind(n,'inf'),...  % 详解: 执行语句
		strfind(n,'nan')]);  % 详解: 调用函数：strfind(n,'nan')])
	[ibu,ibd,ixbu,ixbd]=dupinx(id,tc);  % 详解: 执行语句
	in(ixbu)=true;  % 详解: 执行语句
	in(ixbd)=true;  % 详解: 执行语句
	[ibu,ibd,ixbu,ixbd]=dupinx(ib,tc);  % 详解: 执行语句
	im(ixbu)=true;  % 详解: 执行语句
	in=in&im;  % 详解: 赋值：计算表达式并保存到 in
	in([ixe{:}])=false;  % 详解: 执行语句
	il=~any(t,2);  % 详解: 赋值：计算表达式并保存到 il
	ia=~(in|il);  % 详解: 赋值：计算表达式并保存到 ia

	n=t(in,:).*c(in,:);  % 详解: 赋值：将 t(...) 的结果保存到 n
	n(n==0)=' ';  % 详解: 执行语句
	n=char(n);  % 详解: 赋值：将 char(...) 的结果保存到 n
	dn=strread(n.','  % 详解: 赋值：将 strread(...) 的结果保存到 dn
	if	numel(dn) ~= numel(find(in))  % 详解: 条件判断：if (numel(dn) ~= numel(find(in)))
		if	nargout  % 详解: 条件判断：if (nargout)
			s.c=c;  % 详解: 赋值：计算表达式并保存到 s.c
			s.t=t;  % 详解: 赋值：计算表达式并保存到 s.t
			s.n=n;  % 详解: 赋值：计算表达式并保存到 s.n
			s.d=dn;  % 详解: 赋值：计算表达式并保存到 s.d
			varargout{1}=s;  % 详解: 执行语句
		end  % 详解: 执行语句
		return;  % 详解: 返回：从当前函数返回
	end  % 详解: 执行语句

	[ds,dx]=sort(dn,1,smod);  % 详解: 执行语句
	in=find(in);  % 详解: 赋值：将 find(...) 的结果保存到 in
	anr=ins(in(dx));  % 详解: 赋值：将 ins(...) 的结果保存到 anr
	snr=ins(ia);  % 详解: 赋值：将 ins(...) 的结果保存到 snr
end  % 详解: 执行语句
str=ins(il);  % 详解: 赋值：将 ins(...) 的结果保存到 str
to=clock;  % 详解: 赋值：计算表达式并保存到 to

if	nargout < 3 || sflg  % 详解: 条件判断：if (nargout < 3 || sflg)
	s.magic='ASORT';  % 详解: 赋值：计算表达式并保存到 s.magic
	s.ver='30-Mar-2005 11:57:07';  % 详解: 赋值：计算表达式并保存到 s.ver
	s.time=datestr(clock);  % 详解: 赋值：将 datestr(...) 的结果保存到 s.time
	s.runtime=etime(to,ti);  % 详解: 赋值：将 etime(...) 的结果保存到 s.runtime
	s.input_class=winp.class;  % 详解: 赋值：计算表达式并保存到 s.input_class
	s.input_msize=winp.size;  % 详解: 赋值：计算表达式并保存到 s.input_msize
	s.input_bytes=winp.bytes;  % 详解: 赋值：计算表达式并保存到 s.input_bytes
	s.strng_class=wins.class;  % 详解: 赋值：计算表达式并保存到 s.strng_class
	s.strng_msize=wins.size;  % 详解: 赋值：计算表达式并保存到 s.strng_msize
	s.strng_bytes=wins.bytes;  % 详解: 赋值：计算表达式并保存到 s.strng_bytes
	s.anr=anr;  % 详解: 赋值：计算表达式并保存到 s.anr
	s.snr=snr;  % 详解: 赋值：计算表达式并保存到 s.snr
	s.str=str;  % 详解: 赋值：计算表达式并保存到 s.str
	if	dflg  % 详解: 条件判断：if (dflg)
		s.c=c;  % 详解: 赋值：计算表达式并保存到 s.c
		s.t=t;  % 详解: 赋值：计算表达式并保存到 s.t
		s.n=n;  % 详解: 赋值：计算表达式并保存到 s.n
		s.d=ds;  % 详解: 赋值：计算表达式并保存到 s.d
	end  % 详解: 执行语句
	varargout{1}=s;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
	s={anr,snr,str};  % 详解: 赋值：计算表达式并保存到 s
	for	i=1:nargout  % 详解: for 循环：迭代变量 i 遍历 1:nargout
		varargout{i}=s{i};  % 详解: 执行语句
	end  % 详解: 执行语句
end  % 详解: 执行语句

if	vflg  % 详解: 条件判断：if (vflg)
	inp=cstr(inp);  % 详解: 赋值：将 cstr(...) 的结果保存到 inp
	an=[{'--- NUMERICAL'};		anr];  % 详解: 赋值：计算表达式并保存到 an
	as=[{'--- ASCII NUMBERS'};	snr];  % 详解: 赋值：计算表达式并保存到 as
	at=[{'--- ASCII STRINGS'};	str];  % 详解: 赋值：计算表达式并保存到 at
	nn=[{'--- NUMBERS'};		num2cell(ds)];  % 详解: 赋值：计算表达式并保存到 nn
	ag={' ';' ';' '};  % 详解: 赋值：计算表达式并保存到 ag
	u=[{'INPUT'};			inp;ag];  % 详解: 赋值：计算表达式并保存到 u
	v=[{'ASCII SORT'};		ins;ag];  % 详解: 赋值：计算表达式并保存到 v
	w=[{'NUM SORT'};		an;as;at];  % 详解: 赋值：计算表达式并保存到 w
	x=[{'NUM READ'};		nn;as;at];  % 详解: 赋值：计算表达式并保存到 x
	w=[u,v,w,x];  % 详解: 赋值：计算表达式并保存到 w
	disp(w);  % 详解: 调用函数：disp(w)
end  % 详解: 执行语句

return;  % 详解: 返回：从当前函数返回
function	c=cstr(s)  % 详解: 执行语句

c=s;  % 详解: 赋值：计算表达式并保存到 c
if	ischar(s)  % 详解: 条件判断：if (ischar(s))
	sr=size(s,1);  % 详解: 赋值：将 size(...) 的结果保存到 sr
	c=cell(sr,1);  % 详解: 赋值：将 cell(...) 的结果保存到 c
	for	i=1:sr  % 详解: for 循环：迭代变量 i 遍历 1:sr
		c{i}=s(i,:);  % 详解: 执行语句
	end  % 详解: 执行语句
end  % 详解: 执行语句
return;  % 详解: 返回：从当前函数返回
function	[idu,idd,ixu,ixd]=dupinx(ix,nc)  % 详解: 函数定义：dupinx(ix,nc), 返回：idu,idd,ixu,ixd

if	isempty(ix)  % 详解: 条件判断：if (isempty(ix))
	idu=[];  % 详解: 赋值：计算表达式并保存到 idu
	idd=[];  % 详解: 赋值：计算表达式并保存到 idd
	ixu=[];  % 详解: 赋值：计算表达式并保存到 ixu
	ixd=[];  % 详解: 赋值：计算表达式并保存到 ixd
	return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句
id=fix(ix/nc)+1;  % 详解: 赋值：将 fix(...) 的结果保存到 id
idi=diff(id)~=0;  % 详解: 赋值：将 diff(...) 的结果保存到 idi
ide=[true idi];  % 详解: 赋值：计算表达式并保存到 ide
idb=[idi true];  % 详解: 赋值：计算表达式并保存到 idb
idu=idb & ide;  % 详解: 赋值：计算表达式并保存到 idu
idd=idb==1 & ide==0;  % 详解: 赋值：计算表达式并保存到 idd
ixu=id(idu);  % 详解: 赋值：将 id(...) 的结果保存到 ixu
ixd=id(idd);  % 详解: 赋值：将 id(...) 的结果保存到 ixd
return;  % 详解: 返回：从当前函数返回
function	inp=setinp(inp,tmpl,flg)  % 详解: 执行语句

if	isempty(inp) || ~any(flg)  % 详解: 条件判断：if (isempty(inp) || ~any(flg))
	return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

for	i=sort(flg)  % 详解: for 循环：迭代变量 i 遍历 sort(flg)
	switch	i  % 详解: 多分支选择：switch (i)
		case	flg(1)  % 详解: 分支：case flg(1)
			if	ischar(tmpl)  % 详解: 条件判断：if (ischar(tmpl))
				tmpl={tmpl};  % 详解: 赋值：计算表达式并保存到 tmpl
			end  % 详解: 执行语句
			for	i=1:numel(tmpl)  % 详解: for 循环：迭代变量 i 遍历 1:numel(tmpl)
				inp=strrep(inp,tmpl{i},' ');  % 详解: 赋值：将 strrep(...) 的结果保存到 inp
			end  % 详解: 执行语句
		case	flg(2)  % 详解: 分支：case flg(2)
			inp=strrep(inp,' ','');  % 详解: 赋值：将 strrep(...) 的结果保存到 inp
	end  % 详解: 执行语句
end  % 详解: 执行语句
return;  % 详解: 返回：从当前函数返回




