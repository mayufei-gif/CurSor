% 文件: exportfig.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function varargout = exportfig(varargin)  % 详解: 执行语句


if (nargin < 2)  % 详解: 条件判断：if ((nargin < 2))
  error('Too few input arguments');  % 详解: 调用函数：error('Too few input arguments')
end  % 详解: 执行语句

H = varargin{1};  % 详解: 赋值：计算表达式并保存到 H
if ~LocalIsHG(H,'figure')  % 详解: 条件判断：if (~LocalIsHG(H,'figure'))
  error('First argument must be a handle to a figure.');  % 详解: 调用函数：error('First argument must be a handle to a figure.')
end  % 详解: 执行语句
filename = varargin{2};  % 详解: 赋值：计算表达式并保存到 filename
if ~ischar(filename)  % 详解: 条件判断：if (~ischar(filename))
  error('Second argument must be a string.');  % 详解: 调用函数：error('Second argument must be a string.')
end  % 详解: 执行语句
paramPairs = {varargin{3:end}};  % 详解: 赋值：计算表达式并保存到 paramPairs
if nargin > 2  % 详解: 条件判断：if (nargin > 2)
  if isstruct(paramPairs{1})  % 详解: 条件判断：if (isstruct(paramPairs{1}))
    pcell = LocalToCell(paramPairs{1});  % 详解: 赋值：将 LocalToCell(...) 的结果保存到 pcell
    paramPairs = {pcell{:}, paramPairs{2:end}};  % 详解: 赋值：计算表达式并保存到 paramPairs
  end  % 详解: 执行语句
end  % 详解: 执行语句
verstr = version;  % 详解: 赋值：计算表达式并保存到 verstr
majorver = str2num(verstr(1));  % 详解: 赋值：将 str2num(...) 的结果保存到 majorver
defaults = [];  % 详解: 赋值：计算表达式并保存到 defaults
if majorver > 5  % 详解: 条件判断：if (majorver > 5)
  if ispref('exportfig','defaults')  % 详解: 条件判断：if (ispref('exportfig','defaults'))
    defaults = getpref('exportfig','defaults');  % 详解: 赋值：将 getpref(...) 的结果保存到 defaults
  end  % 详解: 执行语句
elseif exist('getappdata')  % 详解: 条件判断：elseif (exist('getappdata'))
  defaults = getappdata(0,'exportfigdefaults');  % 详解: 赋值：将 getappdata(...) 的结果保存到 defaults
end  % 详解: 执行语句
if ~isempty(defaults)  % 详解: 条件判断：if (~isempty(defaults))
  dcell = LocalToCell(defaults);  % 详解: 赋值：将 LocalToCell(...) 的结果保存到 dcell
  paramPairs = {dcell{:}, paramPairs{:}};  % 详解: 赋值：计算表达式并保存到 paramPairs
end  % 详解: 执行语句

if (rem(length(paramPairs),2) ~= 0)  % 详解: 条件判断：if ((rem(length(paramPairs),2) ~= 0))
  error(['Invalid input syntax. Optional parameters and values' ...  % 详解: 执行语句
	 ' must be in pairs.']);  % 详解: 执行语句
end  % 详解: 执行语句

auto.format = 'eps';  % 详解: 赋值：计算表达式并保存到 auto.format
auto.preview = 'none';  % 详解: 赋值：计算表达式并保存到 auto.preview
auto.width = -1;  % 详解: 赋值：计算表达式并保存到 auto.width
auto.height = -1;  % 详解: 赋值：计算表达式并保存到 auto.height
auto.color = 'bw';  % 详解: 赋值：计算表达式并保存到 auto.color
auto.defaultfontsize=10;  % 详解: 赋值：计算表达式并保存到 auto.defaultfontsize
auto.fontsize = -1;  % 详解: 赋值：计算表达式并保存到 auto.fontsize
auto.fontmode='scaled';  % 详解: 赋值：计算表达式并保存到 auto.fontmode
auto.fontmin = 8;  % 详解: 赋值：计算表达式并保存到 auto.fontmin
auto.fontmax = 60;  % 详解: 赋值：计算表达式并保存到 auto.fontmax
auto.defaultlinewidth = 1.0;  % 详解: 赋值：计算表达式并保存到 auto.defaultlinewidth
auto.linewidth = -1;  % 详解: 赋值：计算表达式并保存到 auto.linewidth
auto.linemode=[];  % 详解: 赋值：计算表达式并保存到 auto.linemode
auto.linemin = 0.5;  % 详解: 赋值：计算表达式并保存到 auto.linemin
auto.linemax = 100;  % 详解: 赋值：计算表达式并保存到 auto.linemax
auto.fontencoding = 'latin1';  % 详解: 赋值：计算表达式并保存到 auto.fontencoding
auto.renderer = [];  % 详解: 赋值：计算表达式并保存到 auto.renderer
auto.resolution = [];  % 详解: 赋值：计算表达式并保存到 auto.resolution
auto.stylemap = [];  % 详解: 赋值：计算表达式并保存到 auto.stylemap
auto.applystyle = 0;  % 详解: 赋值：计算表达式并保存到 auto.applystyle
auto.refobj = -1;  % 详解: 赋值：计算表达式并保存到 auto.refobj
auto.bounds = 'tight';  % 详解: 赋值：计算表达式并保存到 auto.bounds
explicitbounds = 0;  % 详解: 赋值：计算表达式并保存到 explicitbounds
auto.lockaxes = 1;  % 详解: 赋值：计算表达式并保存到 auto.lockaxes
auto.separatetext = 0;  % 详解: 赋值：计算表达式并保存到 auto.separatetext
opts = auto;  % 详解: 赋值：计算表达式并保存到 opts

args = {};  % 详解: 赋值：计算表达式并保存到 args
for k = 1:2:length(paramPairs)  % 详解: for 循环：迭代变量 k 遍历 1:2:length(paramPairs)
  param = lower(paramPairs{k});  % 详解: 赋值：将 lower(...) 的结果保存到 param
  if ~ischar(param)  % 详解: 条件判断：if (~ischar(param))
    error('Optional parameter names must be strings');  % 详解: 调用函数：error('Optional parameter names must be strings')
  end  % 详解: 执行语句
  value = paramPairs{k+1};  % 详解: 赋值：计算表达式并保存到 value
  
  switch (param)  % 详解: 多分支选择：switch ((param))
   case 'format'  % 详解: 分支：case 'format'
    opts.format = LocalCheckAuto(lower(value),auto.format);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.format
    if strcmp(opts.format,'preview')  % 详解: 条件判断：if (strcmp(opts.format,'preview'))
      error(['Format ''preview'' no longer supported. Use PREVIEWFIG' ...  % 详解: 执行语句
	     ' instead.']);  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'preview'  % 详解: 分支：case 'preview'
    opts.preview = LocalCheckAuto(lower(value),auto.preview);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.preview
    if ~strcmp(opts.preview,{'none','tiff'})  % 详解: 条件判断：if (~strcmp(opts.preview,{'none','tiff'}))
      error('Preview must be ''none'' or ''tiff''.');  % 详解: 调用函数：error('Preview must be ''none'' or ''tiff''.')
    end  % 详解: 执行语句
   case 'width'  % 详解: 分支：case 'width'
    opts.width = LocalToNum(value, auto.width);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.width
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.width)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.width))
	error('Width must be a numeric scalar > 0');  % 详解: 调用函数：error('Width must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'height'  % 详解: 分支：case 'height'
    opts.height = LocalToNum(value, auto.height);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.height
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if(~LocalIsPositiveScalar(opts.height))  % 详解: 调用函数：if(~LocalIsPositiveScalar(opts.height))
	error('Height must be a numeric scalar > 0');  % 详解: 调用函数：error('Height must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'color'  % 详解: 分支：case 'color'
    opts.color = LocalCheckAuto(lower(value),auto.color);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.color
    if ~strcmp(opts.color,{'bw','gray','rgb','cmyk'})  % 详解: 条件判断：if (~strcmp(opts.color,{'bw','gray','rgb','cmyk'}))
      error('Color must be ''bw'', ''gray'',''rgb'' or ''cmyk''.');  % 详解: 调用函数：error('Color must be ''bw'', ''gray'',''rgb'' or ''cmyk''.')
    end  % 详解: 执行语句
   case 'fontmode'  % 详解: 分支：case 'fontmode'
    opts.fontmode = LocalCheckAuto(lower(value),auto.fontmode);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.fontmode
    if ~strcmp(opts.fontmode,{'scaled','fixed'})  % 详解: 条件判断：if (~strcmp(opts.fontmode,{'scaled','fixed'}))
      error('FontMode must be ''scaled'' or ''fixed''.');  % 详解: 调用函数：error('FontMode must be ''scaled'' or ''fixed''.')
    end  % 详解: 执行语句
   case 'fontsize'  % 详解: 分支：case 'fontsize'
    opts.fontsize = LocalToNum(value,auto.fontsize);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.fontsize
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.fontsize)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.fontsize))
	error('FontSize must be a numeric scalar > 0');  % 详解: 调用函数：error('FontSize must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'defaultfixedfontsize'  % 详解: 分支：case 'defaultfixedfontsize'
    opts.defaultfontsize = LocalToNum(value,auto.defaultfontsize);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.defaultfontsize
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.defaultfontsize)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.defaultfontsize))
	error('DefaultFixedFontSize must be a numeric scalar > 0');  % 详解: 调用函数：error('DefaultFixedFontSize must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'fontsizemin'  % 详解: 分支：case 'fontsizemin'
    opts.fontmin = LocalToNum(value,auto.fontmin);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.fontmin
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.fontmin)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.fontmin))
	error('FontSizeMin must be a numeric scalar > 0');  % 详解: 调用函数：error('FontSizeMin must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'fontsizemax'  % 详解: 分支：case 'fontsizemax'
    opts.fontmax = LocalToNum(value,auto.fontmax);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.fontmax
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.fontmax)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.fontmax))
	error('FontSizeMax must be a numeric scalar > 0');  % 详解: 调用函数：error('FontSizeMax must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'fontencoding'  % 详解: 分支：case 'fontencoding'
    opts.fontencoding = LocalCheckAuto(lower(value),auto.fontencoding);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.fontencoding
    if ~strcmp(opts.fontencoding,{'latin1','adobe'})  % 详解: 条件判断：if (~strcmp(opts.fontencoding,{'latin1','adobe'}))
      error('FontEncoding must be ''latin1'' or ''adobe''.');  % 详解: 调用函数：error('FontEncoding must be ''latin1'' or ''adobe''.')
    end  % 详解: 执行语句
   case 'linemode'  % 详解: 分支：case 'linemode'
    opts.linemode = LocalCheckAuto(lower(value),auto.linemode);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.linemode
    if ~strcmp(opts.linemode,{'scaled','fixed'})  % 详解: 条件判断：if (~strcmp(opts.linemode,{'scaled','fixed'}))
      error('LineMode must be ''scaled'' or ''fixed''.');  % 详解: 调用函数：error('LineMode must be ''scaled'' or ''fixed''.')
    end  % 详解: 执行语句
   case 'linewidth'  % 详解: 分支：case 'linewidth'
    opts.linewidth = LocalToNum(value,auto.linewidth);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.linewidth
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.linewidth)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.linewidth))
	error('LineWidth must be a numeric scalar > 0');  % 详解: 调用函数：error('LineWidth must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'defaultfixedlinewidth'  % 详解: 分支：case 'defaultfixedlinewidth'
    opts.defaultlinewidth = LocalToNum(value,auto.defaultlinewidth);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.defaultlinewidth
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.defaultlinewidth)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.defaultlinewidth))
	error(['DefaultFixedLineWidth must be a numeric scalar >' ...  % 详解: 执行语句
	       ' 0']);  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'linewidthmin'  % 详解: 分支：case 'linewidthmin'
    opts.linemin = LocalToNum(value,auto.linemin);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.linemin
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.linemin)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.linemin))
	error('LineWidthMin must be a numeric scalar > 0');  % 详解: 调用函数：error('LineWidthMin must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'linewidthmax'  % 详解: 分支：case 'linewidthmax'
    opts.linemax = LocalToNum(value,auto.linemax);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.linemax
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~LocalIsPositiveScalar(opts.linemax)  % 详解: 条件判断：if (~LocalIsPositiveScalar(opts.linemax))
	error('LineWidthMax must be a numeric scalar > 0');  % 详解: 调用函数：error('LineWidthMax must be a numeric scalar > 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'linestylemap'  % 详解: 分支：case 'linestylemap'
    opts.stylemap = LocalCheckAuto(value,auto.stylemap);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.stylemap
   case 'renderer'  % 详解: 分支：case 'renderer'
    opts.renderer = LocalCheckAuto(lower(value),auto.renderer);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.renderer
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~strcmp(opts.renderer,{'painters','zbuffer','opengl'})  % 详解: 条件判断：if (~strcmp(opts.renderer,{'painters','zbuffer','opengl'}))
	error(['Renderer must be ''painters'', ''zbuffer'' or' ...  % 详解: 执行语句
	       ' ''opengl''.']);  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'resolution'  % 详解: 分支：case 'resolution'
    opts.resolution = LocalToNum(value,auto.resolution);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.resolution
    if ~ischar(value) | ~strcmp(value,'auto')  % 详解: 条件判断：if (~ischar(value) | ~strcmp(value,'auto'))
      if ~(isnumeric(value) & (prod(size(value)) == 1) & (value >= 0));  % 详解: 条件判断：if (~(isnumeric(value) & (prod(size(value)) == 1) & (value >= 0));)
	error('Resolution must be a numeric scalar >= 0');  % 详解: 调用函数：error('Resolution must be a numeric scalar >= 0')
      end  % 详解: 执行语句
    end  % 详解: 执行语句
   case 'applystyle'  % 详解: 分支：case 'applystyle'
    opts.applystyle = 1;  % 详解: 赋值：计算表达式并保存到 opts.applystyle
   case 'reference'  % 详解: 分支：case 'reference'
    if ischar(value)  % 详解: 条件判断：if (ischar(value))
      if strcmp(value,'auto')  % 详解: 条件判断：if (strcmp(value,'auto'))
	opts.refobj = auto.refobj;  % 详解: 赋值：计算表达式并保存到 opts.refobj
      else  % 详解: 条件判断：else 分支
	opts.refobj = eval(value);  % 详解: 赋值：将 eval(...) 的结果保存到 opts.refobj
      end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      opts.refobj = value;  % 详解: 赋值：计算表达式并保存到 opts.refobj
    end  % 详解: 执行语句
    if ~LocalIsHG(opts.refobj,'axes')  % 详解: 条件判断：if (~LocalIsHG(opts.refobj,'axes'))
      error('Reference object must evaluate to an axes handle.');  % 详解: 调用函数：error('Reference object must evaluate to an axes handle.')
    end  % 详解: 执行语句
   case 'bounds'  % 详解: 分支：case 'bounds'
    opts.bounds = LocalCheckAuto(lower(value),auto.bounds);  % 详解: 赋值：将 LocalCheckAuto(...) 的结果保存到 opts.bounds
    explicitbounds = 1;  % 详解: 赋值：计算表达式并保存到 explicitbounds
    if ~strcmp(opts.bounds,{'tight','loose'})  % 详解: 条件判断：if (~strcmp(opts.bounds,{'tight','loose'}))
      error('Bounds must be ''tight'' or ''loose''.');  % 详解: 调用函数：error('Bounds must be ''tight'' or ''loose''.')
    end  % 详解: 执行语句
   case 'lockaxes'  % 详解: 分支：case 'lockaxes'
    opts.lockaxes = LocalToNum(value,auto.lockaxes);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.lockaxes
   case 'separatetext'  % 详解: 分支：case 'separatetext'
    opts.separatetext = LocalToNum(value,auto.separatetext);  % 详解: 赋值：将 LocalToNum(...) 的结果保存到 opts.separatetext
   otherwise  % 详解: 默认分支：otherwise
    error(['Unrecognized option ' param '.']);  % 详解: 调用函数：error(['Unrecognized option ' param '.'])
  end  % 详解: 执行语句
end  % 详解: 执行语句

drawnow;  % 详解: 执行语句

allLines  = findall(H, 'type', 'line');  % 详解: 赋值：将 findall(...) 的结果保存到 allLines
allText   = findall(H, 'type', 'text');  % 详解: 赋值：将 findall(...) 的结果保存到 allText
allAxes   = findall(H, 'type', 'axes');  % 详解: 赋值：将 findall(...) 的结果保存到 allAxes
allImages = findall(H, 'type', 'image');  % 详解: 赋值：将 findall(...) 的结果保存到 allImages
allLights = findall(H, 'type', 'light');  % 详解: 赋值：将 findall(...) 的结果保存到 allLights
allPatch  = findall(H, 'type', 'patch');  % 详解: 赋值：将 findall(...) 的结果保存到 allPatch
allSurf   = findall(H, 'type', 'surface');  % 详解: 赋值：将 findall(...) 的结果保存到 allSurf
allRect   = findall(H, 'type', 'rectangle');  % 详解: 赋值：将 findall(...) 的结果保存到 allRect
allFont   = [allText; allAxes];  % 详解: 赋值：计算表达式并保存到 allFont
allColor  = [allLines; allText; allAxes; allLights];  % 详解: 赋值：计算表达式并保存到 allColor
allMarker = [allLines; allPatch; allSurf];  % 详解: 赋值：计算表达式并保存到 allMarker
allEdge   = [allPatch; allSurf];  % 详解: 赋值：计算表达式并保存到 allEdge
allCData  = [allImages; allPatch; allSurf];  % 详解: 赋值：计算表达式并保存到 allCData

old.objs = {};  % 详解: 赋值：计算表达式并保存到 old.objs
old.prop = {};  % 详解: 赋值：计算表达式并保存到 old.prop
old.values = {};  % 详解: 赋值：计算表达式并保存到 old.values

if strncmp(opts.format,'eps',3) & ~strcmp(opts.preview,'none')  % 详解: 条件判断：if (strncmp(opts.format,'eps',3) & ~strcmp(opts.preview,'none'))
  args = {args{:}, ['-' opts.preview]};  % 详解: 赋值：计算表达式并保存到 args
end  % 详解: 执行语句

hadError = 0;  % 详解: 赋值：计算表达式并保存到 hadError
oldwarn = warning;  % 详解: 赋值：计算表达式并保存到 oldwarn
try  % 详解: 异常处理：try 块开始

  if opts.lockaxes  % 详解: 条件判断：if (opts.lockaxes)
    old = LocalManualAxesMode(old, allAxes, 'TickMode');  % 详解: 赋值：将 LocalManualAxesMode(...) 的结果保存到 old
    old = LocalManualAxesMode(old, allAxes, 'TickLabelMode');  % 详解: 赋值：将 LocalManualAxesMode(...) 的结果保存到 old
    old = LocalManualAxesMode(old, allAxes, 'LimMode');  % 详解: 赋值：将 LocalManualAxesMode(...) 的结果保存到 old
  end  % 详解: 执行语句

  figurePaperUnits = get(H, 'PaperUnits');  % 详解: 赋值：将 get(...) 的结果保存到 figurePaperUnits
  oldFigureUnits = get(H, 'Units');  % 详解: 赋值：将 get(...) 的结果保存到 oldFigureUnits
  oldFigPos = get(H,'Position');  % 详解: 赋值：将 get(...) 的结果保存到 oldFigPos
  set(H, 'Units', figurePaperUnits);  % 详解: 调用函数：set(H, 'Units', figurePaperUnits)
  figPos = get(H,'Position');  % 详解: 赋值：将 get(...) 的结果保存到 figPos
  refsize = figPos(3:4);  % 详解: 赋值：将 figPos(...) 的结果保存到 refsize
  if opts.refobj ~= -1  % 详解: 条件判断：if (opts.refobj ~= -1)
    oldUnits = get(opts.refobj, 'Units');  % 详解: 赋值：将 get(...) 的结果保存到 oldUnits
    set(opts.refobj, 'Units', figurePaperUnits);  % 详解: 调用函数：set(opts.refobj, 'Units', figurePaperUnits)
    r = get(opts.refobj, 'Position');  % 详解: 赋值：将 get(...) 的结果保存到 r
    refsize = r(3:4);  % 详解: 赋值：将 r(...) 的结果保存到 refsize
    set(opts.refobj, 'Units', oldUnits);  % 详解: 调用函数：set(opts.refobj, 'Units', oldUnits)
  end  % 详解: 执行语句
  aspectRatio = refsize(1)/refsize(2);  % 详解: 赋值：将 refsize(...) 的结果保存到 aspectRatio
  if (opts.width == -1) & (opts.height == -1)  % 详解: 条件判断：if ((opts.width == -1) & (opts.height == -1))
    opts.width = refsize(1);  % 详解: 赋值：将 refsize(...) 的结果保存到 opts.width
    opts.height = refsize(2);  % 详解: 赋值：将 refsize(...) 的结果保存到 opts.height
  elseif (opts.width == -1)  % 详解: 条件判断：elseif ((opts.width == -1))
    opts.width = opts.height * aspectRatio;  % 详解: 赋值：计算表达式并保存到 opts.width
  elseif (opts.height == -1)  % 详解: 条件判断：elseif ((opts.height == -1))
    opts.height = opts.width / aspectRatio;  % 详解: 赋值：计算表达式并保存到 opts.height
  end  % 详解: 执行语句
  wscale = opts.width/refsize(1);  % 详解: 赋值：计算表达式并保存到 wscale
  hscale = opts.height/refsize(2);  % 详解: 赋值：计算表达式并保存到 hscale
  sizescale = min(wscale,hscale);  % 详解: 赋值：将 min(...) 的结果保存到 sizescale
  old = LocalPushOldData(old,H,'PaperPositionMode', ...  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
			 get(H,'PaperPositionMode'));  % 详解: 调用函数：get(H,'PaperPositionMode'))
  set(H, 'PaperPositionMode', 'auto');  % 详解: 调用函数：set(H, 'PaperPositionMode', 'auto')
  newPos = [figPos(1) figPos(2)+figPos(4)*(1-hscale) ...  % 详解: 赋值：计算表达式并保存到 newPos
	    wscale*figPos(3) hscale*figPos(4)];  % 详解: 执行语句
  set(H, 'Position', newPos);  % 详解: 调用函数：set(H, 'Position', newPos)
  set(H, 'Units', oldFigureUnits);  % 详解: 调用函数：set(H, 'Units', oldFigureUnits)
  
  if ~isempty(opts.stylemap) & ~isempty(allLines)  % 详解: 条件判断：if (~isempty(opts.stylemap) & ~isempty(allLines))
    oldlstyle = LocalGetAsCell(allLines,'LineStyle');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldlstyle
    old = LocalPushOldData(old, allLines, {'LineStyle'}, ...  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
			   oldlstyle);  % 详解: 执行语句
    newlstyle = oldlstyle;  % 详解: 赋值：计算表达式并保存到 newlstyle
    if ischar(opts.stylemap) & strcmpi(opts.stylemap,'bw')  % 详解: 条件判断：if (ischar(opts.stylemap) & strcmpi(opts.stylemap,'bw'))
      newlstyle = LocalMapColorToStyle(allLines);  % 详解: 赋值：将 LocalMapColorToStyle(...) 的结果保存到 newlstyle
    else  % 详解: 条件判断：else 分支
      try  % 详解: 异常处理：try 块开始
	newlstyle = feval(opts.stylemap,allLines);  % 详解: 赋值：将 feval(...) 的结果保存到 newlstyle
      catch  % 详解: 异常处理：catch
	warning(['Skipping stylemap. ' lasterr]);  % 详解: 调用函数：warning(['Skipping stylemap. ' lasterr])
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    set(allLines,{'LineStyle'},newlstyle);  % 详解: 调用函数：set(allLines,{'LineStyle'},newlstyle)
  end  % 详解: 执行语句

  switch (opts.color)  % 详解: 多分支选择：switch ((opts.color))
   case {'bw', 'gray'}  % 详解: 分支：case {'bw', 'gray'}
    if ~strcmp(opts.color,'bw') & strncmp(opts.format,'eps',3)  % 详解: 条件判断：if (~strcmp(opts.color,'bw') & strncmp(opts.format,'eps',3))
      opts.format = [opts.format 'c'];  % 详解: 赋值：计算表达式并保存到 opts.format
    end  % 详解: 执行语句
    args = {args{:}, ['-d' opts.format]};  % 详解: 赋值：计算表达式并保存到 args
    
    oldcmap = get(H,'Colormap');  % 详解: 赋值：将 get(...) 的结果保存到 oldcmap
    newgrays = 0.30*oldcmap(:,1) + 0.59*oldcmap(:,2) + 0.11*oldcmap(:,3);  % 详解: 赋值：计算表达式并保存到 newgrays
    newcmap = [newgrays newgrays newgrays];  % 详解: 赋值：计算表达式并保存到 newcmap
    old = LocalPushOldData(old, H, 'Colormap', oldcmap);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
    set(H, 'Colormap', newcmap);  % 详解: 调用函数：set(H, 'Colormap', newcmap)

    old = LocalUpdateColors(allColor, 'color', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allAxes, 'xcolor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allAxes, 'ycolor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allAxes, 'zcolor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allMarker, 'MarkerEdgeColor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allMarker, 'MarkerFaceColor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allEdge, 'EdgeColor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allEdge, 'FaceColor', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    old = LocalUpdateColors(allCData, 'CData', old);  % 详解: 赋值：将 LocalUpdateColors(...) 的结果保存到 old
    
   case {'rgb','cmyk'}  % 详解: 分支：case {'rgb','cmyk'}
    if strncmp(opts.format,'eps',3)  % 详解: 条件判断：if (strncmp(opts.format,'eps',3))
      opts.format = [opts.format 'c'];  % 详解: 赋值：计算表达式并保存到 opts.format
      args = {args{:}, ['-d' opts.format]};  % 详解: 赋值：计算表达式并保存到 args
      if strcmp(opts.color,'cmyk')  % 详解: 条件判断：if (strcmp(opts.color,'cmyk'))
	args = {args{:}, '-cmyk'};  % 详解: 赋值：计算表达式并保存到 args
      end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      args = {args{:}, ['-d' opts.format]};  % 详解: 赋值：计算表达式并保存到 args
    end  % 详解: 执行语句
   otherwise  % 详解: 默认分支：otherwise
    error('Invalid Color parameter');  % 详解: 调用函数：error('Invalid Color parameter')
  end  % 详解: 执行语句
  if (~isempty(opts.renderer))  % 详解: 条件判断：if ((~isempty(opts.renderer)))
    args = {args{:}, ['-' opts.renderer]};  % 详解: 赋值：计算表达式并保存到 args
  end  % 详解: 执行语句
  if (~isempty(opts.resolution)) | ~strncmp(opts.format,'eps',3)  % 详解: 条件判断：if ((~isempty(opts.resolution)) | ~strncmp(opts.format,'eps',3))
    if isempty(opts.resolution)  % 详解: 条件判断：if (isempty(opts.resolution))
      opts.resolution = 0;  % 详解: 赋值：计算表达式并保存到 opts.resolution
    end  % 详解: 执行语句
    args = {args{:}, ['-r' int2str(opts.resolution)]};  % 详解: 赋值：计算表达式并保存到 args
  end  % 详解: 执行语句

  if ~isempty(opts.fontmode)  % 详解: 条件判断：if (~isempty(opts.fontmode))
    oldfonts = LocalGetAsCell(allFont,'FontSize');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldfonts
    oldfontunits = LocalGetAsCell(allFont,'FontUnits');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldfontunits
    set(allFont,'FontUnits','points');  % 详解: 调用函数：set(allFont,'FontUnits','points')
    switch (opts.fontmode)  % 详解: 多分支选择：switch ((opts.fontmode))
     case 'fixed'  % 详解: 分支：case 'fixed'
      if (opts.fontsize == -1)  % 详解: 条件判断：if ((opts.fontsize == -1))
	set(allFont,'FontSize',opts.defaultfontsize);  % 详解: 调用函数：set(allFont,'FontSize',opts.defaultfontsize)
      else  % 详解: 条件判断：else 分支
	set(allFont,'FontSize',opts.fontsize);  % 详解: 调用函数：set(allFont,'FontSize',opts.fontsize)
      end  % 详解: 执行语句
     case 'scaled'  % 详解: 分支：case 'scaled'
      if (opts.fontsize == -1)  % 详解: 条件判断：if ((opts.fontsize == -1))
	scale = sizescale;  % 详解: 赋值：计算表达式并保存到 scale
      else  % 详解: 条件判断：else 分支
	scale = opts.fontsize;  % 详解: 赋值：计算表达式并保存到 scale
      end  % 详解: 执行语句
      newfonts = LocalScale(oldfonts,scale,opts.fontmin,opts.fontmax);  % 详解: 赋值：将 LocalScale(...) 的结果保存到 newfonts
      set(allFont,{'FontSize'},newfonts);  % 详解: 调用函数：set(allFont,{'FontSize'},newfonts)
     otherwise  % 详解: 默认分支：otherwise
      error('Invalid FontMode parameter');  % 详解: 调用函数：error('Invalid FontMode parameter')
    end  % 详解: 执行语句
    old = LocalPushOldData(old, allFont, {'FontSize'}, oldfonts);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
    old = LocalPushOldData(old, allFont, {'FontUnits'}, oldfontunits);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
  end  % 详解: 执行语句
  if strcmp(opts.fontencoding,'adobe') & strncmp(opts.format,'eps',3)  % 详解: 条件判断：if (strcmp(opts.fontencoding,'adobe') & strncmp(opts.format,'eps',3))
    args = {args{:}, '-adobecset'};  % 详解: 赋值：计算表达式并保存到 args
  end  % 详解: 执行语句

  if ~isempty(opts.linemode)  % 详解: 条件判断：if (~isempty(opts.linemode))
    oldlines = LocalGetAsCell(allMarker,'LineWidth');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldlines
    old = LocalPushOldData(old, allMarker, {'LineWidth'}, oldlines);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
    switch (opts.linemode)  % 详解: 多分支选择：switch ((opts.linemode))
     case 'fixed'  % 详解: 分支：case 'fixed'
      if (opts.linewidth == -1)  % 详解: 条件判断：if ((opts.linewidth == -1))
	set(allMarker,'LineWidth',opts.defaultlinewidth);  % 详解: 调用函数：set(allMarker,'LineWidth',opts.defaultlinewidth)
      else  % 详解: 条件判断：else 分支
	set(allMarker,'LineWidth',opts.linewidth);  % 详解: 调用函数：set(allMarker,'LineWidth',opts.linewidth)
      end  % 详解: 执行语句
     case 'scaled'  % 详解: 分支：case 'scaled'
      if (opts.linewidth == -1)  % 详解: 条件判断：if ((opts.linewidth == -1))
	scale = sizescale;  % 详解: 赋值：计算表达式并保存到 scale
      else  % 详解: 条件判断：else 分支
	scale = opts.linewidth;  % 详解: 赋值：计算表达式并保存到 scale
      end  % 详解: 执行语句
      newlines = LocalScale(oldlines, scale, opts.linemin, opts.linemax);  % 详解: 赋值：将 LocalScale(...) 的结果保存到 newlines
      set(allMarker,{'LineWidth'},newlines);  % 详解: 调用函数：set(allMarker,{'LineWidth'},newlines)
    end  % 详解: 执行语句
  end  % 详解: 执行语句

  if strcmp(opts.bounds,'tight')  % 详解: 条件判断：if (strcmp(opts.bounds,'tight'))
    if (~strncmp(opts.format,'eps',3) & LocalHas3DPlot(allAxes)) | ...  % 详解: 条件判断：if ((~strncmp(opts.format,'eps',3) & LocalHas3DPlot(allAxes)) | ...)
	  (strncmp(opts.format,'eps',3) & opts.separatetext)  % 详解: 执行语句
      if (explicitbounds == 1)  % 详解: 条件判断：if ((explicitbounds == 1))
	warning(['Cannot compute ''tight'' bounds. Using ''loose''' ...  % 详解: 执行语句
		 ' bounds.']);  % 详解: 执行语句
      end  % 详解: 执行语句
      opts.bounds = 'loose';  % 详解: 赋值：计算表达式并保存到 opts.bounds
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  warning('off');  % 详解: 调用函数：warning('off')
  if ~isempty(allAxes)  % 详解: 条件判断：if (~isempty(allAxes))
    if strncmp(opts.format,'eps',3)  % 详解: 条件判断：if (strncmp(opts.format,'eps',3))
      if strcmp(opts.bounds,'loose')  % 详解: 条件判断：if (strcmp(opts.bounds,'loose'))
	args = {args{:}, '-loose'};  % 详解: 赋值：计算表达式并保存到 args
      end  % 详解: 执行语句
      old = LocalPushOldData(old,H,'Position', oldFigPos);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
    elseif strcmp(opts.bounds,'tight')  % 详解: 条件判断：elseif (strcmp(opts.bounds,'tight'))
      oldaunits = LocalGetAsCell(allAxes,'Units');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldaunits
      oldapos = LocalGetAsCell(allAxes,'Position');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldapos
      oldtunits = LocalGetAsCell(allText,'units');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldtunits
      oldtpos = LocalGetAsCell(allText,'Position');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldtpos
      set(allAxes,'units','points');  % 详解: 调用函数：set(allAxes,'units','points')
      apos = LocalGetAsCell(allAxes,'Position');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 apos
      oldunits = get(H,'Units');  % 详解: 赋值：将 get(...) 的结果保存到 oldunits
      set(H,'units','points');  % 详解: 调用函数：set(H,'units','points')
      origfr = get(H,'position');  % 详解: 赋值：将 get(...) 的结果保存到 origfr
      fr = [];  % 详解: 赋值：计算表达式并保存到 fr
      for k=1:length(allAxes)  % 详解: for 循环：迭代变量 k 遍历 1:length(allAxes)
	if ~strcmpi(get(allAxes(k),'Tag'),'legend')  % 详解: 条件判断：if (~strcmpi(get(allAxes(k),'Tag'),'legend'))
	  axesR = apos{k};  % 详解: 赋值：计算表达式并保存到 axesR
	  r = LocalAxesTightBoundingBox(axesR, allAxes(k));  % 详解: 赋值：将 LocalAxesTightBoundingBox(...) 的结果保存到 r
	  r(1:2) = r(1:2) + axesR(1:2);  % 详解: 调用函数：r(1:2) = r(1:2) + axesR(1:2)
	  fr = LocalUnionRect(fr,r);  % 详解: 赋值：将 LocalUnionRect(...) 的结果保存到 fr
	end  % 详解: 执行语句
      end  % 详解: 执行语句
      if isempty(fr)  % 详解: 条件判断：if (isempty(fr))
	fr = [0 0 origfr(3:4)];  % 详解: 赋值：计算表达式并保存到 fr
      end  % 详解: 执行语句
      for k=1:length(allAxes)  % 详解: for 循环：迭代变量 k 遍历 1:length(allAxes)
	ax = allAxes(k);  % 详解: 赋值：将 allAxes(...) 的结果保存到 ax
	r = apos{k};  % 详解: 赋值：计算表达式并保存到 r
	r(1:2) = r(1:2) - fr(1:2);  % 详解: 调用函数：r(1:2) = r(1:2) - fr(1:2)
	set(ax,'Position',r);  % 详解: 调用函数：set(ax,'Position',r)
      end  % 详解: 执行语句
      old = LocalPushOldData(old, allAxes, {'Position'}, oldapos);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
      old = LocalPushOldData(old, allText, {'Position'}, oldtpos);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
      old = LocalPushOldData(old, allText, {'Units'}, oldtunits);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
      old = LocalPushOldData(old, allAxes, {'Units'}, oldaunits);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
      old = LocalPushOldData(old, H, 'Position', oldFigPos);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
      old = LocalPushOldData(old, H, 'Units', oldFigureUnits);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
      r = [origfr(1) origfr(2)+origfr(4)-fr(4) fr(3:4)];  % 详解: 赋值：计算表达式并保存到 r
      set(H,'Position',r);  % 详解: 调用函数：set(H,'Position',r)
    else  % 详解: 条件判断：else 分支
      args = {args{:}, '-loose'};  % 详解: 赋值：计算表达式并保存到 args
      old = LocalPushOldData(old,H,'Position', oldFigPos);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  
  if opts.separatetext & ~opts.applystyle  % 详解: 条件判断：if (opts.separatetext & ~opts.applystyle)
    oldtvis = LocalGetAsCell(allText,'visible');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldtvis
    set(allText,'visible','off');  % 详解: 调用函数：set(allText,'visible','off')
    oldax = LocalGetAsCell(allAxes,'XTickLabel',1);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldax
    olday = LocalGetAsCell(allAxes,'YTickLabel',1);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 olday
    oldaz = LocalGetAsCell(allAxes,'ZTickLabel',1);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldaz
    null = cell(length(oldax),1);  % 详解: 赋值：将 cell(...) 的结果保存到 null
    [null{:}] = deal([]);  % 详解: 执行语句
    set(allAxes,{'XTickLabel'},null);  % 详解: 调用函数：set(allAxes,{'XTickLabel'},null)
    set(allAxes,{'YTickLabel'},null);  % 详解: 调用函数：set(allAxes,{'YTickLabel'},null)
    set(allAxes,{'ZTickLabel'},null);  % 详解: 调用函数：set(allAxes,{'ZTickLabel'},null)
    print(H, filename, args{:});  % 详解: 调用函数：print(H, filename, args{:})
    set(allText,{'Visible'},oldtvis);  % 详解: 调用函数：set(allText,{'Visible'},oldtvis)
    set(allAxes,{'XTickLabel'},oldax);  % 详解: 调用函数：set(allAxes,{'XTickLabel'},oldax)
    set(allAxes,{'YTickLabel'},olday);  % 详解: 调用函数：set(allAxes,{'YTickLabel'},olday)
    set(allAxes,{'ZTickLabel'},oldaz);  % 详解: 调用函数：set(allAxes,{'ZTickLabel'},oldaz)
    [path, name, ext] = fileparts(filename);  % 详解: 执行语句
    tfile = fullfile(path,[name '_t.eps']);  % 详解: 赋值：将 fullfile(...) 的结果保存到 tfile
    tfile2 = fullfile(path,[name '_t2.eps']);  % 详解: 赋值：将 fullfile(...) 的结果保存到 tfile2
    foundRenderer = 0;  % 详解: 赋值：计算表达式并保存到 foundRenderer
    for k=1:length(args)  % 详解: for 循环：迭代变量 k 遍历 1:length(args)
      if strncmp('-d',args{k},2)  % 详解: 条件判断：if (strncmp('-d',args{k},2))
	args{k} = '-deps';  % 详解: 执行语句
      elseif strncmp('-zbuffer',args{k},8) | ...  % 详解: 条件判断：elseif (strncmp('-zbuffer',args{k},8) | ...)
	    strncmp('-opengl', args{k},6)  % 详解: 调用函数：strncmp('-opengl', args{k},6)
	args{k} = '-painters';  % 详解: 执行语句
	foundRenderer = 1;  % 详解: 赋值：计算表达式并保存到 foundRenderer
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    if ~foundRenderer  % 详解: 条件判断：if (~foundRenderer)
      args = {args{:}, '-painters'};  % 详解: 赋值：计算表达式并保存到 args
    end  % 详解: 执行语句
    allNonText = [allLines; allLights; allPatch; ...  % 详解: 赋值：计算表达式并保存到 allNonText
		  allImages; allSurf; allRect];  % 详解: 执行语句
    oldvis = LocalGetAsCell(allNonText,'visible');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldvis
    oldc = LocalGetAsCell(allAxes,'color');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldc
    oldaxg = LocalGetAsCell(allAxes,'XGrid');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldaxg
    oldayg = LocalGetAsCell(allAxes,'YGrid');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldayg
    oldazg = LocalGetAsCell(allAxes,'ZGrid');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldazg
    [null{:}] = deal('off');  % 详解: 执行语句
    set(allAxes,{'XGrid'},null);  % 详解: 调用函数：set(allAxes,{'XGrid'},null)
    set(allAxes,{'YGrid'},null);  % 详解: 调用函数：set(allAxes,{'YGrid'},null)
    set(allAxes,{'ZGrid'},null);  % 详解: 调用函数：set(allAxes,{'ZGrid'},null)
    set(allNonText,'Visible','off');  % 详解: 调用函数：set(allNonText,'Visible','off')
    set(allAxes,'Color','none');  % 详解: 调用函数：set(allAxes,'Color','none')
    print(H, tfile2, args{:});  % 详解: 调用函数：print(H, tfile2, args{:})
    set(allNonText,{'Visible'},oldvis);  % 详解: 调用函数：set(allNonText,{'Visible'},oldvis)
    set(allAxes,{'Color'},oldc);  % 详解: 调用函数：set(allAxes,{'Color'},oldc)
    set(allAxes,{'XGrid'},oldaxg);  % 详解: 调用函数：set(allAxes,{'XGrid'},oldaxg)
    set(allAxes,{'YGrid'},oldayg);  % 详解: 调用函数：set(allAxes,{'YGrid'},oldayg)
    set(allAxes,{'ZGrid'},oldazg);  % 详解: 调用函数：set(allAxes,{'ZGrid'},oldazg)
    fid1 = fopen(tfile,'w');  % 详解: 赋值：将 fopen(...) 的结果保存到 fid1
    fid2 = fopen(tfile2,'r');  % 详解: 赋值：将 fopen(...) 的结果保存到 fid2
    line = fgetl(fid2);  % 详解: 赋值：将 fgetl(...) 的结果保存到 line
    while ischar(line)  % 详解: while 循环：当 (ischar(line)) 为真时迭代
      if strncmp(line,'%%Title',7)  % 详解: 条件判断：if (strncmp(line,'%%Title',7))
	fprintf(fid1,'%s\n',['%%Title: ', tfile]);  % 详解: 调用函数：fprintf(fid1,'%s\n',['%%Title: ', tfile])
      elseif (length(line) < 3)  % 详解: 条件判断：elseif ((length(line) < 3))
	fprintf(fid1,'%s\n',line);  % 详解: 调用函数：fprintf(fid1,'%s\n',line)
      elseif ~strcmp(line(end-2:end),' PR') & ...  % 详解: 条件判断：elseif (~strcmp(line(end-2:end),' PR') & ...)
	    ~strcmp(line(end-1:end),' L')  % 详解: 执行语句
	fprintf(fid1,'%s\n',line);  % 详解: 调用函数：fprintf(fid1,'%s\n',line)
      end  % 详解: 执行语句
      line = fgetl(fid2);  % 详解: 赋值：将 fgetl(...) 的结果保存到 line
    end  % 详解: 执行语句
    fclose(fid1);  % 详解: 调用函数：fclose(fid1)
    fclose(fid2);  % 详解: 调用函数：fclose(fid2)
    delete(tfile2);  % 详解: 调用函数：delete(tfile2)
    
  elseif ~opts.applystyle  % 详解: 条件判断：elseif (~opts.applystyle)
    drawnow;  % 详解: 执行语句
    print(H, filename, args{:});  % 详解: 调用函数：print(H, filename, args{:})
  end  % 详解: 执行语句
  warning(oldwarn);  % 详解: 调用函数：warning(oldwarn)
  
catch  % 详解: 异常处理：catch
  warning(oldwarn);  % 详解: 调用函数：warning(oldwarn)
  hadError = 1;  % 详解: 赋值：计算表达式并保存到 hadError
end  % 详解: 执行语句

if opts.applystyle  % 详解: 条件判断：if (opts.applystyle)
  varargout{1} = old;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  for n=1:length(old.objs)  % 详解: for 循环：迭代变量 n 遍历 1:length(old.objs)
    if ~iscell(old.values{n}) & iscell(old.prop{n})  % 详解: 条件判断：if (~iscell(old.values{n}) & iscell(old.prop{n}))
      old.values{n} = {old.values{n}};  % 详解: 执行语句
    end  % 详解: 执行语句
    set(old.objs{n}, old.prop{n}, old.values{n});  % 详解: 调用函数：set(old.objs{n}, old.prop{n}, old.values{n})
  end  % 详解: 执行语句
end  % 详解: 执行语句

if hadError  % 详解: 条件判断：if (hadError)
  error(deblank(lasterr));  % 详解: 调用函数：error(deblank(lasterr))
end  % 详解: 执行语句


function outData = LocalPushOldData(inData, objs, prop, values)  % 详解: 执行语句
outData.objs = {objs, inData.objs{:}};  % 详解: 赋值：计算表达式并保存到 outData.objs
outData.prop = {prop, inData.prop{:}};  % 详解: 赋值：计算表达式并保存到 outData.prop
outData.values = {values, inData.values{:}};  % 详解: 赋值：计算表达式并保存到 outData.values

function cellArray = LocalGetAsCell(fig,prop,allowemptycell);  % 详解: 执行语句
cellArray = get(fig,prop);  % 详解: 赋值：将 get(...) 的结果保存到 cellArray
if nargin < 3  % 详解: 条件判断：if (nargin < 3)
  allowemptycell = 0;  % 详解: 赋值：计算表达式并保存到 allowemptycell
end  % 详解: 执行语句
if ~iscell(cellArray) & (allowemptycell | ~isempty(cellArray))  % 详解: 条件判断：if (~iscell(cellArray) & (allowemptycell | ~isempty(cellArray)))
  cellArray = {cellArray};  % 详解: 赋值：计算表达式并保存到 cellArray
end  % 详解: 执行语句

function newArray = LocalScale(inArray, scale, minv, maxv)  % 详解: 执行语句
n = length(inArray);  % 详解: 赋值：将 length(...) 的结果保存到 n
newArray = cell(n,1);  % 详解: 赋值：将 cell(...) 的结果保存到 newArray
for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
  newArray{k} = min(maxv,max(minv,scale*inArray{k}(1)));  % 详解: 统计：最大/最小值
end  % 详解: 执行语句

function gray = LocalMapToGray1(color)  % 详解: 执行语句
gray = color;  % 详解: 赋值：计算表达式并保存到 gray
if ischar(color)  % 详解: 条件判断：if (ischar(color))
  switch color(1)  % 详解: 多分支选择：switch (color(1))
   case 'y'  % 详解: 分支：case 'y'
    color = [1 1 0];  % 详解: 赋值：计算表达式并保存到 color
   case 'm'  % 详解: 分支：case 'm'
    color = [1 0 1];  % 详解: 赋值：计算表达式并保存到 color
   case 'c'  % 详解: 分支：case 'c'
    color = [0 1 1];  % 详解: 赋值：计算表达式并保存到 color
   case 'r'  % 详解: 分支：case 'r'
    color = [1 0 0];  % 详解: 赋值：计算表达式并保存到 color
   case 'g'  % 详解: 分支：case 'g'
    color = [0 1 0];  % 详解: 赋值：计算表达式并保存到 color
   case 'b'  % 详解: 分支：case 'b'
    color = [0 0 1];  % 详解: 赋值：计算表达式并保存到 color
   case 'w'  % 详解: 分支：case 'w'
    color = [1 1 1];  % 详解: 赋值：计算表达式并保存到 color
   case 'k'  % 详解: 分支：case 'k'
    color = [0 0 0];  % 详解: 赋值：计算表达式并保存到 color
  end  % 详解: 执行语句
end  % 详解: 执行语句
if ~ischar(color)  % 详解: 条件判断：if (~ischar(color))
  gray = 0.30*color(1) + 0.59*color(2) + 0.11*color(3);  % 详解: 赋值：计算表达式并保存到 gray
end  % 详解: 执行语句

function newArray = LocalMapToGray(inArray);  % 详解: 执行语句
n = length(inArray);  % 详解: 赋值：将 length(...) 的结果保存到 n
newArray = cell(n,1);  % 详解: 赋值：将 cell(...) 的结果保存到 newArray
for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
  color = inArray{k};  % 详解: 赋值：计算表达式并保存到 color
  if ~isempty(color)  % 详解: 条件判断：if (~isempty(color))
    color = LocalMapToGray1(color);  % 详解: 赋值：将 LocalMapToGray1(...) 的结果保存到 color
  end  % 详解: 执行语句
  if isempty(color) | ischar(color)  % 详解: 条件判断：if (isempty(color) | ischar(color))
    newArray{k} = color;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    newArray{k} = [color color color];  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

function newArray = LocalMapColorToStyle(inArray);  % 详解: 执行语句
inArray = LocalGetAsCell(inArray,'Color');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 inArray
n = length(inArray);  % 详解: 赋值：将 length(...) 的结果保存到 n
newArray = cell(n,1);  % 详解: 赋值：将 cell(...) 的结果保存到 newArray
styles = {'-','--',':','-.'};  % 详解: 赋值：计算表达式并保存到 styles
uniques = [];  % 详解: 赋值：计算表达式并保存到 uniques
nstyles = length(styles);  % 详解: 赋值：将 length(...) 的结果保存到 nstyles
for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
  gray = LocalMapToGray1(inArray{k});  % 详解: 赋值：将 LocalMapToGray1(...) 的结果保存到 gray
  if isempty(gray) | ischar(gray) | gray < .05  % 详解: 条件判断：if (isempty(gray) | ischar(gray) | gray < .05)
    newArray{k} = '-';  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    if ~isempty(uniques) & any(gray == uniques)  % 详解: 条件判断：if (~isempty(uniques) & any(gray == uniques))
      ind = find(gray==uniques);  % 详解: 赋值：将 find(...) 的结果保存到 ind
    else  % 详解: 条件判断：else 分支
      uniques = [uniques gray];  % 详解: 赋值：计算表达式并保存到 uniques
      ind = length(uniques);  % 详解: 赋值：将 length(...) 的结果保存到 ind
    end  % 详解: 执行语句
    newArray{k} = styles{mod(ind-1,nstyles)+1};  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

function newArray = LocalMapCData(inArray);  % 详解: 执行语句
n = length(inArray);  % 详解: 赋值：将 length(...) 的结果保存到 n
newArray = cell(n,1);  % 详解: 赋值：将 cell(...) 的结果保存到 newArray
for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
  color = inArray{k};  % 详解: 赋值：计算表达式并保存到 color
  if (ndims(color) == 3) & isa(color,'double')  % 详解: 条件判断：if ((ndims(color) == 3) & isa(color,'double'))
    gray = 0.30*color(:,:,1) + 0.59*color(:,:,2) + 0.11*color(:,:,3);  % 详解: 赋值：计算表达式并保存到 gray
    color(:,:,1) = gray;  % 详解: 执行语句
    color(:,:,2) = gray;  % 详解: 执行语句
    color(:,:,3) = gray;  % 详解: 执行语句
  end  % 详解: 执行语句
  newArray{k} = color;  % 详解: 执行语句
end  % 详解: 执行语句

function outData = LocalUpdateColors(inArray, prop, inData)  % 详解: 执行语句
value = LocalGetAsCell(inArray,prop);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 value
outData.objs = {inData.objs{:}, inArray};  % 详解: 赋值：计算表达式并保存到 outData.objs
outData.prop = {inData.prop{:}, {prop}};  % 详解: 赋值：计算表达式并保存到 outData.prop
outData.values = {inData.values{:}, value};  % 详解: 赋值：计算表达式并保存到 outData.values
if (~isempty(value))  % 详解: 条件判断：if ((~isempty(value)))
  if strcmp(prop,'CData')  % 详解: 条件判断：if (strcmp(prop,'CData'))
    value = LocalMapCData(value);  % 详解: 赋值：将 LocalMapCData(...) 的结果保存到 value
  else  % 详解: 条件判断：else 分支
    value = LocalMapToGray(value);  % 详解: 赋值：将 LocalMapToGray(...) 的结果保存到 value
  end  % 详解: 执行语句
  set(inArray,{prop},value);  % 详解: 调用函数：set(inArray,{prop},value)
end  % 详解: 执行语句

function bool = LocalIsPositiveScalar(value)  % 详解: 执行语句
bool = isnumeric(value) & ...  % 详解: 赋值：将 isnumeric(...) 的结果保存到 bool
       prod(size(value)) == 1 & ...  % 详解: 获取向量/矩阵尺寸
       value > 0;  % 详解: 执行语句

function value = LocalToNum(value,auto)  % 详解: 执行语句
if ischar(value)  % 详解: 条件判断：if (ischar(value))
  if strcmp(value,'auto')  % 详解: 条件判断：if (strcmp(value,'auto'))
    value = auto;  % 详解: 赋值：计算表达式并保存到 value
  else  % 详解: 条件判断：else 分支
    value = str2num(value);  % 详解: 赋值：将 str2num(...) 的结果保存到 value
  end  % 详解: 执行语句
end  % 详解: 执行语句

function c = LocalToCell(s)  % 详解: 执行语句
f = fieldnames(s);  % 详解: 赋值：将 fieldnames(...) 的结果保存到 f
v = struct2cell(s);  % 详解: 赋值：将 struct2cell(...) 的结果保存到 v
opts = cell(2,length(f));  % 详解: 赋值：将 cell(...) 的结果保存到 opts
opts(1,:) = f;  % 详解: 执行语句
opts(2,:) = v;  % 详解: 执行语句
c = {opts{:}};  % 详解: 赋值：计算表达式并保存到 c

function c = LocalIsHG(obj,hgtype)  % 详解: 执行语句
c = 0;  % 详解: 赋值：计算表达式并保存到 c
if (length(obj) == 1) & ishandle(obj)  % 详解: 条件判断：if ((length(obj) == 1) & ishandle(obj))
  c = strcmp(get(obj,'type'),hgtype);  % 详解: 赋值：将 strcmp(...) 的结果保存到 c
end  % 详解: 执行语句

function c = LocalHas3DPlot(a)  % 详解: 执行语句
zticks = LocalGetAsCell(a,'ZTickLabel');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 zticks
c = 0;  % 详解: 赋值：计算表达式并保存到 c
for k=1:length(zticks)  % 详解: for 循环：迭代变量 k 遍历 1:length(zticks)
  if ~isempty(zticks{k})  % 详解: 条件判断：if (~isempty(zticks{k}))
    c = 1;  % 详解: 赋值：计算表达式并保存到 c
    return;  % 详解: 返回：从当前函数返回
  end  % 详解: 执行语句
end  % 详解: 执行语句

function r = LocalUnionRect(r1,r2)  % 详解: 执行语句
if isempty(r1)  % 详解: 条件判断：if (isempty(r1))
  r = r2;  % 详解: 赋值：计算表达式并保存到 r
elseif isempty(r2)  % 详解: 条件判断：elseif (isempty(r2))
  r = r1;  % 详解: 赋值：计算表达式并保存到 r
elseif max(r2(3:4)) > 0  % 详解: 条件判断：elseif (max(r2(3:4)) > 0)
  left = min(r1(1),r2(1));  % 详解: 赋值：将 min(...) 的结果保存到 left
  bot = min(r1(2),r2(2));  % 详解: 赋值：将 min(...) 的结果保存到 bot
  right = max(r1(1)+r1(3),r2(1)+r2(3));  % 详解: 赋值：将 max(...) 的结果保存到 right
  top = max(r1(2)+r1(4),r2(2)+r2(4));  % 详解: 赋值：将 max(...) 的结果保存到 top
  r = [left bot right-left top-bot];  % 详解: 赋值：计算表达式并保存到 r
else  % 详解: 条件判断：else 分支
  r = r1;  % 详解: 赋值：计算表达式并保存到 r
end  % 详解: 执行语句

function c = LocalLabelsMatchTicks(labs,ticks)  % 详解: 执行语句
c = 0;  % 详解: 赋值：计算表达式并保存到 c
try  % 详解: 异常处理：try 块开始
  t1 = num2str(ticks(1));  % 详解: 赋值：将 num2str(...) 的结果保存到 t1
  n = length(ticks);  % 详解: 赋值：将 length(...) 的结果保存到 n
  tend = num2str(ticks(n));  % 详解: 赋值：将 num2str(...) 的结果保存到 tend
  c = strncmp(labs(1),t1,length(labs(1))) & ...  % 详解: 赋值：将 strncmp(...) 的结果保存到 c
      strncmp(labs(n),tend,length(labs(n)));  % 详解: 调用函数：strncmp(labs(n),tend,length(labs(n)))
end  % 详解: 执行语句

function r = LocalAxesTightBoundingBox(axesR, a)  % 详解: 执行语句
r = [];  % 详解: 赋值：计算表达式并保存到 r
atext = findall(a,'type','text','visible','on');  % 详解: 赋值：将 findall(...) 的结果保存到 atext
if ~isempty(atext)  % 详解: 条件判断：if (~isempty(atext))
  set(atext,'units','points');  % 详解: 调用函数：set(atext,'units','points')
  res=LocalGetAsCell(atext,'extent');  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 res
  for n=1:length(atext)  % 详解: for 循环：迭代变量 n 遍历 1:length(atext)
    r = LocalUnionRect(r,res{n});  % 详解: 赋值：将 LocalUnionRect(...) 的结果保存到 r
  end  % 详解: 执行语句
end  % 详解: 执行语句
if strcmp(get(a,'visible'),'on')  % 详解: 条件判断：if (strcmp(get(a,'visible'),'on'))
  r = LocalUnionRect(r,[0 0 axesR(3:4)]);  % 详解: 赋值：将 LocalUnionRect(...) 的结果保存到 r
  oldunits = get(a,'fontunits');  % 详解: 赋值：将 get(...) 的结果保存到 oldunits
  set(a,'fontunits','points');  % 详解: 调用函数：set(a,'fontunits','points')
  label = text(0,0,'','parent',a,...  % 详解: 赋值：将 text(...) 的结果保存到 label
	       'units','points',...  % 详解: 执行语句
	       'fontsize',get(a,'fontsize'),...  % 详解: 执行语句
	       'fontname',get(a,'fontname'),...  % 详解: 执行语句
	       'fontweight',get(a,'fontweight'),...  % 详解: 执行语句
	       'fontangle',get(a,'fontangle'),...  % 详解: 执行语句
	       'visible','off');  % 详解: 执行语句
  fs = get(a,'fontsize');  % 详解: 赋值：将 get(...) 的结果保存到 fs

  ry = [0 -fs/2 0 axesR(4)+fs];  % 详解: 赋值：计算表达式并保存到 ry
  ylabs = get(a,'yticklabels');  % 详解: 赋值：将 get(...) 的结果保存到 ylabs
  yticks = get(a,'ytick');  % 详解: 赋值：将 get(...) 的结果保存到 yticks
  maxw = 0;  % 详解: 赋值：计算表达式并保存到 maxw
  if ~isempty(ylabs)  % 详解: 条件判断：if (~isempty(ylabs))
    for n=1:size(ylabs,1)  % 详解: for 循环：迭代变量 n 遍历 1:size(ylabs,1)
      set(label,'string',ylabs(n,:));  % 详解: 调用函数：set(label,'string',ylabs(n,:))
      ext = get(label,'extent');  % 详解: 赋值：将 get(...) 的结果保存到 ext
      maxw = max(maxw,ext(3));  % 详解: 赋值：将 max(...) 的结果保存到 maxw
    end  % 详解: 执行语句
    if ~LocalLabelsMatchTicks(ylabs,yticks) & ...  % 详解: 条件判断：if (~LocalLabelsMatchTicks(ylabs,yticks) & ...)
	  strcmp(get(a,'xaxislocation'),'bottom')  % 详解: 调用函数：strcmp(get(a,'xaxislocation'),'bottom')
      ry(4) = ry(4) + 1.5*ext(4);  % 详解: 调用函数：ry(4) = ry(4) + 1.5*ext(4)
    end  % 详解: 执行语句
    if strcmp(get(a,'yaxislocation'),'left')  % 详解: 条件判断：if (strcmp(get(a,'yaxislocation'),'left'))
      ry(1) = -(maxw+5);  % 详解: 调用函数：ry(1) = -(maxw+5)
    else  % 详解: 条件判断：else 分支
      ry(1) = axesR(3);  % 详解: 调用函数：ry(1) = axesR(3)
    end  % 详解: 执行语句
    ry(3) = maxw+5;  % 详解: 执行语句
    r = LocalUnionRect(r,ry);  % 详解: 赋值：将 LocalUnionRect(...) 的结果保存到 r
  end  % 详解: 执行语句

  rx = [0 0 0 fs+5];  % 详解: 赋值：计算表达式并保存到 rx
  xlabs = get(a,'xticklabels');  % 详解: 赋值：将 get(...) 的结果保存到 xlabs
  xticks = get(a,'xtick');  % 详解: 赋值：将 get(...) 的结果保存到 xticks
  if ~isempty(xlabs)  % 详解: 条件判断：if (~isempty(xlabs))
    if strcmp(get(a,'xaxislocation'),'bottom')  % 详解: 条件判断：if (strcmp(get(a,'xaxislocation'),'bottom'))
      rx(2) = -(fs+5);  % 详解: 调用函数：rx(2) = -(fs+5)
      if ~LocalLabelsMatchTicks(xlabs,xticks);  % 详解: 条件判断：if (~LocalLabelsMatchTicks(xlabs,xticks);)
	rx(4) = rx(4) + 2*fs;  % 详解: 执行语句
	rx(2) = rx(2) - 2*fs;  % 详解: 执行语句
      end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      rx(2) = axesR(4);  % 详解: 调用函数：rx(2) = axesR(4)
      if ~LocalLabelsMatchTicks(xlabs,xticks);  % 详解: 条件判断：if (~LocalLabelsMatchTicks(xlabs,xticks);)
	rx(4) = rx(4) + axesR(4) + 2*fs;  % 详解: 执行语句
	rx(2) = -2*fs;  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    set(label,'string',xlabs(1,:));  % 详解: 调用函数：set(label,'string',xlabs(1,:))
    ext1 = get(label,'extent');  % 详解: 赋值：将 get(...) 的结果保存到 ext1
    rx(1) = -ext1(3)/2;  % 详解: 执行语句
    set(label,'string',xlabs(size(xlabs,1),:));  % 详解: 调用函数：set(label,'string',xlabs(size(xlabs,1),:))
    ext2 = get(label,'extent');  % 详解: 赋值：将 get(...) 的结果保存到 ext2
    rx(3) = axesR(3) + (ext2(3) + ext1(3))/2;  % 详解: 执行语句
    r = LocalUnionRect(r,rx);  % 详解: 赋值：将 LocalUnionRect(...) 的结果保存到 r
  end  % 详解: 执行语句
  set(a,'fontunits',oldunits);  % 详解: 调用函数：set(a,'fontunits',oldunits)
  delete(label);  % 详解: 调用函数：delete(label)
end  % 详解: 执行语句

function c = LocalManualAxesMode(old, allAxes, base)  % 详解: 执行语句
xs = ['X' base];  % 详解: 赋值：计算表达式并保存到 xs
ys = ['Y' base];  % 详解: 赋值：计算表达式并保存到 ys
zs = ['Z' base];  % 详解: 赋值：计算表达式并保存到 zs
oldXMode = LocalGetAsCell(allAxes,xs);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldXMode
oldYMode = LocalGetAsCell(allAxes,ys);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldYMode
oldZMode = LocalGetAsCell(allAxes,zs);  % 详解: 赋值：将 LocalGetAsCell(...) 的结果保存到 oldZMode
old = LocalPushOldData(old, allAxes, {xs}, oldXMode);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
old = LocalPushOldData(old, allAxes, {ys}, oldYMode);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
old = LocalPushOldData(old, allAxes, {zs}, oldZMode);  % 详解: 赋值：将 LocalPushOldData(...) 的结果保存到 old
set(allAxes,xs,'manual');  % 详解: 调用函数：set(allAxes,xs,'manual')
set(allAxes,ys,'manual');  % 详解: 调用函数：set(allAxes,ys,'manual')
set(allAxes,zs,'manual');  % 详解: 调用函数：set(allAxes,zs,'manual')
c = old;  % 详解: 赋值：计算表达式并保存到 c

function val = LocalCheckAuto(val, auto)  % 详解: 执行语句
if ischar(val) & strcmp(val,'auto')  % 详解: 条件判断：if (ischar(val) & strcmp(val,'auto'))
  val = auto;  % 详解: 赋值：计算表达式并保存到 val
end  % 详解: 执行语句




