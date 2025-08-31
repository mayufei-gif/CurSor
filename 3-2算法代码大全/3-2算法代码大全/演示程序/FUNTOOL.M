% 文件: FUNTOOL.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function funtool(keyword,varargin)  % 详解: 函数定义：funtool(keyword,varargin)



if nargin == 0  % 详解: 条件判断：if (nargin == 0)

init.f = 'x';  % 详解: 赋值：计算表达式并保存到 init.f
init.g = '1';  % 详解: 赋值：计算表达式并保存到 init.g
init.x = '[-2*pi, 2*pi]';  % 详解: 赋值：计算表达式并保存到 init.x
init.a = '1/2';  % 详解: 赋值：计算表达式并保存到 init.a
init.l = str2mat( ...  % 详解: 赋值：将 str2mat(...) 的结果保存到 init.l
    '1/(5+4*cos(x));  [-2*pi, 2*pi]', ...  % 详解: 执行语句
    'cos(x^3)/(1+x^2);  [-2*pi, 2*pi]', ...  % 详解: 执行语句
    'x^4*(1-x)^4/(1+x^2);  [0, 1]', ...  % 详解: 执行语句
    'x^7-7*x^6+21*x^5-35*x^4+35*x^3-21*x^2+7*x-1;  [0.985, 1.015]', ...  % 详解: 执行语句
    'log(abs(sqrt(x)));  [0, 2]', ...  % 详解: 执行语句
    'tan(sin(x))-sin(tan(x));  [-pi, pi]');  % 详解: 执行语句

f = sym(init.f);  % 详解: 赋值：将 sym(...) 的结果保存到 f
g = sym(init.g);  % 详解: 赋值：将 sym(...) 的结果保存到 g
a = sym(init.a);  % 详解: 赋值：将 sym(...) 的结果保存到 a
x = init.x;  % 详解: 赋值：计算表达式并保存到 x
fxlist = init.l;  % 详解: 赋值：计算表达式并保存到 fxlist

blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks
p = .12*(1:7) - .02;  % 详解: 赋值：计算表达式并保存到 p
q = .60 - .14*(1:4);  % 详解: 赋值：计算表达式并保存到 q
r = [.10 .10];  % 详解: 赋值：计算表达式并保存到 r

figf = figure('units','normalized','pos',[.01 .50 .48 .48],...  % 详解: 赋值：将 figure(...) 的结果保存到 figf
              'menu','none','tag','figf');  % 详解: 执行语句
figg = figure('units','normalized','pos',[.50 .50 .48 .48],...  % 详解: 赋值：将 figure(...) 的结果保存到 figg
              'menu','none', 'tag','figg');  % 详解: 执行语句
figp = figure('units','normalized','pos',[.25 .05 .50 .40],'menu','none', ...  % 详解: 赋值：将 figure(...) 的结果保存到 figp
              'tag','figp',...  % 详解: 执行语句
              'Color',get(0,'DefaultUIControlBackgroundColor'), ...  % 详解: 执行语句
              'DefaultUIControlUnit','norm','UserData',fxlist);  % 详解: 执行语句

figure(figf)  % 详解: 调用函数：figure(figf)
ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)
figure(figg)  % 详解: 调用函数：figure(figg)
ezplot(g,x,figg)  % 详解: 调用函数：ezplot(g,x,figg)

figure(figp)  % 详解: 调用函数：figure(figp)
uicontrol('style','frame','pos',[0.01 0.60 0.98 0.38]);  % 详解: 调用函数：uicontrol('style','frame','pos',[0.01 0.60 0.98 0.38])
uicontrol('style','frame','pos',[0.01 0.01 0.98 0.58]);  % 详解: 调用函数：uicontrol('style','frame','pos',[0.01 0.01 0.98 0.58])
uicontrol('style','text','string','f = ','pos',[0.04 0.86 0.09 0.10],...  % 详解: 执行语句
    'tag','fobj','UserData',f);  % 详解: 执行语句
uicontrol('style','text','string','g = ','pos',[0.04 0.74 0.09 0.10],...  % 详解: 执行语句
    'tag','gobj','UserData',g);  % 详解: 执行语句
uicontrol('style','text','string','x = ','pos',[0.04 0.62 0.09 0.10],...  % 详解: 执行语句
    'tag','xstr','UserData',x);  % 详解: 执行语句
uicontrol('style','text','string','a = ','pos',[0.54 0.62 0.09 0.10],...  % 详解: 执行语句
    'tag','aobj','UserData',a);  % 详解: 执行语句
uicontrol('pos',[.12 .86 .82 .10],'style','edit','horiz','left', ...  % 详解: 执行语句
    'backgroundcolor','white', ...  % 详解: 执行语句
    'string', [blanks char(f)],'tag','Sf', ...  % 详解: 执行语句
    'CallBack','funtool Sfcallback');  % 详解: 执行语句
uicontrol('pos',[.12 .74 .82 .10],'style','edit','horiz','left', ...  % 详解: 执行语句
    'backgroundcolor','white', ...  % 详解: 执行语句
    'string', [blanks char(g)], 'tag','Sg',...  % 详解: 执行语句
    'CallBack','funtool Sgcallback');  % 详解: 执行语句
uicontrol('pos',[.12 .62 .32 .10],'style','edit','horiz','left', ...  % 详解: 执行语句
    'backgroundcolor','white','string',[blanks x], 'tag','Sx',...  % 详解: 执行语句
    'CallBack','funtool Sxcallback');  % 详解: 执行语句
uicontrol('pos',[.62 .62 .32 .10],'style','edit','horiz','left', ...  % 详解: 执行语句
    'backgroundcolor','white', 'tag','Sa',...  % 详解: 执行语句
    'string',[blanks char(a)],'CallBack','funtool Sacallback');  % 详解: 执行语句

uicontrol('pos',[p(1) q(1) r],'string','df/dx', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''diff'')');  % 详解: 执行语句
uicontrol('pos',[p(2) q(1) r],'string','int f', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''int'')');  % 详解: 执行语句
uicontrol('pos',[p(3) q(1) r],'string','simple f', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''simple'')');  % 详解: 执行语句
uicontrol('pos',[p(4) q(1) r],'string','num f', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''num'')');  % 详解: 执行语句
uicontrol('pos',[p(5) q(1) r],'string','den f', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''den'')');  % 详解: 执行语句
uicontrol('pos',[p(6) q(1) r],'string','1/f', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''1/f'')');  % 详解: 执行语句
uicontrol('pos',[p(7) q(1) r],'string','finv', ...  % 详解: 执行语句
   'CallBack','funtool(''row1'',''finverse'')');  % 详解: 执行语句

uicontrol('pos',[p(1) q(2) r],'string','f+a', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f+a'')');  % 详解: 执行语句
uicontrol('pos',[p(2) q(2) r],'string','f-a', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f-a'')');  % 详解: 执行语句
uicontrol('pos',[p(3) q(2) r],'string','f*a', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f*a'')');  % 详解: 执行语句
uicontrol('pos',[p(4) q(2) r],'string','f/a', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f/a'')');  % 详解: 执行语句
uicontrol('pos',[p(5) q(2) r],'string','f^a', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f^a'')');  % 详解: 执行语句
uicontrol('pos',[p(6) q(2) r],'string','f(x+a)', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f(x+a)'')');  % 详解: 执行语句
uicontrol('pos',[p(7) q(2) r],'string','f(x*a)', ...  % 详解: 执行语句
   'CallBack','funtool(''row2'',''f(x*a)'')');  % 详解: 执行语句

uicontrol('pos',[p(1) q(3) r],'string','f + g', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''f+g'')');  % 详解: 执行语句
uicontrol('pos',[p(2) q(3) r],'string','f - g', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''f-g'')');  % 详解: 执行语句
uicontrol('pos',[p(3) q(3) r],'string','f * g', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''f*g'')');  % 详解: 执行语句
uicontrol('pos',[p(4) q(3) r],'string','f / g', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''f/g'')');  % 详解: 执行语句
uicontrol('pos',[p(5) q(3) r],'string','f(g)', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''f(g)'')');  % 详解: 执行语句
uicontrol('pos',[p(6) q(3) r],'string','g = f', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''g=f'')');  % 详解: 执行语句
uicontrol('pos',[p(7) q(3) r],'string','swap', ...  % 详解: 执行语句
    'CallBack','funtool(''row3'',''swap'')');  % 详解: 执行语句

uicontrol('pos',[p(1) q(4) r],'string','Insert','CallBack','funtool Insert');  % 详解: 调用函数：uicontrol('pos',[p(1) q(4) r],'string','Insert','CallBack','funtool Insert')
uicontrol('pos',[p(2) q(4) r],'string','Cycle','CallBack','funtool Cycle');  % 详解: 调用函数：uicontrol('pos',[p(2) q(4) r],'string','Cycle','CallBack','funtool Cycle')
uicontrol('pos',[p(3) q(4) r],'string','Delete','CallBack','funtool Delete');  % 详解: 调用函数：uicontrol('pos',[p(3) q(4) r],'string','Delete','CallBack','funtool Delete')
uicontrol('pos',[p(4) q(4) r],'string','Reset','Tag','reset', ...  % 详解: 执行语句
    'UserData', init, 'CallBack', 'funtool Reset');  % 详解: 执行语句
uicontrol('pos',[p(5) q(4) r],'string','Help', ...  % 详解: 执行语句
    'CallBack','helpwin funtool');  % 详解: 执行语句
uicontrol('pos',[p(6) q(4) r],'string','Demo', ...  % 详解: 执行语句
    'CallBack','funtool Demo');  % 详解: 执行语句
uicontrol('pos',[p(7) q(4) r],'string','Close', ...  % 详解: 执行语句
    'CallBack','funtool close');  % 详解: 执行语句


else  % 详解: 条件判断：else 分支
switch keyword  % 详解: 多分支选择：switch (keyword)

case 'row1'  % 详解: 分支：case 'row1'
   figp = findobj(0,'tag','figp');  % 详解: 赋值：将 findobj(...) 的结果保存到 figp
   fhndl = findobj(figp,'tag','fobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 fhndl
   f = get(fhndl,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 f
   x = get(findobj(figp,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf

   switch varargin{1}  % 详解: 多分支选择：switch (varargin{1})
     case 'diff'  % 详解: 分支：case 'diff'
       f = diff(f);  % 详解: 赋值：将 diff(...) 的结果保存到 f
     case 'int'  % 详解: 分支：case 'int'
       f = int(f);  % 详解: 赋值：将 int(...) 的结果保存到 f
     case 'simple'  % 详解: 分支：case 'simple'
       f = simple(f);  % 详解: 赋值：将 simple(...) 的结果保存到 f
     case 'num'  % 详解: 分支：case 'num'
       [f,ans] = numden(f);  % 详解: 执行语句
     case 'den'  % 详解: 分支：case 'den'
       [ans,f] = numden(f);  % 详解: 执行语句
     case '1/f'  % 详解: 分支：case '1/f'
       f = 1/f;  % 详解: 赋值：计算表达式并保存到 f
     case 'finverse'  % 详解: 分支：case 'finverse'
       f = finverse(f);  % 详解: 赋值：将 finverse(...) 的结果保存到 f
   end  % 详解: 执行语句

   set(fhndl,'UserData',f);  % 详解: 调用函数：set(fhndl,'UserData',f)
   ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)
   blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks
   set(findobj(figp,'tag','Sf'),'string',[blanks char(f)]);  % 详解: 调用函数：set(findobj(figp,'tag','Sf'),'string',[blanks char(f)])
   
case 'row2'  % 详解: 分支：case 'row2'
   figp = findobj(0,'tag','figp');  % 详解: 赋值：将 findobj(...) 的结果保存到 figp
   fhndl = findobj(figp,'tag','fobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 fhndl
   f = get(fhndl,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 f
   x = get(findobj(figp,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   a = get(findobj(figp,'tag','aobj'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 a
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf

   switch varargin{1}  % 详解: 多分支选择：switch (varargin{1})
     case 'f+a'  % 详解: 分支：case 'f+a'
       f = f+a;  % 详解: 赋值：计算表达式并保存到 f
     case 'f-a'  % 详解: 分支：case 'f-a'
       f = f-a;  % 详解: 赋值：计算表达式并保存到 f
     case 'f*a'  % 详解: 分支：case 'f*a'
       f = f*a;  % 详解: 赋值：计算表达式并保存到 f
     case 'f/a'  % 详解: 分支：case 'f/a'
       f = f/a;  % 详解: 赋值：计算表达式并保存到 f
     case 'f^a'  % 详解: 分支：case 'f^a'
       f = f^a;  % 详解: 赋值：计算表达式并保存到 f
     case 'f(x+a)'  % 详解: 分支：case 'f(x+a)'
       f = subs(f,sym('x'),sym('x')+a);  % 详解: 赋值：将 subs(...) 的结果保存到 f
     case 'f(x*a)', ...  % 详解: 分支：case 'f(x*a)', ...
       f = subs(f,sym('x'),sym('x')*a);  % 详解: 赋值：将 subs(...) 的结果保存到 f
   end  % 详解: 执行语句

   set(fhndl,'UserData',f);  % 详解: 调用函数：set(fhndl,'UserData',f)
   ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)
   blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks
   set(findobj(figp,'tag','Sf'),'string',[blanks char(f)]);  % 详解: 调用函数：set(findobj(figp,'tag','Sf'),'string',[blanks char(f)])
   
case 'row3'  % 详解: 分支：case 'row3'
   blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks

   figp = findobj(0,'tag','figp');  % 详解: 赋值：将 findobj(...) 的结果保存到 figp
   fhndl = findobj(figp,'tag','fobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 fhndl
   ghndl = findobj(figp,'tag','gobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 ghndl
   f = get(fhndl,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 f
   g = get(ghndl,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 g
   x = get(findobj(figp,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   a = get(findobj(figp,'tag','aobj'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 a
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf
   figg = findobj(0,'tag','figg');  % 详解: 赋值：将 findobj(...) 的结果保存到 figg

   if strcmp(varargin{1}, 'g=f')  % 详解: 条件判断：if (strcmp(varargin{1}, 'g=f'))
     g = f;  % 详解: 赋值：计算表达式并保存到 g
     set(ghndl,'UserData',g);  % 详解: 调用函数：set(ghndl,'UserData',g)
     ezplot(g,x,figg)  % 详解: 调用函数：ezplot(g,x,figg)
     set(findobj(figp,'tag','Sg'),'string',[blanks char(g)]);  % 详解: 调用函数：set(findobj(figp,'tag','Sg'),'string',[blanks char(g)])

   elseif strcmp(varargin{1}, 'swap')  % 详解: 条件判断：elseif (strcmp(varargin{1}, 'swap'))
     h = f; f = g; g = h;  % 详解: 赋值：计算表达式并保存到 h
     set(fhndl,'UserData',f);  % 详解: 调用函数：set(fhndl,'UserData',f)
     ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)
     set(findobj(figp,'tag','Sf'),'string',[blanks char(f)]);  % 详解: 调用函数：set(findobj(figp,'tag','Sf'),'string',[blanks char(f)])
     set(ghndl,'UserData',g);  % 详解: 调用函数：set(ghndl,'UserData',g)
     ezplot(g,x,figg)  % 详解: 调用函数：ezplot(g,x,figg)
     set(findobj(figp,'tag','Sg'),'string',[blanks char(g)]);  % 详解: 调用函数：set(findobj(figp,'tag','Sg'),'string',[blanks char(g)])

   else  % 详解: 条件判断：else 分支
     switch varargin{1}  % 详解: 多分支选择：switch (varargin{1})
       case 'f+g'  % 详解: 分支：case 'f+g'
         f = f+g;  % 详解: 赋值：计算表达式并保存到 f
       case 'f-g'  % 详解: 分支：case 'f-g'
         f = f-g;  % 详解: 赋值：计算表达式并保存到 f
       case 'f*g'  % 详解: 分支：case 'f*g'
         f = f*g;  % 详解: 赋值：计算表达式并保存到 f
       case 'f/g'  % 详解: 分支：case 'f/g'
         f = f/g;  % 详解: 赋值：计算表达式并保存到 f
       case 'f(g)'  % 详解: 分支：case 'f(g)'
         f = compose(f,g);  % 详解: 赋值：将 compose(...) 的结果保存到 f
     end  % 详解: 执行语句

     set(fhndl,'UserData',f);  % 详解: 调用函数：set(fhndl,'UserData',f)
     ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)
     set(findobj(figp,'tag','Sf'),'string',[blanks char(f)]);  % 详解: 调用函数：set(findobj(figp,'tag','Sf'),'string',[blanks char(f)])
   end  % 详解: 执行语句

case 'Sfcallback'  % 详解: 分支：case 'Sfcallback'
   figp = findobj(0,'tag','figp');  % 详解: 赋值：将 findobj(...) 的结果保存到 figp
   f = sym(get(gco,'string'));  % 详解: 赋值：将 sym(...) 的结果保存到 f
   fhndl = findobj(figp,'tag','fobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 fhndl
   set(fhndl,'UserData',f);  % 详解: 调用函数：set(fhndl,'UserData',f)
   x = get(findobj(figp,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf
   ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)

case 'Sgcallback'  % 详解: 分支：case 'Sgcallback'
   figp = findobj(0,'tag','figp');  % 详解: 赋值：将 findobj(...) 的结果保存到 figp
   g = sym(get(gco,'string'));  % 详解: 赋值：将 sym(...) 的结果保存到 g
   ghndl = findobj(figp,'tag','gobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 ghndl
   set(ghndl,'UserData',g);  % 详解: 调用函数：set(ghndl,'UserData',g)
   x = get(findobj(figp,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   figg = findobj(0,'tag','figg');  % 详解: 赋值：将 findobj(...) 的结果保存到 figg
   ezplot(g,x,figg)  % 详解: 调用函数：ezplot(g,x,figg)

case 'Sacallback'  % 详解: 分支：case 'Sacallback'
   a = sym(get(gco,'String'));  % 详解: 赋值：将 sym(...) 的结果保存到 a
   set(findobj(gcf,'tag','aobj'),'UserData',a);  % 详解: 调用函数：set(findobj(gcf,'tag','aobj'),'UserData',a)

case 'Sxcallback'  % 详解: 分支：case 'Sxcallback'

   figp = findobj(0,'tag','figp');  % 详解: 赋值：将 findobj(...) 的结果保存到 figp
   x = get(gco,'String');  % 详解: 赋值：将 get(...) 的结果保存到 x

   if isempty(x=='['),  % 详解: 条件判断：if (isempty(x=='['),)
      x = ['[' x ']'];  % 详解: 赋值：计算表达式并保存到 x
      blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks
      set(gco,'string',[blanks x]),  % 详解: 执行语句
   end;  % 详解: 执行语句
   set(findobj(figp,'tag','xstr'),'UserData',x);  % 详解: 调用函数：set(findobj(figp,'tag','xstr'),'UserData',x)

   fhndl = findobj(figp,'tag','fobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 fhndl
   ghndl = findobj(figp,'tag','gobj');  % 详解: 赋值：将 findobj(...) 的结果保存到 ghndl
   f = get(fhndl,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 f
   g = get(ghndl,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 g
   a = get(findobj(figp,'tag','aobj'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 a
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf
   figg = findobj(0,'tag','figg');  % 详解: 赋值：将 findobj(...) 的结果保存到 figg
   
   ezplot(f,x,figf)  % 详解: 调用函数：ezplot(f,x,figf)
   ezplot(g,x,figg)  % 详解: 调用函数：ezplot(g,x,figg)

case 'Insert'  % 详解: 分支：case 'Insert'
   f = get(findobj(gcf,'tag','fobj'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 f
   x = get(findobj(gcf,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   fxlist = get(gcf,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 fxlist
   fxlist = str2mat(fxlist,[char(f) ';  ' x]);  % 详解: 赋值：将 str2mat(...) 的结果保存到 fxlist
   set(gcf,'UserData',fxlist);  % 详解: 调用函数：set(gcf,'UserData',fxlist)

case 'Cycle'  % 详解: 分支：case 'Cycle'

   figp = gcf;  % 详解: 赋值：计算表达式并保存到 figp
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf
   fxlist = get(figp,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 fxlist

   fx = fxlist(1,:);  % 详解: 赋值：将 fxlist(...) 的结果保存到 fx
   fx(find(fx==' ')) = [];  % 详解: 执行语句
   k = find(fx == ';');  % 详解: 赋值：将 find(...) 的结果保存到 k
   fstr = fx(1:k-1); f = sym(fstr);  % 详解: 赋值：将 fx(...) 的结果保存到 fstr
   set(findobj(figp,'tag','fobj'),'UserData',f);  % 详解: 调用函数：set(findobj(figp,'tag','fobj'),'UserData',f)
   x = fx(k+1:length(fx));  % 详解: 赋值：将 fx(...) 的结果保存到 x
   set(findobj(figp,'tag','xstr'),'UserData',x);  % 详解: 调用函数：set(findobj(figp,'tag','xstr'),'UserData',x)

   blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks
   set(findobj(figp,'tag','Sx'),'string',[blanks x]);  % 详解: 调用函数：set(findobj(figp,'tag','Sx'),'string',[blanks x])
   set(findobj(figp,'tag','Sf'),'string',[blanks fstr]);  % 详解: 调用函数：set(findobj(figp,'tag','Sf'),'string',[blanks fstr])
   ezplot(f,x,figf);  % 详解: 调用函数：ezplot(f,x,figf)
   k = [2:size(fxlist,1),1];  % 详解: 赋值：计算表达式并保存到 k
   fxlist = fxlist(k,:);  % 详解: 赋值：将 fxlist(...) 的结果保存到 fxlist
   set(figp,'UserData',fxlist);  % 详解: 调用函数：set(figp,'UserData',fxlist)

case 'Delete'  % 详解: 分支：case 'Delete'

   figp = gcf;  % 详解: 赋值：计算表达式并保存到 figp
   f = get(findobj(figp,'tag','fobj'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 f
   x = get(findobj(figp,'tag','xstr'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 x
   fxlist = get(figp,'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 fxlist

   fx = [char(f) ';' x];  % 详解: 赋值：计算表达式并保存到 fx
   fx(find(fx==' ')) = [];  % 详解: 执行语句
   for k = 1:size(fxlist,1),  % 详解: for 循环：迭代变量 k 遍历 1:size(fxlist,1),
      element = fxlist(k,:);  % 详解: 赋值：将 fxlist(...) 的结果保存到 element
      element(find(element==' ')) = [];  % 详解: 执行语句
      if strcmp(fx,element)  % 详解: 条件判断：if (strcmp(fx,element))
        fxlist(k,:) = [];  % 详解: 执行语句
        break  % 详解: 跳出循环：break
      end  % 详解: 执行语句
   end;  % 详解: 执行语句
   if isempty(fxlist), fxlist = '0-0;  [0,1]'; end  % 详解: 条件判断：if (isempty(fxlist), fxlist = '0-0;  [0,1]'; end)

   set(figp,'UserData',fxlist);  % 详解: 调用函数：set(figp,'UserData',fxlist)

case 'Reset'  % 详解: 分支：case 'Reset'
   blanks = '  ';  % 详解: 赋值：计算表达式并保存到 blanks

   figp = gcf;  % 详解: 赋值：计算表达式并保存到 figp
   figf = findobj(0,'tag','figf');  % 详解: 赋值：将 findobj(...) 的结果保存到 figf
   figg = findobj(0,'tag','figg');  % 详解: 赋值：将 findobj(...) 的结果保存到 figg
   init = get(findobj(figp,'tag','reset'),'UserData');  % 详解: 赋值：将 get(...) 的结果保存到 init

   set(findobj(figp,'tag','fobj'),'UserData',sym(init.f));  % 详解: 调用函数：set(findobj(figp,'tag','fobj'),'UserData',sym(init.f))
   set(findobj(figp,'tag','Sf'),'String',[blanks init.f]);  % 详解: 调用函数：set(findobj(figp,'tag','Sf'),'String',[blanks init.f])
   ezplot(sym(init.f),init.x,figf);  % 详解: 调用函数：ezplot(sym(init.f),init.x,figf)

   set(findobj(figp,'tag','gobj'),'UserData',sym(init.g));  % 详解: 调用函数：set(findobj(figp,'tag','gobj'),'UserData',sym(init.g))
   set(findobj(figp,'tag','Sg'),'String',[blanks init.g]);  % 详解: 调用函数：set(findobj(figp,'tag','Sg'),'String',[blanks init.g])
   ezplot(sym(init.g),init.x,figg);  % 详解: 调用函数：ezplot(sym(init.g),init.x,figg)

   set(findobj(figp,'tag','xstr'),'UserData',init.x);  % 详解: 调用函数：set(findobj(figp,'tag','xstr'),'UserData',init.x)
   set(findobj(figp,'tag','Sx'),'string',[blanks init.x]);  % 详解: 调用函数：set(findobj(figp,'tag','Sx'),'string',[blanks init.x])

   set(findobj(figp,'tag','aobj'),'UserData',sym(init.a));  % 详解: 调用函数：set(findobj(figp,'tag','aobj'),'UserData',sym(init.a))
   set(findobj(figp,'tag','Sa'),'string',[blanks init.a]);  % 详解: 调用函数：set(findobj(figp,'tag','Sa'),'string',[blanks init.a])

   set(figp,'UserData',init.l);  % 详解: 调用函数：set(figp,'UserData',init.l)

   set(findobj(figp,'Style','pushbutton'),'BackgroundColor','default');  % 详解: 调用函数：set(findobj(figp,'Style','pushbutton'),'BackgroundColor','default')

case 'close'  % 详解: 分支：case 'close'
   close(findobj(0,'tag','figf'));  % 详解: 调用函数：close(findobj(0,'tag','figf'))
   close(findobj(0,'tag','figg'));  % 详解: 调用函数：close(findobj(0,'tag','figg'))
   close(findobj(0,'tag','figp'));  % 详解: 调用函数：close(findobj(0,'tag','figp'))

case 'Demo'  % 详解: 分支：case 'Demo'


   prog = {'Reset','f/a','int f','f + g','1/f', ...  % 详解: 赋值：计算表达式并保存到 prog
           'int f','finv','int f','df/dx','num f'};  % 详解: 执行语句
   B = findobj(gcf,'Style','pushbutton');  % 详解: 赋值：将 findobj(...) 的结果保存到 B
   for k = 1: length(prog)  % 详解: for 循环：迭代变量 k 遍历 1: length(prog)
      currB = findobj(B,'String',prog{k});  % 详解: 赋值：将 findobj(...) 的结果保存到 currB
      set(currB,'BackgroundColor','white');  % 详解: 调用函数：set(currB,'BackgroundColor','white')
      eval(get(currB,'Callback'));  % 详解: 调用函数：eval(get(currB,'Callback'))
      pause(1)  % 详解: 调用函数：pause(1)
      set(currB,'BackgroundColor','default');  % 详解: 调用函数：set(currB,'BackgroundColor','default')
   end  % 详解: 执行语句

end  % 详解: 执行语句
end  % 详解: 执行语句




