% 文件: PLOT.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%二维图（一元函数图）
%用法 plot(x,y,s) 其中x,y为向量，每一对分量代表一个数据点。
%                    s为表示颜色、连线和标记选择的字符串
%图形的线型，标记，颜色均可设定，常用有
%       颜色	   |                线型	                标记
% ---------------------      -------------------  -------------------
%   b          蓝(默认)	       -      实线(默认)	无(默认)
%   g           绿             --     虚线         *          星
%   r           红             :      粗点线       .          点
%   y           黄             -.     点划线       o          圈
%   m           洋红                               x          叉
%   c           青                                 +          十字
%   w           白                                 s          方块
%   k           黑                                 d          菱形
%                                                  v          下三角形
%                                                  ^          上三角形
%                                                  <          左三角形
%                                                  >          右三角形
%                                                  h          六角形
%                                                  p          五角形
%例如
%    x=0:0.2:2*pi;y=sin(x);
%    plot(x,y,'r*')
%
%PLOT   Linear plot. 
%   PLOT(X,Y) plots vector Y versus vector X. If X or Y is a matrix,
%   then the vector is plotted versus the rows or columns of the matrix,
%   whichever line up. 
%
%   PLOT(Y) plots the columns of Y versus their index.  % 中文: 绘制（y）绘制了Y与其索引的列。 |||如果y很复杂，则图（y）等效于绘图（真实（y），图像（y））。 |||在图的所有其他用途中，虚构部分被忽略。 |||可以使用|||获得各种线类型，绘图符号和颜色绘图（x，y，s），其中s是一个由一个元素制成的字符串|||从以下任何或所有3个Colunms中：||| y黄色。     点 - 实心||| m洋红色o圈子：点缀||| C CYAN X X -MARK-。    dashdot
%   If Y is complex, PLOT(Y) is equivalent to PLOT(real(Y),imag(Y)).  % 中文: r红色 + plus-虚线||| G绿色 *星||| B蓝色S广场||| W WHITE D DIAMOND
%   In all other uses of PLOT, the imaginary part is ignored.  % 中文: k黑色V三角形（下）||| ^三角形（向上）||| <三角形（左）
%
%   Various line types, plot symbols and colors may be obtained with
%   PLOT(X,Y,S) where S is a character string made from one element
%   from any or all the following 3 colunms:
%
%          y     yellow        .     point              -     solid
%          m     magenta       o     circle             :     dotted
%          c     cyan          x     x-mark             -.    dashdot 
%          r     red           +     plus               --    dashed   
%          g     green         *     star
%          b     blue          s     square
%          w     white         d     diamond
%          k     black         v     triangle (down)
%                              ^     triangle (up)
%                              <     triangle (left)
%                              >     triangle (right)  % 中文: >三角形（右）||| p pentagram
%                              p     pentagram  % 中文: h hexagram |||例如，绘图（x，y，'c+：'）绘制了一条用a plus |||的青色虚线在每个数据点；图（x，y，'bd'）在每个数据上绘制蓝色钻石|||点，但不绘制任何线路。 |||绘图（x1，y1，s1，x2，y2，s2，x3，y3，s3，...）结合了|||定义的图（x，y，s）三元组，x和y是向量或矩阵||| S是弦。 |||例如，绘图（x，y，'y  - '，x，y，'go'）用|||绘制数据两次实心黄线在数据点插值绿色圆圈。 |||图命令，如果未指定颜色，则可以自动使用||| Axes Colororder属性指定的颜色。  默认|||上表中列出了颜色系统的颜色系统，其中|||默认值是一条线的黄色，并且对于多行，可循环|||通过桌子上的前六种颜色。  对于单色系统，|||绘制循环在轴线级别的属性上。 |||情节将手柄的列向量返回到线对象，一个|||每行处理。
%                              h     hexagram
%                         
%   For example, PLOT(X,Y,'c+:') plots a cyan dotted line with a plus 
%   at each data point; PLOT(X,Y,'bd') plots blue diamond at each data 
%   point but does not draw any line.
%
%   PLOT(X1,Y1,S1,X2,Y2,S2,X3,Y3,S3,...) combines the plots defined by
%   the (X,Y,S) triples, where the X's and Y's are vectors or matrices 
%   and the S's are strings.  
%
%   For example, PLOT(X,Y,'y-',X,Y,'go') plots the data twice, with a
%   solid yellow line interpolating green circles at the data points.
%
%   The PLOT command, if no color is specified, makes automatic use of
%   the colors specified by the axes ColorOrder property.  The default
%   ColorOrder is listed in the table above for color systems where the
%   default is yellow for one line, and for multiple lines, to cycle
%   through the first six colors in the table.  For monochrome systems,
%   PLOT cycles over the axes LineStyleOrder property.
%
%   PLOT returns a column vector of handles to LINE objects, one
%   handle per line. 
%
%   The X,Y pairs, or X,Y,S triples, can be followed by   % 中文: x，y对或x，y，s三元组，后面是|||参数/值对指定其他属性|||线。 |||另请参见Semilogx，Semilogy，Loglog，网格，CLF，CLC，Title，||| Xlabel，Ylabel，轴，轴，保持，Colordef和子图。 |||如果NextPlot Axes属性为“替换”（固定为关闭），则绘图重置|||除位置以外的所有轴属性，其默认值|||删除所有轴儿童（线，补丁，文本，表面和|||图像对象），并将视图属性设置为[0 90]。
%   parameter/value pairs to specify additional properties   % 中文: $修订：5.7 $ $日期：1997/11/21 23:33:07 $ |||内置功能。
%   of the lines.  % 中文: clear; clos; z = 0：0.1：20; r = exp（-0.2*z）; th = 0.5*pi*z;
%                                   
%   See also SEMILOGX, SEMILOGY, LOGLOG, GRID, CLF, CLC, TITLE,  % 中文: x = r。 |||图3绘图线和点在3-D空间中。
%   XLABEL, YLABEL, AXIS, AXES, HOLD, COLORDEF, and SUBPLOT.  % 中文: plot3（）是图（）的三维类似物。

%   If the NextPlot axes property is "replace" (HOLD is off), PLOT resets   % 中文: plot3（x，y，z），其中x，y和z是相同长度的三个向量，|||通过三个空间绘制一条线，其坐标为|||的点x，y和z的元素。
%   all axes properties, except Position, to their default values,
%   deletes all axes children (line, patch, text, surface, and
%   image objects), and sets the View property to [0 90].

%   Copyright (c) 1984-98 by The MathWorks, Inc.  % 中文: Mathworks，Inc。版权（C）1984-98
%   $Revision: 5.7 $  $Date: 1997/11/21 23:33:07 $
%   Built-in function.



