% 文件: BNBGUI.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function BNBGUI(file)  % 详解: 函数定义：BNBGUI(file)
if nargin==0  % 详解: 条件判断：if (nargin==0)
   bnbguicb('init');  % 详解: 调用函数：bnbguicb('init')
   bnbguicb('main');  % 详解: 调用函数：bnbguicb('main')
elseif nargin==1  % 详解: 条件判断：elseif (nargin==1)
   bnbguicb('load',file);  % 详解: 调用函数：bnbguicb('load',file)
end;  % 详解: 执行语句




