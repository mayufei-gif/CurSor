% 文件: subv2ind.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function ndx = subv2ind(siz, subv)  % 详解: 执行语句


if isempty(subv)  % 详解: 条件判断：if (isempty(subv))
  ndx = [];  % 详解: 赋值：计算表达式并保存到 ndx
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

if isempty(siz)  % 详解: 条件判断：if (isempty(siz))
  ndx = 1;  % 详解: 赋值：计算表达式并保存到 ndx
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

[ncases ndims] = size(subv);  % 详解: 获取向量/矩阵尺寸


if all(siz==2)  % 详解: 条件判断：if (all(siz==2))
  twos = pow2(0:ndims-1);  % 详解: 赋值：将 pow2(...) 的结果保存到 twos
  ndx = ((subv-1) * twos(:)) + 1;  % 详解: 赋值：计算表达式并保存到 ndx
else  % 详解: 条件判断：else 分支
  cp = [1 cumprod(siz(1:end-1))]';  % 赋值：设置变量 cp  % 详解: 赋值：计算表达式并保存到 cp  % 详解: 赋值：计算表达式并保存到 cp
  ndx = (subv-1)*cp + 1;  % 详解: 赋值：计算表达式并保存到 ndx
end  % 详解: 执行语句


function d = bitv2dec(bits)  % 详解: 执行语句

[m n] = size(bits);  % 详解: 获取向量/矩阵尺寸
twos = pow2(n-1:-1:0);  % 详解: 赋值：将 pow2(...) 的结果保存到 twos
d = sum(bits .* twos(ones(m,1),:),2);  % 详解: 赋值：将 sum(...) 的结果保存到 d





