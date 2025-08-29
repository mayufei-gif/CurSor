import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 塔的参数
tower_height = 10
num_layers = 20  # 层数
radius = 1       # 塔的半径
max_tilt = 1.5   # 倾斜最大水平位移
max_bend = 1     # 弯曲最大偏移
max_twist = np.pi / 3  # 最大扭曲角度

# 每一层的高度
z = np.linspace(0, tower_height, num_layers)
theta = np.linspace(0, 2 * np.pi, 50)

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化函数，绘制静态塔的初始状态
def init():
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, tower_height)
    ax.set_title("Combined Deformation (Tilt, Bend, Twist)")
    return fig,

# 更新函数，逐帧绘制变形塔和中间线
def update(frame):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, tower_height)
    
    # 倾斜变形：计算每层的水平位移
    tilt_offset = max_tilt * frame / 100  # 根据帧数控制水平位移
    
    # 弯曲变形：计算每层的偏移量（中部弯曲最显著）
    bend_offset = max_bend * np.sin(np.pi * z / tower_height) * frame / 100
    
    # 扭曲变形：计算每层的旋转角度
    twist_angles = np.linspace(0, max_twist * frame / 100, num_layers)
    
    # 绘制塔的各层剖面（带有倾斜、弯曲、扭曲的变形）
    for i in range(num_layers):
        x = radius * np.cos(theta + twist_angles[i]) + tilt_offset * (z[i] / tower_height) + bend_offset[i]
        y = radius * np.sin(theta + twist_angles[i])
        ax.plot(x, y, z[i], color='b')
    
    # 绘制中间的线（随着塔的变形而变化）
    x_center = tilt_offset * (z / tower_height) + bend_offset
    y_center = np.zeros(num_layers)  # 中心线的y坐标保持为0
    ax.plot(x_center, y_center, z, color='r', lw=3, label='Center Line')
    
    # 标签与标题
    ax.legend()
    ax.set_title(f"Frame: {frame}")
    return fig,

# 创建动画
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False, interval=100)

# 展示动画
plt.show()
