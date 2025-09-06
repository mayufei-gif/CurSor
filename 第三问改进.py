# -*- coding: utf-8 -*-
"""
第三问（定向定速）—— 图论+贪心优化版
 - 固定 FY1 的航向角 ψ 与速度 vU（同一解中三弹共享，不可改变）
 - 候选生成包含窗口期/几何“在前方/在线段之间”的硬筛
 - 位集贪心选择三枚（带投放间隔≥1s），并做(1,1)交换精化
 - 写 result1.xlsx（3行；方向/速度三行相同），绘制3D与雷达图

依赖: numpy, pandas, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ================== 物理与场景常量 ==================
v_M    = 300.0     # 导弹速度 m/s
v_sink = 3.0       # 烟幕下沉 m/s
R      = 10.0      # 云幕半径 m
T_win  = 20.0      # 起爆后有效窗 s
g      = 9.8       # 重力 m/s^2

M0   = np.array([20000.0, 0.0, 2000.0])   # M1 初始
FY1  = np.array([17800.0, 0.0, 1800.0])   # FY1 初始
O    = np.array([0.0, 0.0, 0.0])          # 假目标
T    = np.array([0.0, 200.0, 0.0])        # 真目标

u_M  = (O - M0) / np.linalg.norm(O - M0)  # 指向 O 的单位向量
t_hit = np.linalg.norm(M0 - O) / v_M      # 命中时间 ~66.7 s

# ================ 约束参数（可调） ==================
LAMBDA_MIN   = 0.02   # 在线段之间的安全裕度
FRONT_MARGIN = 10.0   # 前置裕度 (m)
DT_GLOBAL    = 0.02   # 全局时间网格步长
DROP_RANGE   = (0.5, 15.0)  # 投放时刻搜索范围 (s)
DROP_STEP    = 0.1          # 投放粗步长
DELAY_LIST   = [2.0, 3.0, 4.0, 5.0]  # 起爆延时候选 (s)
MIN_GAP      = 1.0          # 投放间隔 (s)

# 外层起点（定向定速）的扫描网格（可适当加密/缩小）
PSI_LIST_DEG = list(range(150, 211, 10))  # 150°~210° 步10°
V_LIST       = [80.0, 100.0, 120.0, 140.0]

# ==================== 工具函数 =======================
def heading_unit(psi_rad):
    return np.array([np.cos(psi_rad), np.sin(psi_rad), 0.0])

def s_progress(X):
    """进度标尺: s(X) = (O - X)·u_M; 值越小越靠近O"""
    return float(np.dot(O - X, u_M))

def missile_pos(t):
    return M0 + v_M * t * u_M

def build_candidate_mask(psi, vU, t_drop, t_delay, dt=DT_GLOBAL):
    """为某个 (psi, vU, t_drop, t_delay) 构造遮蔽布尔掩码（全局时间轴），并做硬约束筛选"""
    b = t_drop + t_delay
    if b >= t_hit - 1e-12:
        return None

    hU = heading_unit(psi)
    P_drop = FY1 + vU * t_drop * hU
    P_expl = P_drop + vU * t_delay * hU + 0.5 * np.array([0.0, 0.0, -g]) * t_delay**2

    # 窗口期硬筛（投放/起爆“在导弹前方且在导弹与目标之间”）
    sM_drop = s_progress(missile_pos(t_drop))
    s_drop  = s_progress(P_drop)
    if s_drop > sM_drop - FRONT_MARGIN:
        return None

    sM_b = s_progress(missile_pos(b))
    s_ex = s_progress(P_expl)
    if not (0.0 <= s_ex <= sM_b - FRONT_MARGIN):
        return None

    # 相对起爆的有效窗
    t_max = min(T_win, t_hit - b)
    if t_max <= 0:
        return None
    tau  = np.arange(0.0, t_max + 1e-12, dt)
    t_abs= b + tau

    # 轨迹
    M_pos = M0 + (v_M * t_abs)[:, None] * u_M
    C_pos = P_expl + np.array([0.0, 0.0, -v_sink]) * tau[:, None]

    # 线段最近距离 + 未截断 λ_raw
    AB = (T - M_pos)            # shape (m,3)
    AC = (C_pos - M_pos)
    AB2 = np.sum(AB*AB, axis=1)
    AB2 = np.where(AB2 < 1e-12, 1e-12, AB2)
    lam_raw = np.sum(AC*AB, axis=1) / AB2
    Q = M_pos + lam_raw[:, None] * AB
    d = np.linalg.norm(C_pos - Q, axis=1)

    mask_rel = (d <= R) & (lam_raw >= LAMBDA_MIN) & (lam_raw <= 1.0 - LAMBDA_MIN)
    if not np.any(mask_rel):
        return None

    # 写到全局时间轴
    N = int(np.round(t_hit / dt)) + 1
    global_mask = np.zeros(N, dtype=bool)
    start = int(np.round(b / dt))
    end   = min(start + len(mask_rel), N)
    if end > start:
        global_mask[start:end] = mask_rel[:(end - start)]

    return dict(
        t_drop=float(t_drop),
        t_delay=float(t_delay),
        t_burst=float(b),
        P_drop=P_drop,
        P_expl=P_expl,
        mask=global_mask,
        indiv_len=float(mask_rel.sum() * dt)
    )

def generate_candidates(psi, vU, drop_range=DROP_RANGE, drop_step=DROP_STEP, delay_list=DELAY_LIST):
    """批量生成候选，已含硬筛；返回 list[dict]"""
    cand = []
    grid = np.arange(drop_range[0], drop_range[1] + 1e-12, drop_step)
    for td in grid:
        for dly in delay_list:
            c = build_candidate_mask(psi, vU, td, dly)
            if c is not None:
                cand.append(c)
    return cand

def greedy_select_three(cands, dt=DT_GLOBAL):
    """贪心 + 投放间隔限制；返回 selected list 与并集长度"""
    if len(cands) == 0:
        return [], 0.0
    N = int(np.round(t_hit / dt)) + 1
    U = np.zeros(N, dtype=bool)  # 并集
    selected = []
    remain = list(range(len(cands)))

    for _ in range(3):
        best_id, best_gain = -1, -1
        for idx in remain:
            c = cands[idx]
            # 投放间隔限制
            ok = True
            for s in selected:
                if abs(c['t_drop'] - s['t_drop']) < MIN_GAP - 1e-9:
                    ok = False; break
            if not ok:
                continue
            gain = np.count_nonzero((~U) & c['mask'])
            if gain > best_gain:
                best_gain = gain
                best_id   = idx
        if best_id < 0:
            break
        # 选中
        selected.append(cands[best_id])
        U |= cands[best_id]['mask']
        remain.remove(best_id)

    L = float(np.count_nonzero(U) * dt)
    return selected, L

def one_one_swap_refine(selected, cands, dt=DT_GLOBAL, max_iter=20):
    """(1,1) 交换精化：尝试用未选候选替换已选之一以提升并集"""
    if len(selected) == 0:
        return selected, 0.0
    def union_len(sel):
        U = np.zeros(int(np.round(t_hit / dt)) + 1, dtype=bool)
        for s in sel: U |= s['mask']
        return float(np.count_nonzero(U) * dt)

    best_sel = selected[:]
    best_L   = union_len(best_sel)
    for _ in range(max_iter):
        improved = False
        for i in range(len(best_sel)):
            for c in cands:
                # 投放间隔限制（和其余两个保持 ≥1s）
                other = [best_sel[j] for j in range(len(best_sel)) if j != i]
                ok = True
                for s in other:
                    if abs(c['t_drop'] - s['t_drop']) < MIN_GAP - 1e-9:
                        ok = False; break
                if not ok: continue
                trial = other + [c]
                L = union_len(trial)
                if L > best_L + 1e-9:
                    best_sel, best_L = trial, L
                    improved = True
        if not improved: break
    return best_sel, best_L

def local_tweak(selected, psi, vU, dt=DT_GLOBAL):
    """对已选三枚在±0.2s内做细调（只调 t_drop），减少离散误差"""
    tweaked = []
    for s in selected:
        best = s
        best_len = s['indiv_len']
        for delta in np.arange(-0.2, 0.201, 0.02):
            td = max(DROP_RANGE[0], min(DROP_RANGE[1], s['t_drop'] + delta))
            newc = build_candidate_mask(psi, vU, td, s['t_delay'], dt)
            if newc is not None and newc['indiv_len'] > best_len + 1e-9:
                best = newc; best_len = newc['indiv_len']
        tweaked.append(best)
    # 交换精化一次
    tweaked, _ = one_one_swap_refine(tweaked, tweaked, dt)
    return tweaked

# ================== 主求解流程（外层定向定速） ==================
def solve_q3_fixed_heading_speed():
    best_global = None
    best_records = None
    best_tuple = None

    for psi_deg in PSI_LIST_DEG:
        psi = np.deg2rad(psi_deg % 360)
        for vU in V_LIST:
            # 生成候选（定向定速，三弹共用）
            cands = generate_candidates(psi, vU)
            if len(cands) == 0: 
                continue
            # 贪心+间隔
            sel, L = greedy_select_three(cands)
            if len(sel) < 3: 
                continue
            # (1,1)交换精化
            sel, L = one_one_swap_refine(sel, cands)
            # 局部微调
            sel = local_tweak(sel, psi, vU)
            # 重算并集
            _, L = greedy_select_three(sel)  # 用并集长度计算器
            if (best_global is None) or (L > best_global + 1e-9):
                best_global = L
                best_records = (psi_deg, vU, sel)
                best_tuple = (psi, vU, sel)

    return best_tuple, best_global

# ================== 结果写出 & 可视化 ==================
def save_result_excel(psi_deg, vU, selected):
    rows = []
    for i, s in enumerate(selected, 1):
        rows.append({
            '无人机运动方向(°)': float(psi_deg % 360),
            '无人机运动速度(m/s)': float(vU),
            '烟幕干扰弹编号': i,
            '投放点x': float(s['P_drop'][0]),
            '投放点y': float(s['P_drop'][1]),
            '投放点z': float(s['P_drop'][2]),
            '起爆点x': float(s['P_expl'][0]),
            '起爆点y': float(s['P_expl'][1]),
            '起爆点z': float(s['P_expl'][2]),
            '有效干扰时长(s)': round(s['indiv_len'], 3),
            '投放时刻(s)': round(s['t_drop'], 3),
            '起爆延时(s)': round(s['t_delay'], 3)
        })
    df = pd.DataFrame(rows, columns=[
        '无人机运动方向(°)','无人机运动速度(m/s)','烟幕干扰弹编号',
        '投放点x','投放点y','投放点z','起爆点x','起爆点y','起爆点z',
        '有效干扰时长(s)','投放时刻(s)','起爆延时(s)'
    ])
    df.to_excel('result1.xlsx', index=False)
    return df

def plot_3d_and_radar(psi_deg, vU, selected, total_union_len):
    psi = np.deg2rad(psi_deg % 360)
    hU  = heading_unit(psi)

    # 3D
    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection='3d')
    # 导弹
    tt  = np.linspace(0, t_hit, 200)
    mtraj = M0 + (v_M*tt)[:,None]*u_M
    ax.plot(mtraj[:,0], mtraj[:,1], mtraj[:,2], 'r-', lw=2.5, label='导弹M1')
    # FY1
    td = np.linspace(0, max([s['t_burst'] for s in selected])+1.0, 200)
    utraj = FY1 + (vU*td)[:,None]*hU
    ax.plot(utraj[:,0], utraj[:,1], utraj[:,2], 'b-', lw=2.0, label='FY1')

    cols = ['#2ca02c','#ff7f0e','#9467bd']
    for i,s in enumerate(selected,1):
        # 抛物线（投放->起爆）
        tb = np.linspace(0, s['t_delay'], 80)
        bomb = s['P_drop'] + (vU*tb)[:,None]*hU + 0.5*np.array([0,0,-g])*(tb[:,None]**2)
        ax.plot(bomb[:,0],bomb[:,1],bomb[:,2],'--',color=cols[i-1],lw=1.8,label=f'第{i}枚弹道')
        # 点
        ax.scatter(*s['P_drop'], c=cols[i-1], s=40, marker='o', label=f'投放{i}')
        ax.scatter(*s['P_expl'], c=cols[i-1], s=80, marker='^', edgecolors='k', label=f'起爆{i}')
    # 关键点
    ax.scatter(*M0, c='r', s=50, marker='o'); ax.text(*M0, 'M0', color='r')
    ax.scatter(*FY1, c='b', s=50, marker='s'); ax.text(*FY1,'FY1', color='b')
    ax.scatter(*O, c='k', s=50, marker='^');   ax.text(*O,'O(假)', color='k')
    ax.scatter(*T, c='gold', s=60, marker='D');ax.text(*T,'T(真)', color='goldenrod')

    ax.set_title(f'第三问定向定速最优三维轨迹 | 并集遮蔽≈{total_union_len:.2f}s', fontsize=12)
    ax.set_xlabel('X (m)', fontsize=10); ax.set_ylabel('Y (m)', fontsize=10); ax.set_zlabel('Z (m)', fontsize=10)
    ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('第三问_定向定速_三维.png', dpi=220, bbox_inches='tight')

    # 雷达图
    fig2 = plt.figure(figsize=(6,6)); ax2 = fig2.add_subplot(111, polar=True)
    vals = [ (psi_deg%360)/360.0, (vU-70.0)/70.0 ]
    for s in selected: vals += [ s['t_drop']/5.0, s['t_delay']/6.0 ]
    labels = ['航向角','速度','投放1','延1','投放2','延2','投放3','延3']
    th = np.linspace(0, 2*np.pi, len(vals)+1)
    vv = np.array(vals + [vals[0]])
    ax2.plot(th, vv, 'o-', lw=2.5, color='crimson'); ax2.fill(th, vv, alpha=0.25, color='crimson')
    ax2.set_xticks(th[:-1]); ax2.set_xticklabels(labels); ax2.set_ylim(0,1.0)
    ax2.set_title('参数分布雷达图（定向定速）', fontsize=12)
    plt.tight_layout(); plt.savefig('第三问_定向定速_雷达.png', dpi=220, bbox_inches='tight')
    plt.close('all')

# ================== 运行 ==================
if __name__ == '__main__':
    best, L = solve_q3_fixed_heading_speed()
    if best is None:
        print('未找到可行解，请适当放宽 DROP_RANGE/调整网格。')
    else:
        psi, vU, selected = best
        psi_deg = (np.rad2deg(psi)) % 360.0
        # 并集时长（再算一次以展示）
        sel, Lchk = greedy_select_three(selected)
        print(f'[定向定速] 最优 ψ={psi_deg:.2f}°, vU={vU:.1f} m/s, 并集遮蔽 ≈ {Lchk:.2f} s')
        df = save_result_excel(psi_deg, vU, selected)
        plot_3d_and_radar(psi_deg, vU, selected, Lchk)
        print('已写出 result1.xlsx，并生成 3D 与雷达图：')
        print('  - 第三问_定向定速_三维.png')
        print('  - 第三问_定向定速_雷达.png')
