"""
中心线剖面特征与临床指标逐点相关性分析（v3 - NaN-aware）
============================================================
新版变化:
  - 每个位置单独过滤 NaN 后再做 Spearman (跳过端点掩码)
  - 输入文件: centerline_pointwise_profiles.json (与分段 JSON 区分)
  - 支持的分支: MPV / SV / SMV / LPV / RPV / TIPS / LGV / PGV
    (任何在 JSON 中存在的非 None 段都会被分析)
  - 自动跳过该患者缺失的段

输出:
  1. pointwise_correlation.png  — 逐点相关性曲线 (显著区域高亮)
  2. profile_heatmap.png        — 剖面热力图 (按 target 排序)
  3. group_comparison.png       — 高/低 target 组剖面对比
  4. peak_correlations.csv      — 峰值相关性汇总
  5. profile_report.txt         — 文字报告
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 配置
# ============================================================

# 全部可能出现的分支 (按解剖顺序)
ALL_BRANCH_NAMES = ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips', 'lgv', 'pgv']

BRANCH_LABELS = {
    'mpv':  'MPV (门静脉主干)',
    'sv':   'SV (脾静脉)',
    'smv':  'SMV (肠系膜上静脉)',
    'lpv':  'LPV (左肝静脉)',
    'rpv':  'RPV (右肝静脉)',
    'tips': 'TIPS (支架)',
    'lgv':  'LGV (胃左静脉)',
    'pgv':  'PGV (胃后静脉)',
}

BRANCH_COLORS = {
    'mpv':  '#ef4444',
    'sv':   '#3b82f6',
    'smv':  '#f59e0b',
    'lpv':  '#a855f7',
    'rpv':  '#10b981',
    'tips': '#06b6d4',
    'lgv':  '#eab308',
    'pgv':  '#ec4899',
}

FEATURE_KEYS = ['area', 'eq_diameter', 'circularity', 'curvature',
                'perimeter', 'inscribed_radius']

FEATURE_LABELS = {
    'area':             '真实截面积 (mm²)',
    'eq_diameter':      '等效直径 (mm)',
    'circularity':      '截面圆度',
    'curvature':        '曲率 (1/mm)',
    'perimeter':        '截面周长 (mm)',
    'inscribed_radius': '内切圆半径 (mm)',
}

TARGET_INFO = {
    'PVP': {'cn': '门静脉压力', 'unit': 'mmHg'},
    'PCG': {'cn': '门静脉压力梯度', 'unit': 'mmHg'},
}

PROFILE_FILENAME = "centerline_pointwise_profiles.json"


# ============================================================
# 数据收集
# ============================================================

def _read_label(folder_path, target):
    txt = os.path.join(folder_path, "label", f"{target}.txt")
    if not os.path.exists(txt):
        return None
    try:
        with open(txt, 'r', encoding='utf-8') as f:
            for token in f.read().strip().replace(',', ' ').split():
                try:
                    return float(token)
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def collect_profiles(root_folder, target="PVP",
                     min_branch_coverage=0.3):
    """
    收集所有患者的剖面 + target 值。

    参数:
        min_branch_coverage: 一条分支需在至少这个比例的样本中存在
                             才参与分析 (例如 LGV 太罕见会被剔除)

    返回:
        data: list of dict
        n_points: int
        active_branches: list[str], 实际参与分析的分支
    """
    target = target.upper()
    print(f"\n收集剖面 + {target}...")

    subfolders = sorted(
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)))

    data = []
    n_points = None

    for folder in subfolders:
        folder_path = os.path.join(root_folder, folder)
        json_path = os.path.join(folder_path, PROFILE_FILENAME)

        if not os.path.exists(json_path):
            continue

        label_val = _read_label(folder_path, target)
        if label_val is None:
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
        except Exception:
            continue

        # 取每条剖面的点数 (用第一个非空段)
        if n_points is None:
            for branch in ALL_BRANCH_NAMES:
                p = profiles.get(branch)
                if p and 'position' in p:
                    n_points = len(p['position'])
                    break

        data.append({
            'name': folder,
            'target_value': label_val,
            'profiles': profiles,
        })

    if not data:
        print("  没有可用样本!")
        return data, n_points, []

    # 选活跃分支: 出现率 ≥ min_branch_coverage
    n_total = len(data)
    branch_coverage = {}
    for branch in ALL_BRANCH_NAMES:
        n_present = sum(1 for d in data
                        if d['profiles'].get(branch) is not None)
        branch_coverage[branch] = n_present / n_total

    active_branches = [b for b in ALL_BRANCH_NAMES
                       if branch_coverage[b] >= min_branch_coverage]
    dropped = [b for b in ALL_BRANCH_NAMES
               if 0 < branch_coverage[b] < min_branch_coverage]

    print(f"  收集到 {n_total} 个样本, 每条剖面 {n_points} 个点")
    print(f"  活跃分支 (覆盖率 ≥ {100*min_branch_coverage:.0f}%): "
          f"{[b.upper() for b in active_branches]}")
    if dropped:
        print(f"  剔除分支 (覆盖率不足): " +
              ", ".join(f"{b.upper()}({100*branch_coverage[b]:.0f}%)"
                        for b in dropped))

    return data, n_points, active_branches


# ============================================================
# 逐点相关性 (NaN-aware)
# ============================================================

def compute_pointwise_correlation(data, n_points, active_branches,
                                   target="PVP"):
    """
    对每个 (branch, feature, position) 算 Spearman 相关。
    端点掩码区 (NaN) 自动跳过, 每个位置单独按有效样本算 ρ。
    """
    target = target.upper()
    target_values = np.array([d['target_value'] for d in data])

    results = {}
    for branch in active_branches:
        results[branch] = {}
        for feat in FEATURE_KEYS:
            rho = np.full(n_points, np.nan)
            pval = np.full(n_points, np.nan)
            n_valid = np.zeros(n_points, dtype=int)

            for pos_idx in range(n_points):
                feat_vals, tgt_vals = [], []
                for i, d in enumerate(data):
                    prof = d['profiles'].get(branch)
                    if prof is None or feat not in prof:
                        continue
                    arr = prof[feat]
                    if pos_idx >= len(arr):
                        continue
                    v = arr[pos_idx]
                    # 跳过 None / NaN (含端点掩码)
                    if v is None:
                        continue
                    try:
                        v_f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(v_f):
                        continue
                    if not np.isfinite(target_values[i]):
                        continue
                    feat_vals.append(v_f)
                    tgt_vals.append(target_values[i])

                feat_arr = np.array(feat_vals, dtype=float)
                tgt_arr = np.array(tgt_vals, dtype=float)
                n_valid[pos_idx] = len(feat_arr)

                if len(feat_arr) >= 5 and np.std(feat_arr) > 1e-10:
                    try:
                        r, p = spearmanr(feat_arr, tgt_arr)
                        if np.isfinite(r) and np.isfinite(p):
                            rho[pos_idx] = r
                            pval[pos_idx] = p
                    except Exception:
                        pass

            results[branch][feat] = {
                'position': np.linspace(0, 1, n_points),
                'rho': rho, 'p_value': pval, 'n_valid': n_valid,
            }
    return results


def extract_peak_correlations(pw_results):
    """
    每个 (branch, feature) 组合的峰值相关。

    在中间 90% 区间找峰值 (跳过端点 5%, 避免端点掩码区间偶发的伪峰)。
    """
    rows = []
    for branch, feats in pw_results.items():
        for feat, res in feats.items():
            rho = res['rho']
            pval = res['p_value']
            n_valid = res['n_valid']
            valid_mask = ~np.isnan(rho)
            if not np.any(valid_mask):
                continue

            # 端点保护: 优先在中间 90% 找峰值
            n = len(rho)
            edge_margin = 0.05
            lo = int(n * edge_margin)
            hi = int(n * (1 - edge_margin))

            abs_rho_full = np.abs(rho)
            abs_rho_full[~valid_mask] = 0

            # 中间区间
            mid_abs = abs_rho_full[lo:hi].copy()
            if np.any(mid_abs > 0):
                peak_idx = int(np.argmax(mid_abs)) + lo
            else:
                # 中间无有效值, 退化为整段
                peak_idx = int(np.argmax(abs_rho_full))

            peak_pos = float(res['position'][peak_idx])
            peak_rho = float(rho[peak_idx])
            peak_p = float(pval[peak_idx])
            peak_n = int(n_valid[peak_idx])

            sig_mask = valid_mask & (pval < 0.05)
            sig_frac = (np.sum(sig_mask) / np.sum(valid_mask)
                        if np.sum(valid_mask) > 0 else 0)

            # 端点掩码影响: 多少位置 ρ 完全无法算
            n_masked = int(np.sum(~valid_mask))

            rows.append({
                'branch': branch.upper(),
                'feature': FEATURE_LABELS.get(feat, feat),
                'feature_key': feat,
                'peak_position': round(peak_pos, 3),
                'peak_position_pct': f"{peak_pos*100:.1f}%",
                'peak_rho': round(peak_rho, 4),
                'peak_p': round(peak_p, 5),
                'peak_n': peak_n,
                'n_masked_positions': n_masked,
                'sig_fraction': round(sig_frac, 3),
                'significant': peak_p < 0.05,
            })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('peak_p').reset_index(drop=True)
    return df


# ============================================================
# 可视化 1: 逐点相关性曲线
# ============================================================

def plot_pointwise_correlation(pw_results, active_branches,
                                output_dir, target="PVP"):
    target = target.upper()
    target_cn = TARGET_INFO.get(target, {}).get('cn', target)
    print("\n绘制逐点相关性曲线...")

    n_branches = len(active_branches)
    if n_branches == 0:
        return

    fig, axes = plt.subplots(len(FEATURE_KEYS), n_branches,
                              figsize=(5 * n_branches, 4 * len(FEATURE_KEYS)))
    if n_branches == 1:
        axes = axes[:, np.newaxis]
    if len(FEATURE_KEYS) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'逐点 Spearman 相关性: 剖面特征 vs {target}\n'
                 f'Pointwise Correlation: Profile Features vs {target_cn}',
                 fontsize=16, fontweight='bold', y=1.02)

    for fi, feat in enumerate(FEATURE_KEYS):
        for bi, branch in enumerate(active_branches):
            ax = axes[fi, bi]
            res = pw_results[branch][feat]
            pos, rho, pval = res['position'], res['rho'], res['p_value']
            color = BRANCH_COLORS.get(branch, '#64748b')

            ax.plot(pos * 100, rho, color=color, linewidth=2, zorder=3)

            sig_mask = (~np.isnan(pval)) & (pval < 0.05)
            if np.any(sig_mask):
                ax.fill_between(pos * 100, rho, 0,
                                where=sig_mask, alpha=0.25, color=color,
                                label='p < 0.05', zorder=2)

            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=0.3, color='gray', linewidth=0.5,
                       linestyle='--', alpha=0.5)
            ax.axhline(y=-0.3, color='gray', linewidth=0.5,
                       linestyle='--', alpha=0.5)

            valid = ~np.isnan(rho)
            if np.any(valid):
                # 端点保护峰值
                n = len(rho)
                lo, hi = int(n * 0.05), int(n * 0.95)
                abs_rho = np.abs(rho.copy())
                abs_rho[~valid] = 0
                mid_abs = abs_rho[lo:hi].copy()
                if np.any(mid_abs > 0):
                    peak_idx = int(np.argmax(mid_abs)) + lo
                else:
                    peak_idx = int(np.argmax(abs_rho))
                ax.plot(pos[peak_idx] * 100, rho[peak_idx], 'o',
                        color='red', markersize=8, zorder=5)
                ax.annotate(f'ρ={rho[peak_idx]:.3f}\n@{pos[peak_idx]*100:.0f}%',
                            xy=(pos[peak_idx] * 100, rho[peak_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=8, fontweight='bold', color='red',
                            arrowprops=dict(arrowstyle='->', color='red', lw=1))

            ax.set_xlim(0, 100)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if fi == 0:
                ax.set_title(BRANCH_LABELS.get(branch, branch),
                             fontsize=11, fontweight='bold', color=color)
            if bi == 0:
                ax.set_ylabel(f'{FEATURE_LABELS[feat]}\nSpearman ρ',
                              fontsize=10)
            if fi == len(FEATURE_KEYS) - 1:
                ax.set_xlabel('归一化位置 (%)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pointwise_correlation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 可视化 2: 剖面热力图 (按 target 排序)
# ============================================================

def plot_profile_heatmaps(data, active_branches, output_dir, target="PVP"):
    target = target.upper()
    print("\n绘制剖面热力图...")

    sorted_data = sorted(data, key=lambda d: d['target_value'])
    n_branches = len(active_branches)
    if n_branches == 0:
        return

    feats_to_plot = ['area', 'eq_diameter']

    fig, axes = plt.subplots(len(feats_to_plot), n_branches,
                              figsize=(6 * n_branches, 5 * len(feats_to_plot)))
    if n_branches == 1:
        axes = axes[:, np.newaxis]
    if len(feats_to_plot) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'剖面热力图 (按 {target} 排序)',
                 fontsize=16, fontweight='bold', y=1.02)

    for fi, feat in enumerate(feats_to_plot):
        for bi, branch in enumerate(active_branches):
            ax = axes[fi, bi]
            rows, valid_targets = [], []
            for d in sorted_data:
                prof = d['profiles'].get(branch)
                if prof and feat in prof:
                    rows.append(prof[feat])
                    valid_targets.append(d['target_value'])

            if not rows:
                ax.text(0.5, 0.5, '无数据', transform=ax.transAxes,
                        ha='center', va='center')
                continue

            # 转 float 数组, NaN 保留 (matplotlib 会以白色显示)
            matrix = np.array(rows, dtype=float)
            # 用 masked array 让 NaN 显示为透明 (避免 imshow 把 NaN 当 0)
            masked = np.ma.masked_invalid(matrix)
            cmap = plt.cm.YlOrRd.copy()
            cmap.set_bad(color='white', alpha=0.5)

            im = ax.imshow(masked, aspect='auto', cmap=cmap,
                           extent=[0, 100, len(rows), 0])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=FEATURE_LABELS.get(feat, feat))

            n_ticks = min(8, len(valid_targets))
            tick_positions = np.linspace(0, len(valid_targets) - 1,
                                         n_ticks, dtype=int)
            ax.set_yticks(tick_positions + 0.5)
            ax.set_yticklabels([f'{valid_targets[i]:.0f}'
                                for i in tick_positions], fontsize=8)

            if fi == 0:
                ax.set_title(BRANCH_LABELS.get(branch, branch),
                             fontsize=11, fontweight='bold',
                             color=BRANCH_COLORS.get(branch, '#64748b'))
            if bi == 0:
                ax.set_ylabel(f'{FEATURE_LABELS[feat]}\n{target} →',
                              fontsize=10)
            if fi == len(feats_to_plot) - 1:
                ax.set_xlabel('归一化位置 (%)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'profile_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 可视化 3: 高/低 target 组对比 (NaN-aware)
# ============================================================

def plot_group_comparison(data, active_branches, output_dir, target="PVP"):
    target = target.upper()
    unit = TARGET_INFO.get(target, {}).get('unit', '')
    print("\n绘制高/低组对比...")

    target_vals = np.array([d['target_value'] for d in data])
    median_val = np.median(target_vals)
    low_data = [d for d, m in zip(data, target_vals <= median_val) if m]
    high_data = [d for d, m in zip(data, target_vals > median_val) if m]
    print(f"  低组: n={len(low_data)}, {target}≤{median_val:.1f}")
    print(f"  高组: n={len(high_data)}, {target}>{median_val:.1f}")

    feats_to_plot = ['area', 'eq_diameter', 'circularity', 'curvature']
    n_branches = len(active_branches)
    if n_branches == 0:
        return

    fig, axes = plt.subplots(len(feats_to_plot), n_branches,
                              figsize=(6 * n_branches, 4 * len(feats_to_plot)))
    if n_branches == 1:
        axes = axes[:, np.newaxis]
    if len(feats_to_plot) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'高/低 {target} 组剖面对比 (中位数={median_val:.1f}{unit})',
                 fontsize=16, fontweight='bold', y=1.02)

    for fi, feat in enumerate(feats_to_plot):
        for bi, branch in enumerate(active_branches):
            ax = axes[fi, bi]

            for grp, label, color, alpha in [
                (low_data, f'低{target}', '#3b82f6', 0.2),
                (high_data, f'高{target}', '#ef4444', 0.2),
            ]:
                values = []
                for d in grp:
                    prof = d['profiles'].get(branch)
                    if prof and feat in prof:
                        values.append(prof[feat])
                if not values:
                    continue
                # 转 float 数组, NaN 用 nanmean/nanstd 跳过
                matrix = np.array(values, dtype=float)
                if not np.any(np.isfinite(matrix)):
                    continue
                mean = np.nanmean(matrix, axis=0)
                std = np.nanstd(matrix, axis=0)
                x = np.linspace(0, 100, len(mean))
                ax.plot(x, mean, color=color, linewidth=2, label=label)
                ax.fill_between(x, mean - std, mean + std,
                                color=color, alpha=alpha)

            # Mann-Whitney 显著点 (NaN-aware)
            n_pts = None
            for d in data:
                prof = d['profiles'].get(branch)
                if prof and feat in prof:
                    n_pts = len(prof[feat])
                    break

            if n_pts:
                sig_positions = []
                for pi in range(n_pts):
                    low_vals = []
                    for d in low_data:
                        prof = d['profiles'].get(branch)
                        if prof and feat in prof and pi < len(prof[feat]):
                            v = prof[feat][pi]
                            if v is not None:
                                try:
                                    v_f = float(v)
                                    if np.isfinite(v_f):
                                        low_vals.append(v_f)
                                except (TypeError, ValueError):
                                    pass

                    high_vals = []
                    for d in high_data:
                        prof = d['profiles'].get(branch)
                        if prof and feat in prof and pi < len(prof[feat]):
                            v = prof[feat][pi]
                            if v is not None:
                                try:
                                    v_f = float(v)
                                    if np.isfinite(v_f):
                                        high_vals.append(v_f)
                                except (TypeError, ValueError):
                                    pass

                    if len(low_vals) >= 3 and len(high_vals) >= 3:
                        try:
                            _, p = mannwhitneyu(low_vals, high_vals,
                                                alternative='two-sided')
                            if p < 0.05:
                                sig_positions.append(pi / (n_pts - 1) * 100)
                        except Exception:
                            pass
                for sp in sig_positions:
                    ax.axvline(x=sp, color='gold', alpha=0.3, linewidth=1.5)

            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if fi == 0:
                ax.set_title(BRANCH_LABELS.get(branch, branch),
                             fontsize=11, fontweight='bold',
                             color=BRANCH_COLORS.get(branch, '#64748b'))
            if bi == 0:
                ax.set_ylabel(FEATURE_LABELS.get(feat, feat), fontsize=10)
            if fi == len(feats_to_plot) - 1:
                ax.set_xlabel('归一化位置 (%)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'group_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {path}")


# ============================================================
# 报告
# ============================================================

def generate_profile_report(peak_df, data, active_branches,
                             output_dir, target="PVP"):
    target = target.upper()
    target_cn = TARGET_INFO.get(target, {}).get('cn', target)
    unit = TARGET_INFO.get(target, {}).get('unit', '')
    print("\n生成报告...")

    target_vals = [d['target_value'] for d in data]

    # 端点掩码元信息 (从第一个有 _meta 的样本读)
    edge_pct = None
    edge_mm = None
    for d in data:
        meta = d['profiles'].get('_meta')
        if meta:
            edge_pct = meta.get('edge_margin_pct')
            edge_mm = meta.get('edge_margin_mm')
            break

    lines = [
        "=" * 70,
        f"中心线剖面特征与{target_cn}({target})逐点相关性报告",
        "=" * 70, "",
        f"样本数: {len(data)}",
        f"{target} 范围: {min(target_vals):.1f} - {max(target_vals):.1f} {unit}",
        f"{target} 均值: {np.mean(target_vals):.1f} ± {np.std(target_vals):.1f} {unit}",
        f"分析的分支: {[b.upper() for b in active_branches]}",
    ]
    if edge_pct is not None:
        lines.append(f"端点掩码: 比例 {edge_pct*100:.1f}%, 距离 {edge_mm:.1f} mm")
    lines += [
        "",
        "-" * 70,
        "峰值相关性 (按 p 值排序, 端点 5% 区间已排除):", "",
    ]

    if len(peak_df) > 0:
        sig_peaks = peak_df[peak_df['significant']]
        if len(sig_peaks) > 0:
            lines.append(f"  显著 (p<0.05): {len(sig_peaks)} / {len(peak_df)}")
            lines.append("")
            lines.append(f"  {'分支':>6s}  {'特征':>14s}  {'位置':>8s}  "
                         f"{'Spearman ρ':>12s}  {'p-value':>10s}  "
                         f"{'N':>4s}  {'显著区占比':>10s}")
            lines.append("  " + "-" * 80)
            for _, r in sig_peaks.iterrows():
                lines.append(f"  {r['branch']:>6s}  {r['feature']:>14s}  "
                             f"{r['peak_position_pct']:>8s}  "
                             f"{r['peak_rho']:>+12.4f}  {r['peak_p']:>10.5f}  "
                             f"{r['peak_n']:>4d}  "
                             f"{r['sig_fraction']*100:>9.1f}%")
        else:
            lines.append("  未发现显著逐点相关 (可能样本量不足)")
    else:
        lines.append("  无有效数据")

    lines += [
        "", "-" * 70,
        "分析说明:", "",
        "  ・每条分支被归一化到 [0%, 100%], 0%=起点 bp, 100%=末端",
        "  ・在每个百分位位置, 计算所有患者该位置特征值与目标值的 Spearman 相关",
        "  ・端点 5% 区间不参与峰值检索 (extract_profiles 还有更严格的 NaN 掩码)",
        "  ・'显著区占比' = p<0.05 的位置占整条分支的比例",
        "  ・'N' = 该位置有效样本数 (端点掩码区会比中段少)",
        "  ・高占比说明该特征沿分支大范围与目标相关, 不只是局部",
        "", "=" * 70,
    ]

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'profile_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  保存: {path}")
    print("\n" + report)


# ============================================================
# 主流程
# ============================================================

def _setup_matplotlib():
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans',
        'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def run_profile_analysis(root_folder, output_dir=None, target="PVP",
                          min_branch_coverage=0.3):
    """完整流程: 收集 → 逐点相关 → 可视化 → 报告。"""
    target = target.upper()
    if output_dir is None:
        output_dir = os.path.join(root_folder,
                                   f"profile_correlation_{target.lower()}")
    os.makedirs(output_dir, exist_ok=True)
    _setup_matplotlib()

    data, n_points, active_branches = collect_profiles(
        root_folder, target, min_branch_coverage)

    if len(data) < 5:
        print(f"样本数 ({len(data)}) 不足, 至少需要 5 个")
        return
    if not active_branches:
        print(f"无活跃分支 (覆盖率均不足 {100*min_branch_coverage:.0f}%)")
        return

    print(f"\n计算逐点相关性 ({len(data)} 样本 × {n_points} 位置 × "
          f"{len(active_branches)} 分支)...")
    pw_results = compute_pointwise_correlation(
        data, n_points, active_branches, target)

    peak_df = extract_peak_correlations(pw_results)
    csv_path = os.path.join(output_dir, 'peak_correlations.csv')
    peak_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  峰值 CSV: {csv_path}")

    plot_pointwise_correlation(pw_results, active_branches, output_dir, target)
    plot_profile_heatmaps(data, active_branches, output_dir, target)
    plot_group_comparison(data, active_branches, output_dir, target)
    generate_profile_report(peak_df, data, active_branches,
                             output_dir, target)

    print(f"\n{'='*60}")
    print(f"剖面分析完成! 结果: {output_dir}")
    print(f"{'='*60}")


# ============================================================
# 用户配置
# ============================================================

if __name__ == '__main__':
    TARGET = "PVP"
    ROOT_FOLDER = r"F:\PCG data\dataset\zhengzhou_vkan_qian47"
    OUTPUT_DIR = None  # None = 自动: root/profile_correlation_pvp/

    run_profile_analysis(ROOT_FOLDER, OUTPUT_DIR, TARGET)