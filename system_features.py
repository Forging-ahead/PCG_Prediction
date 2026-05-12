"""
门静脉系统级 / 联合特征 (system / joint features)
====================================================
不再只看单根血管，而是从血管之间的几何关系派生与门静脉压力梯度
(PPG / HVPG) 相关的系统特征。

参考文献依据:
  Peng et al. QIMS 2019 — 微血管 Murray cube law 偏差与 PP 相关
  Qi et al. Hepatology 2014 — CT 几何 + 1D 阻力可预测 vHVPG
  Kassab AJP-Heart 2006 — 血管树缩放定律
  Maruyama QIMS 2021 — 侧支体积/直径与 CSPH
  Mostafa Clin Exp Gastroenterol 2015 — D_SV/D_MPV 区分静脉曲张
  Berzigotti J Hepatol 2016 — 影像下门脉高压评估综述
  Ciurică Hypertension 2019 — 直径加权曲折度

特征分组:
  (A) Angles: SV-SMV, MPV-LPV/RPV, planarity, TIPS take-off
  (B) Diameter / Area ratios: Murray-3 偏差, area conservation, 不对称性
  (C) Length & Tortuosity ratios: 弯曲度比, 直径加权 tortuosity
  (D) Hydraulic-resistance-style: ∫dl/r⁴ 形式的 Poiseuille 阻力项
  (E) Topology / Asymmetry: 侧支负担, 锥度, 脾主导指数

所有特征都从已有的:
  centerline_profiles.json     (分段路径)
  centerline_pointwise_profiles.json (逐点 area / eq_diameter / inscribed_radius)
  + 平滑后的中心线 nodes
派生, 无需 CFD。

每个返回值若无法计算 (依赖段缺失等) 则为 None。
"""

import numpy as np

from utils import path_to_coords, path_physical_length


# ============================================================
# 工具
# ============================================================

def _safe_div(a, b, eps=1e-9):
    if a is None or b is None:
        return None
    if abs(b) < eps:
        return None
    return float(a) / float(b)


def _safe_get(d, *keys, default=None):
    """嵌套 dict 安全取值, 任一键缺失返回 default。"""
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _seg_pointwise(profile_data, seg_name):
    """从 centerline_pointwise_profiles.json 中取某段逐点剖面 (NaN-aware)。"""
    if profile_data is None:
        return None
    return profile_data.get(seg_name)


def _nan_mean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmean(arr))


def _nan_max(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmax(arr))


def _nan_min(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmin(arr))


# ============================================================
# 几何工具: 段端方向向量
# ============================================================

def _direction_from_branchpoint(seg_path, nodes, sample_dist_mm=10.0,
                                from_end=False):
    """
    从段端点 (默认起点) 沿段走 sample_dist_mm 取一点, 计算单位方向。

    from_end=True: 从段终点回退 sample_dist_mm 取方向 (指向终点)。
    返回单位向量, 方向退化时返回 None。
    """
    if seg_path is None or len(seg_path) < 2:
        return None
    coords = path_to_coords(seg_path, nodes)
    if from_end:
        coords = coords[::-1]

    base = coords[0]
    cumlen = 0.0
    sample_pt = coords[-1]
    for i in range(1, len(coords)):
        cumlen += np.linalg.norm(coords[i] - coords[i - 1])
        if cumlen >= sample_dist_mm:
            sample_pt = coords[i]
            break
    direction = sample_pt - base
    n = np.linalg.norm(direction)
    return direction / n if n > 1e-6 else None


def _angle_between(v1, v2):
    """两单位向量夹角 (度)。任一为 None 返回 None。"""
    if v1 is None or v2 is None:
        return None
    cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _coplanarity(v_axis, v_left, v_right):
    """
    左右分支平面与主干轴的偏角 (度)。
    分支平面法向 n = v_left × v_right, 与主干轴夹角偏离 90° 的程度.
    完美 T 型分叉 → 0°; 严重非平面 → 接近 90°。
    """
    if v_axis is None or v_left is None or v_right is None:
        return None
    plane_n = np.cross(v_left, v_right)
    nn = np.linalg.norm(plane_n)
    if nn < 1e-6:
        return None
    plane_n /= nn
    cos_a = float(np.clip(abs(np.dot(plane_n, v_axis)), 0.0, 1.0))
    angle = float(np.degrees(np.arcsin(cos_a)))  # 0 = 主干在平面内
    return angle


# ============================================================
# (A) 角度特征
# ============================================================

def _angle_features(seg_dict, nodes):
    """计算所有相关分叉角."""
    out = {}

    # SV-SMV (重复 extract_features 的, 这里以系统特征命名)
    mpv = _safe_get(seg_dict, 'mpv', 'path')
    sv = _safe_get(seg_dict, 'sv', 'path')
    smv = _safe_get(seg_dict, 'smv', 'path')
    lpv = _safe_get(seg_dict, 'lpv', 'path')
    rpv = _safe_get(seg_dict, 'rpv', 'path')
    tips = _safe_get(seg_dict, 'tips', 'path')

    # SV-SMV 汇合角: 两段从共同起点出发的方向夹角
    if sv and smv and len(sv) > 1 and len(smv) > 1 and sv[0] == smv[0]:
        d_sv = _direction_from_branchpoint(sv, nodes)
        d_smv = _direction_from_branchpoint(smv, nodes)
        out['angle_sv_smv'] = _angle_between(d_sv, d_smv)
    else:
        out['angle_sv_smv'] = None

    # MPV→LPV 角: MPV 入射 (从 lpv 起点回头) vs LPV 出射
    if mpv and lpv and len(mpv) > 1 and len(lpv) > 1:
        mpv_in = _direction_from_branchpoint(mpv, nodes, from_end=True)
        lpv_out = _direction_from_branchpoint(lpv, nodes)
        out['angle_mpv_lpv'] = _angle_between(mpv_in, lpv_out)
    else:
        out['angle_mpv_lpv'] = None

    if mpv and rpv and len(mpv) > 1 and len(rpv) > 1:
        mpv_in = _direction_from_branchpoint(mpv, nodes, from_end=True)
        rpv_out = _direction_from_branchpoint(rpv, nodes)
        out['angle_mpv_rpv'] = _angle_between(mpv_in, rpv_out)
    else:
        out['angle_mpv_rpv'] = None

    # LPV-RPV 之间的角
    if lpv and rpv and len(lpv) > 1 and len(rpv) > 1:
        d_lpv = _direction_from_branchpoint(lpv, nodes)
        d_rpv = _direction_from_branchpoint(rpv, nodes)
        out['angle_lpv_rpv'] = _angle_between(d_lpv, d_rpv)
    else:
        out['angle_lpv_rpv'] = None

    # MPV 分叉总角度 = LPV 角 + RPV 角 (粗略反映分叉张开)
    a1 = out['angle_mpv_lpv']
    a2 = out['angle_mpv_rpv']
    out['angle_mpv_bifurc_total'] = (a1 + a2) if (a1 is not None and a2 is not None) else None

    # MPV 分叉非平面性
    if mpv and lpv and rpv and len(mpv) > 1 and len(lpv) > 1 and len(rpv) > 1:
        mpv_axis = _direction_from_branchpoint(mpv, nodes, from_end=True)
        d_lpv = _direction_from_branchpoint(lpv, nodes)
        d_rpv = _direction_from_branchpoint(rpv, nodes)
        out['mpv_bifurc_planarity_deg'] = _coplanarity(mpv_axis, d_lpv, d_rpv)
    else:
        out['mpv_bifurc_planarity_deg'] = None

    # TIPS take-off (post-tips)
    if mpv and tips and len(mpv) > 1 and len(tips) > 1:
        mpv_in = _direction_from_branchpoint(mpv, nodes, from_end=True)
        tips_out = _direction_from_branchpoint(tips, nodes)
        out['angle_mpv_tips'] = _angle_between(mpv_in, tips_out)
    else:
        out['angle_mpv_tips'] = None

    return out


# ============================================================
# (B) 直径 / 面积比 (Murray, conservation, asymmetry)
# ============================================================

def _diameter_ratio_features(stat_features):
    """从段统计特征 (mean_diameter, mean_area) 派生比率。"""
    out = {}

    def D(seg):
        return stat_features.get(f"{seg}_mean_diameter")

    def A(seg):
        return stat_features.get(f"{seg}_mean_area")

    d_mpv, d_sv, d_smv = D('mpv'), D('sv'), D('smv')
    d_lpv, d_rpv = D('lpv'), D('rpv')
    d_lgv, d_pgv = D('lgv'), D('pgv')
    a_mpv, a_sv, a_smv = A('mpv'), A('sv'), A('smv')
    a_lpv, a_rpv = A('lpv'), A('rpv')

    # SV/SMV 不对称
    if d_sv is not None and d_smv is not None:
        denom = d_sv + d_smv
        out['sv_smv_diameter_asymmetry'] = _safe_div(d_sv - d_smv, denom)
    else:
        out['sv_smv_diameter_asymmetry'] = None

    # SV/MPV 直径比
    out['sv_mpv_diameter_ratio'] = _safe_div(d_sv, d_mpv)
    # SMV/MPV
    out['smv_mpv_diameter_ratio'] = _safe_div(d_smv, d_mpv)

    # 汇合 Murray-3: D_MPV^3 / (D_SV^3 + D_SMV^3) — 理想 ≈ 1
    if d_mpv is not None and d_sv is not None and d_smv is not None:
        num = d_mpv ** 3
        den = d_sv ** 3 + d_smv ** 3
        out['confluence_murray3_ratio'] = _safe_div(num, den)
        out['confluence_murray3_deviation'] = abs(out['confluence_murray3_ratio'] - 1.0) \
            if out['confluence_murray3_ratio'] is not None else None
    else:
        out['confluence_murray3_ratio'] = None
        out['confluence_murray3_deviation'] = None

    # 汇合面积守恒: A_MPV / (A_SV + A_SMV)
    if a_mpv is not None and a_sv is not None and a_smv is not None:
        out['confluence_area_ratio'] = _safe_div(a_mpv, a_sv + a_smv)
    else:
        out['confluence_area_ratio'] = None

    # MPV→LPV/RPV Murray-3
    if d_mpv is not None and d_lpv is not None and d_rpv is not None:
        out['mpv_bifurc_murray3_ratio'] = _safe_div(d_mpv ** 3, d_lpv ** 3 + d_rpv ** 3)
        out['mpv_bifurc_murray3_deviation'] = (
            abs(out['mpv_bifurc_murray3_ratio'] - 1.0)
            if out['mpv_bifurc_murray3_ratio'] is not None else None)
    else:
        out['mpv_bifurc_murray3_ratio'] = None
        out['mpv_bifurc_murray3_deviation'] = None

    # MPV→LPV/RPV 面积守恒
    if a_mpv is not None and a_lpv is not None and a_rpv is not None:
        out['mpv_bifurc_area_ratio'] = _safe_div(a_mpv, a_lpv + a_rpv)
    else:
        out['mpv_bifurc_area_ratio'] = None

    # LPV/RPV 不对称
    if d_lpv is not None and d_rpv is not None:
        out['lpv_rpv_diameter_asymmetry'] = _safe_div(
            d_lpv - d_rpv, d_lpv + d_rpv)
    else:
        out['lpv_rpv_diameter_asymmetry'] = None

    # 侧支/MPV 直径比
    out['lgv_mpv_diameter_ratio'] = _safe_div(d_lgv, d_mpv)
    out['pgv_mpv_diameter_ratio'] = _safe_div(d_pgv, d_mpv)

    # 脾主导指数 (r⁴ 加权): r_SV⁴/(r_SV⁴+r_SMV⁴) 近似流量分配
    if d_sv is not None and d_smv is not None:
        r_sv4 = (0.5 * d_sv) ** 4
        r_smv4 = (0.5 * d_smv) ** 4
        out['splenic_dominance_index'] = _safe_div(r_sv4, r_sv4 + r_smv4)
    else:
        out['splenic_dominance_index'] = None

    return out


# ============================================================
# (C) 长度 / Tortuosity 比
# ============================================================

def _length_tortuosity_features(seg_dict, stat_features, nodes):
    out = {}

    # 脾门→门静脉分叉 路径/弦比 (SV + MPV)
    sv = _safe_get(seg_dict, 'sv', 'path')
    mpv = _safe_get(seg_dict, 'mpv', 'path')
    if sv and mpv and len(sv) > 1 and len(mpv) > 1:
        sv_coords = path_to_coords(sv, nodes)
        mpv_coords = path_to_coords(mpv, nodes)
        path_len = (path_physical_length(sv, nodes)
                    + path_physical_length(mpv, nodes))
        # 弦长: 脾端端点 → MPV 终点
        chord = float(np.linalg.norm(sv_coords[-1] - mpv_coords[-1]))
        out['splenoportal_path_chord_ratio'] = _safe_div(path_len, chord)
    else:
        out['splenoportal_path_chord_ratio'] = None

    # 侧支总长 / MPV 长 (含 LGV / PGV)
    L_mpv = stat_features.get('mpv_length')
    L_collateral = 0.0
    has_any = False
    for cn in ['lgv', 'pgv']:
        L_c = stat_features.get(f"{cn}_length")
        if L_c is not None:
            L_collateral += L_c
            has_any = True
    out['collateral_length_mpv_ratio'] = (
        _safe_div(L_collateral, L_mpv) if has_any else None)

    # 直径加权 tortuosity (主干 + 主要分支): Σ τ_i · D_i^4 / Σ D_i^4
    weighted_num, weighted_den = 0.0, 0.0
    n_segs = 0
    for seg_name in ['mpv', 'sv', 'smv', 'lpv', 'rpv']:
        L_s = stat_features.get(f"{seg_name}_length")
        D_s = stat_features.get(f"{seg_name}_mean_diameter")
        T_s = stat_features.get(f"{seg_name}_tortuosity")
        if L_s is None or D_s is None or T_s is None:
            continue
        # tortuosity 字段是 arc/chord, 转换为 (arc/chord - 1) 作为弯曲程度
        tortuosity_score = max(T_s - 1.0, 0.0)
        w = D_s ** 4
        weighted_num += tortuosity_score * w
        weighted_den += w
        n_segs += 1
    if n_segs >= 2 and weighted_den > 0:
        out['diameter_weighted_tortuosity'] = float(weighted_num / weighted_den)
    else:
        out['diameter_weighted_tortuosity'] = None

    return out


# ============================================================
# (D) Hydraulic 阻力 (Poiseuille-like)
# ============================================================

def _segment_resistance_integral(profile, length_mm,
                                  use_inscribed=True, eps=1e-3):
    """
    沿一段中心线积分 ∫ dl / r^4, 单位: 1/mm^3 (省略 8μ/π 常数因子)。

    优先用 inscribed_radius (来自距离变换, 较稳健),
    否则回退 eq_diameter / 2。

    跳过 NaN / 半径太小的位置。返回 (R_int, n_used) 或 (None, 0)。
    """
    if profile is None:
        return None, 0
    n_pts = len(profile.get('arc_length_mm', []))
    if n_pts < 2 or length_mm is None or length_mm < 1e-6:
        return None, 0

    arc = np.asarray(profile['arc_length_mm'], dtype=float)
    r_inscribed = np.asarray(profile.get('inscribed_radius', []), dtype=float)
    eq_d = np.asarray(profile.get('eq_diameter', []), dtype=float)

    # 选择半径序列
    if use_inscribed and np.any(np.isfinite(r_inscribed) & (r_inscribed > eps)):
        r_arr = r_inscribed
    else:
        r_arr = 0.5 * eq_d  # 退回等效直径/2

    valid = np.isfinite(r_arr) & (r_arr > eps)
    if np.sum(valid) < 2:
        return None, 0

    arc_v = arc[valid]
    r_v = r_arr[valid]

    # 中点法积分
    dl = np.diff(arc_v)
    r_mid = 0.5 * (r_v[:-1] + r_v[1:])
    R = float(np.sum(dl / (r_mid ** 4)))
    return R, int(np.sum(valid))


def _hydraulic_features(stat_features, profile_data):
    """Poiseuille 阻力风格特征."""
    out = {}

    if profile_data is None:
        for k in ['mpv_resistance_integral', 'sv_resistance_integral',
                  'smv_resistance_integral', 'lpv_resistance_integral',
                  'rpv_resistance_integral', 'tips_resistance_integral',
                  'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
                  'mpv_effective_radius', 'tips_inflow_resistance_ratio']:
            out[k] = None
        return out

    # 各段阻力积分
    seg_R = {}
    for seg in ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips']:
        L = stat_features.get(f"{seg}_length")
        prof = _seg_pointwise(profile_data, seg)
        R, _ = _segment_resistance_integral(prof, L)
        seg_R[seg] = R
        out[f"{seg}_resistance_integral"] = R

    # 入流并联阻力 R_in = (1/R_SV + 1/R_SMV)^{-1}
    R_sv, R_smv = seg_R.get('sv'), seg_R.get('smv')
    if R_sv is not None and R_smv is not None and R_sv > 0 and R_smv > 0:
        out['inflow_parallel_resistance'] = float(
            1.0 / (1.0 / R_sv + 1.0 / R_smv))
    else:
        out['inflow_parallel_resistance'] = None

    # 入流阻力不对称 (脾侧 vs 肠系膜侧)
    if R_sv is not None and R_smv is not None and (R_sv + R_smv) > 0:
        out['inflow_resistance_asymmetry'] = float(
            (R_sv - R_smv) / (R_sv + R_smv))
    else:
        out['inflow_resistance_asymmetry'] = None

    # MPV 等效半径: r_eff^4 = L / R_int
    L_mpv = stat_features.get('mpv_length')
    R_mpv = seg_R.get('mpv')
    if L_mpv is not None and R_mpv is not None and R_mpv > 1e-12:
        out['mpv_effective_radius'] = float((L_mpv / R_mpv) ** 0.25)
    else:
        out['mpv_effective_radius'] = None

    # TIPS / 入流阻力比
    R_tips = seg_R.get('tips')
    R_par = out['inflow_parallel_resistance']
    if R_tips is not None and R_par is not None and R_par > 1e-12:
        out['tips_inflow_resistance_ratio'] = _safe_div(R_tips, R_par)
    else:
        out['tips_inflow_resistance_ratio'] = None

    return out


# ============================================================
# (E) 拓扑 / 不对称 / 主干形态
# ============================================================

def _topology_features(seg_dict, stat_features, profile_data, branch_points):
    out = {}

    # 侧支负担: 体积加权 = Σ (D_c^2 · L_c) / (D_MPV^2 · L_MPV)
    L_mpv = stat_features.get('mpv_length')
    D_mpv = stat_features.get('mpv_mean_diameter')
    burden = 0.0
    has_any = False
    for cn in ['lgv', 'pgv']:
        L_c = stat_features.get(f"{cn}_length")
        D_c = stat_features.get(f"{cn}_mean_diameter")
        if L_c is not None and D_c is not None:
            burden += (D_c ** 2) * L_c
            has_any = True
    if (has_any and L_mpv is not None and D_mpv is not None
            and (D_mpv ** 2) * L_mpv > 1e-9):
        out['collateral_burden_score'] = float(
            burden / ((D_mpv ** 2) * L_mpv))
    else:
        out['collateral_burden_score'] = 0.0 if (
            stat_features.get('has_compensation_vessel', 0) == 0) else None

    # 侧支根数
    n_collat = sum(1 for cn in ['lgv', 'pgv']
                    if seg_dict.get(cn) is not None)
    out['n_collaterals_detected'] = int(n_collat)

    # 整树分叉点数 / 单位 MPV 长度
    n_bp = len(branch_points) if branch_points is not None else None
    if n_bp is not None and L_mpv is not None and L_mpv > 1e-6:
        out['branchpoint_density_per_cm'] = float(n_bp / (L_mpv / 10.0))
    else:
        out['branchpoint_density_per_cm'] = None

    # MPV 锥度系数: (D_proximal - D_distal) / L
    pw_mpv = _seg_pointwise(profile_data, 'mpv')
    if pw_mpv is not None and L_mpv is not None and L_mpv > 1e-6:
        eq_d = np.asarray(pw_mpv.get('eq_diameter', []), dtype=float)
        valid = np.isfinite(eq_d) & (eq_d > 0)
        if np.any(valid):
            idx_first = int(np.argmax(valid))
            idx_last = len(eq_d) - 1 - int(np.argmax(valid[::-1]))
            d_prox = float(eq_d[idx_first])
            d_dist = float(eq_d[idx_last])
            out['mpv_taper_coefficient'] = float((d_prox - d_dist) / L_mpv)
            out['mpv_proximal_diameter'] = d_prox
            out['mpv_distal_diameter'] = d_dist
            out['mpv_min_max_diameter_ratio'] = _safe_div(
                _nan_min(eq_d[valid]), _nan_max(eq_d[valid]))
        else:
            out['mpv_taper_coefficient'] = None
            out['mpv_proximal_diameter'] = None
            out['mpv_distal_diameter'] = None
            out['mpv_min_max_diameter_ratio'] = None
    else:
        out['mpv_taper_coefficient'] = None
        out['mpv_proximal_diameter'] = None
        out['mpv_distal_diameter'] = None
        out['mpv_min_max_diameter_ratio'] = None

    # 整树分叉处面积守恒平均偏离: 仅汇合 + MPV 分叉点 (现有数据下)
    deviations = []
    a_mpv = stat_features.get('mpv_mean_area')
    a_sv = stat_features.get('sv_mean_area')
    a_smv = stat_features.get('smv_mean_area')
    a_lpv = stat_features.get('lpv_mean_area')
    a_rpv = stat_features.get('rpv_mean_area')
    if a_mpv and a_sv and a_smv:
        deviations.append(abs(a_mpv - (a_sv + a_smv)) / a_mpv)
    if a_mpv and a_lpv and a_rpv:
        deviations.append(abs(a_mpv - (a_lpv + a_rpv)) / a_mpv)
    out['tree_area_conservation_mean_dev'] = (
        float(np.mean(deviations)) if deviations else None)

    return out


# ============================================================
# 主入口
# ============================================================

def compute_system_features(seg_dict, stat_features, profile_data,
                             nodes, branch_points=None):
    """
    入口: 计算所有系统/联合特征。

    参数:
        seg_dict: dict, centerline_profiles.json 的 'segments' 子项
        stat_features: dict, extract_features 的 flat dict (含 mpv_length, ...)
        profile_data: dict 或 None, centerline_pointwise_profiles.json
        nodes: dict, 中心线节点 (id -> {x,y,z})
        branch_points: iterable of int, 分叉点 id

    返回:
        flat dict: { 'angle_sv_smv': ..., 'confluence_murray3_ratio': ..., ... }
    """
    out = {}
    out.update(_angle_features(seg_dict, nodes))
    out.update(_diameter_ratio_features(stat_features))
    out.update(_length_tortuosity_features(seg_dict, stat_features, nodes))
    out.update(_hydraulic_features(stat_features, profile_data))
    out.update(_topology_features(
        seg_dict, stat_features, profile_data, branch_points))
    return out


# 所有可能输出的键 (供 correlation_analysis.py 引用, 顺序固定)
SYSTEM_FEATURE_NAMES = [
    # (A) angles
    'angle_sv_smv',
    'angle_mpv_lpv', 'angle_mpv_rpv', 'angle_lpv_rpv',
    'angle_mpv_bifurc_total', 'mpv_bifurc_planarity_deg',
    'angle_mpv_tips',
    # (B) diameter / area ratios
    'sv_smv_diameter_asymmetry',
    'sv_mpv_diameter_ratio', 'smv_mpv_diameter_ratio',
    'confluence_murray3_ratio', 'confluence_murray3_deviation',
    'confluence_area_ratio',
    'mpv_bifurc_murray3_ratio', 'mpv_bifurc_murray3_deviation',
    'mpv_bifurc_area_ratio',
    'lpv_rpv_diameter_asymmetry',
    'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
    'splenic_dominance_index',
    # (C) length / tortuosity ratios
    'splenoportal_path_chord_ratio',
    'collateral_length_mpv_ratio',
    'diameter_weighted_tortuosity',
    # (D) hydraulic
    'mpv_resistance_integral', 'sv_resistance_integral',
    'smv_resistance_integral', 'lpv_resistance_integral',
    'rpv_resistance_integral', 'tips_resistance_integral',
    'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
    'mpv_effective_radius', 'tips_inflow_resistance_ratio',
    # (E) topology / asymmetry
    'collateral_burden_score', 'n_collaterals_detected',
    'branchpoint_density_per_cm',
    'mpv_taper_coefficient',
    'mpv_proximal_diameter', 'mpv_distal_diameter',
    'mpv_min_max_diameter_ratio',
    'tree_area_conservation_mean_dev',
]

# 每个系统特征的中文标签 (用于报告)
SYSTEM_FEATURE_LABELS_CN = {
    'angle_sv_smv': 'SV-SMV夹角',
    'angle_mpv_lpv': 'MPV-LPV夹角',
    'angle_mpv_rpv': 'MPV-RPV夹角',
    'angle_lpv_rpv': 'LPV-RPV夹角',
    'angle_mpv_bifurc_total': 'MPV分叉总角',
    'mpv_bifurc_planarity_deg': 'MPV分叉非平面度',
    'angle_mpv_tips': 'TIPS入射角',
    'sv_smv_diameter_asymmetry': 'SV-SMV直径不对称',
    'sv_mpv_diameter_ratio': 'SV/MPV直径比',
    'smv_mpv_diameter_ratio': 'SMV/MPV直径比',
    'confluence_murray3_ratio': '汇合处Murray³比',
    'confluence_murray3_deviation': '汇合Murray³偏离',
    'confluence_area_ratio': '汇合面积比',
    'mpv_bifurc_murray3_ratio': 'MPV分叉Murray³比',
    'mpv_bifurc_murray3_deviation': 'MPV分叉Murray³偏离',
    'mpv_bifurc_area_ratio': 'MPV分叉面积比',
    'lpv_rpv_diameter_asymmetry': 'LPV-RPV直径不对称',
    'lgv_mpv_diameter_ratio': 'LGV/MPV直径比',
    'pgv_mpv_diameter_ratio': 'PGV/MPV直径比',
    'splenic_dominance_index': '脾主导指数',
    'splenoportal_path_chord_ratio': '脾门路径/弦长比',
    'collateral_length_mpv_ratio': '侧支长度/MPV比',
    'diameter_weighted_tortuosity': '直径加权曲折度',
    'mpv_resistance_integral': 'MPV阻力积分',
    'sv_resistance_integral': 'SV阻力积分',
    'smv_resistance_integral': 'SMV阻力积分',
    'lpv_resistance_integral': 'LPV阻力积分',
    'rpv_resistance_integral': 'RPV阻力积分',
    'tips_resistance_integral': 'TIPS阻力积分',
    'inflow_parallel_resistance': '入流并联阻力',
    'inflow_resistance_asymmetry': '入流阻力不对称',
    'mpv_effective_radius': 'MPV等效半径',
    'tips_inflow_resistance_ratio': 'TIPS/入流阻力比',
    'collateral_burden_score': '侧支负担评分',
    'n_collaterals_detected': '侧支数量',
    'branchpoint_density_per_cm': '分叉点密度/cm',
    'mpv_taper_coefficient': 'MPV锥度系数',
    'mpv_proximal_diameter': 'MPV近端直径',
    'mpv_distal_diameter': 'MPV远端直径',
    'mpv_min_max_diameter_ratio': 'MPV最小/最大直径',
    'tree_area_conservation_mean_dev': '树面积守恒偏离',
}
