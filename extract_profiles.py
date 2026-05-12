"""
中心线逐点剖面特征提取（v3 - 读取分段 JSON 驱动）
==================================================
不再做解剖识别, 直接读 centerline_profiles.json 拿到每段路径,
对每段提取逐点剖面 (面积/周长/直径/圆度/曲率/内切半径)。

输出文件: centerline_pointwise_profiles.json
        (注意: 与分段文件 centerline_profiles.json 区分)

支持的段:
    MPV / SV / SMV / LPV / RPV / TIPS / LGV / PGV
    (任何 segment_vessels.py 输出的非 None 段都会被处理)
"""

import os
import json
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
import trimesh
import trimesh.intersections

from utils import (load_tree, path_to_coords, voxelize_stl, physical_to_voxel)


# ============================================================
# 截面计算核心 (与原版一致)
# ============================================================

def _make_orthonormal_basis(normal):
    """为法向量构造正交基 (u, v)"""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, ref)
    u /= (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-15)
    return u, v


def _polygon_aspect_ratio(poly_coords):
    """
    用 PCA 估计多边形顶点的长短轴比 (aspect_ratio = √(λ_max / λ_min)).

    aspect_ratio = 1.0  圆形/正方形
    aspect_ratio ≈ 1.4  椭圆 (b/a=2/3, 真实血管常见)
    aspect_ratio > 4    显著拉长 (沿管轴薄片切, 或跨血管切)

    返回值在 [1, +∞), 顶点不足或退化时返回 999.0
    """
    pts = np.asarray(poly_coords, dtype=float)
    if len(pts) < 3:
        return 999.0
    if pts.ndim != 2 or pts.shape[1] < 2:
        return 999.0
    pts = pts[:, :2]
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # 协方差矩阵 (2x2)
    try:
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
    except Exception:
        return 999.0
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals[1] < 1e-12 or eigvals[0] < 1e-12:
        return 999.0
    return float(np.sqrt(eigvals[1] / eigvals[0]))


def _section_one(mesh, point, normal, max_eq_diameter=None,
                 return_ring=False, return_metrics=False):
    """
    用一个法线做截面, 返回截面几何 + 形状质量指标。

    流程: mesh_plane → 交线段 → 投影 2D → polygonize → 候选多边形过滤
    候选选择策略 (按优先级):
      1. 多边形必须包含中心点 (0,0) — 否则该多边形属于其他血管
      2. 包含中心的多边形中, 面积最小者 (避免选到合并的"图8"形状外环)
      3. 若无包含中心者, 退而求距中心最近的有效多边形

    形状质量指标 (用于上层做"自适应"过滤, 无需固定大小阈值):
      - aspect_ratio: PCA 长短轴比. 1.0 = 圆/正方; >4 通常是沿管轴薄片切或跨血管.
      - circularity:  4πA/P². 1.0 = 完美圆; <0.3 形状极不规则.

    防止边界效应 (邻近血管"渗透"):
      若 max_eq_diameter (一般取 1.6 ~ 2 倍局部内切直径) 给定,
      且最终候选多边形的等效直径 > max_eq_diameter, 视为污染并丢弃.

    参数:
        max_eq_diameter: float 或 None — 等效直径上界 (mm)
        return_ring:     是否同时返回 2D 多边形轮廓 (用于可视化)
        return_metrics:  是否同时返回 (aspect_ratio, circularity)

    返回 (按 flag 组合):
        默认                                            (area, peri)
        return_metrics=True                             (area, peri, AR, circ)
        return_ring=True                                (area, peri, ring_2d)
        return_ring=True, return_metrics=True           (area, peri, AR, circ, ring_2d)
        失败时各位置填 0/0/999/0/None
    """
    if return_ring and return_metrics:
        fail = (0.0, 0.0, 999.0, 0.0, None)
    elif return_metrics:
        fail = (0.0, 0.0, 999.0, 0.0)
    elif return_ring:
        fail = (0.0, 0.0, None)
    else:
        fail = (0.0, 0.0)

    try:
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal=normal, plane_origin=point)
        if lines is None or len(lines) == 0:
            return fail

        u, v = _make_orthonormal_basis(normal)

        segs_2d = []
        for seg in lines:
            r0, r1 = seg[0] - point, seg[1] - point
            p0 = (float(np.dot(r0, u)), float(np.dot(r0, v)))
            p1 = (float(np.dot(r1, u)), float(np.dot(r1, v)))
            if abs(p0[0] - p1[0]) > 1e-8 or abs(p0[1] - p1[1]) > 1e-8:
                segs_2d.append((p0, p1))

        if len(segs_2d) < 3:
            return fail

        from shapely.geometry import LineString, Point as SPoint
        from shapely.ops import polygonize, unary_union

        ls_list = [LineString([s[0], s[1]]) for s in segs_2d]
        merged = unary_union(ls_list)
        polys = list(polygonize(merged))

        if not polys:
            try:
                from shapely.ops import snap
            except ImportError:
                from shapely import snap
            snapped = snap(merged, merged, tolerance=0.05)
            polys = list(polygonize(snapped))

        if not polys:
            buffered = merged.buffer(0.01)
            if hasattr(buffered, 'geoms'):
                polys = list(buffered.geoms)
            elif buffered.area > 0:
                polys = [buffered]

        if not polys:
            return fail

        center = SPoint(0.0, 0.0)
        containing = [p for p in polys if p.is_valid and p.contains(center)]

        if containing:
            best = min(containing, key=lambda p: p.area)
        else:
            valid_polys = [p for p in polys if p.is_valid and p.area > 0]
            if not valid_polys:
                return fail
            best = min(valid_polys, key=lambda p: p.distance(center))

        area = float(best.area)
        peri = float(best.exterior.length)

        # 边界效应保护: 若给了内切直径上界, 直接拒绝越界候选
        if max_eq_diameter is not None and area > 0:
            eq_d = float(np.sqrt(4.0 * area / np.pi))
            if eq_d > max_eq_diameter:
                return fail

        # 形状指标
        ring_2d_list = list(best.exterior.coords)
        if return_metrics:
            aspect_ratio = _polygon_aspect_ratio(ring_2d_list)
            if peri > 1e-6:
                circularity = float(min(1.5, 4.0 * np.pi * area / (peri * peri)))
            else:
                circularity = 0.0

        if return_ring and return_metrics:
            return area, peri, aspect_ratio, circularity, ring_2d_list
        if return_metrics:
            return area, peri, aspect_ratio, circularity
        if return_ring:
            return area, peri, ring_2d_list
        return area, peri

    except Exception:
        return fail


def _generate_normal_candidates(normal, n_perturb=12, max_angle_deg=15):
    """生成扰动法线候选集 (确定性, 与可视化端共用)."""
    normal = normal / (np.linalg.norm(normal) + 1e-15)
    u, v = _make_orthonormal_basis(normal)
    candidates = [normal]
    for angle_frac in [1.0, 0.5]:
        tan_a = np.tan(np.radians(max_angle_deg * angle_frac))
        n_dirs = n_perturb if angle_frac == 1.0 else n_perturb // 2
        for i in range(n_dirs):
            theta = 2.0 * np.pi * i / n_dirs
            if angle_frac < 1.0:
                theta += np.pi / n_perturb
            pert = normal + tan_a * (np.cos(theta) * u + np.sin(theta) * v)
            pert /= np.linalg.norm(pert)
            candidates.append(pert)
    return candidates


def _shape_score(area, aspect_ratio, circularity):
    """
    综合评分 (越小越好):
      score = area × (1 + 1.5·max(0, AR-1.3)) × (1 + (1-min(1, circ)))

    目的: 同时偏好"面积小"和"形状圆". 真正垂直切管 = 两者兼得; 沿轴薄片 =
    AR 大被惩罚; 跨血管 = 圆度低被惩罚.
    """
    elong_pen = 1.0 + 1.5 * max(0.0, aspect_ratio - 1.3)
    irreg_pen = 1.0 + (1.0 - min(1.0, max(0.0, circularity)))
    return float(area) * elong_pen * irreg_pen


def _compute_cross_section(mesh, point, normal,
                           n_perturb=12, max_angle_deg=15,
                           max_eq_diameter=None,
                           max_aspect_ratio=4.0,
                           min_circularity=0.30,
                           return_normal=False):
    """
    鲁棒截面: 扰动法线 → 形状硬过滤 → 综合评分选最佳.

    自适应判定 (无固定面积阈值, 仅靠几何形状):
      硬剔除:
        1. aspect_ratio > max_aspect_ratio (默认 4.0): 沿轴向薄片切 / 跨血管切
        2. circularity   < min_circularity  (默认 0.30): 形状极不规则
        3. eq_diameter   > max_eq_diameter (若给定): 越界穿透

      综合打分 (见 _shape_score): area × elongation_pen × irregularity_pen
      选 score 最小的候选, 等价于"面积小且形状接近圆"的真正垂直切.

    若所有候选都被形状过滤掉 → 返回 0 (该点截面记为缺失, 后续插值或 NaN).
    这比"硬选一个明显错误的"对训练集更友好 — 缺失值上层可处理, 错误值会污染统计.

    参数:
        max_eq_diameter:     等效直径上界 (mm), 默认 None 不限.
        max_aspect_ratio:    硬剔除阈值, 默认 4.0.
        min_circularity:     硬剔除阈值, 默认 0.30.
        return_normal:       是否返回所选最佳法线 (供可视化复现).

    返回:
        (area, perimeter)              — return_normal=False
        (area, perimeter, best_normal) — return_normal=True
    """
    normal = normal / (np.linalg.norm(normal) + 1e-15)
    candidates = _generate_normal_candidates(normal, n_perturb, max_angle_deg)

    best_score = float('inf')
    best_area, best_peri, best_normal = 0.0, 0.0, normal
    for n in candidates:
        a, p, ar, circ = _section_one(
            mesh, point, n,
            max_eq_diameter=max_eq_diameter,
            return_metrics=True)
        if a <= 0:
            continue
        # 形状硬过滤
        if ar > max_aspect_ratio or circ < min_circularity:
            continue
        score = _shape_score(a, ar, circ)
        if score < best_score:
            best_score = score
            best_area, best_peri, best_normal = a, p, n

    if best_score == float('inf'):
        # 所有候选均不合格 → 该点截面无效
        if return_normal:
            return 0.0, 0.0, normal
        return 0.0, 0.0

    if return_normal:
        return best_area, best_peri, best_normal
    return best_area, best_peri


def _compute_tangents(coords, smooth_window=5):
    """
    中心线每点的切线方向.

    使用 ±half 邻居端点连线作为切线 (default window=5 ⇒ ±2 邻居),
    比 3 点中心差分更平滑 — 在中心线轻微抖动 / 分叉点近邻处更稳定,
    避免切平面与血管轴近似平行造成"沿轴薄片切"的错误截面.
    """
    M = len(coords)
    tangents = np.zeros((M, 3))
    half = max(1, smooth_window // 2)
    for i in range(M):
        lo = max(0, i - half)
        hi = min(M - 1, i + half)
        if hi == lo:
            tangents[i] = np.array([0, 0, 1])
            continue
        t = coords[hi] - coords[lo]
        norm = np.linalg.norm(t)
        tangents[i] = t / norm if norm > 1e-10 else np.array([0, 0, 1])
    return tangents


def _remove_local_outliers(area, perimeter, eq_diameter,
                           window=15, mad_factor=3.5):
    """
    沿中心线的局部一致性检测: 用滑窗中位数 + MAD 自适应剔除异常截面.

    思想: 真实血管的横截面沿管轴是缓变的. 若某点的等效直径相对其
    局部邻居 (±half 个采样点) 的中位数偏差超过 mad_factor × 1.4826 × MAD,
    认为该点是污染 (邻近血管渗透 / 沿轴薄片), 标记为 0 (后续 NaN).

    完全自适应: 阈值由数据自身分布决定, 无任何硬编码尺寸. 在 MPV 这种
    粗血管处容忍大值, 在 LGV 等细血管处容忍小值.

    参数:
        window:     滑窗大小, 默认 15 (≈ 5 mm 在 1mm 间距下).
        mad_factor: MAD 倍数门槛 (近似 σ 倍数), 默认 3.5.

    返回:
        (area, perimeter, eq_diameter, n_removed) —— 原数组就地修改.
    """
    M = len(area)
    if M < window:
        return area, perimeter, eq_diameter, 0

    valid = eq_diameter > 0
    if int(np.sum(valid)) < window // 2:
        return area, perimeter, eq_diameter, 0

    half = window // 2
    n_removed = 0
    flagged = np.zeros(M, dtype=bool)

    for i in range(M):
        if not valid[i]:
            continue
        lo, hi = max(0, i - half), min(M, i + half + 1)
        # 排除自身, 取邻居有效值
        win = eq_diameter[lo:hi]
        win = win[win > 0]
        if len(win) < 5:
            continue
        med = float(np.median(win))
        mad = float(np.median(np.abs(win - med)))
        if mad < 1e-6:
            continue
        # 1.4826·MAD ≈ σ (正态)
        sigma_est = 1.4826 * mad
        deviation = abs(eq_diameter[i] - med) / sigma_est
        if deviation > mad_factor:
            flagged[i] = True

    if np.any(flagged):
        n_removed = int(np.sum(flagged))
        area[flagged] = 0.0
        perimeter[flagged] = 0.0
        eq_diameter[flagged] = 0.0

    return area, perimeter, eq_diameter, n_removed


def _curvature_sliding_window(coords, window=7):
    """滑窗法离散曲率 (1/mm)"""
    N = len(coords)
    curvatures = np.zeros(N)
    if N < 3:
        return curvatures
    half = window // 2
    for i in range(N):
        lo, hi = max(0, i - half), min(N - 1, i + half)
        a = coords[i] - coords[lo]
        b = coords[hi] - coords[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        lc = np.linalg.norm(coords[hi] - coords[lo])
        if la < 1e-10 or lb < 1e-10 or lc < 1e-10:
            continue
        area2 = np.linalg.norm(np.cross(a, b))
        curvatures[i] = 2.0 * area2 / (la * lb * lc)
    return curvatures


# ============================================================
# 沿分支提取逐点特征
# ============================================================

def _compute_inscribed_radius_per_point(coords, mesh):
    """
    对每个中心线点, 计算其到 STL 表面的最近距离 (≈ 局部内切球半径).
    使用 trimesh.proximity.signed_distance: 内部为正, 外部为负.

    返回 (M,) ndarray, 单位 mm. 失败时返回 0 数组.
    """
    try:
        import trimesh.proximity
        sd = trimesh.proximity.signed_distance(mesh, coords)
        sd = np.asarray(sd, dtype=float)
        # 中心线点应该在 mesh 内部 → sd > 0; 取正值, 异常点置 0
        sd = np.clip(sd, 0.0, None)
        return sd
    except Exception as e:
        print(f"    [warn] inscribed_radius 计算失败: {e}, 用 0 填充")
        return np.zeros(len(coords))


def _extract_branch_raw_profile(branch_path, nodes, mesh,
                                dt=None, origin=None, pitch=None,
                                curvature_window=7, section_step=1,
                                inscribed_factor=1.8):
    """
    沿一段中心线提取逐点剖面.

    inscribed_factor: 等效直径相对于内切直径 (2*r) 的最大允许倍数.
        越界则该位置截面记为 0 (后续会变 NaN).
        默认 1.8: 真实截面通常 1.0~1.4 倍 (圆形=1, 椭圆稍大).
        分叉点处穿透到邻近血管时, 比值会显著 > 2.

    返回: dict (含原始 area/eq_diameter/inscribed_radius/...) 或 None.
    """
    if len(branch_path) < 2:
        return None

    coords = path_to_coords(branch_path, nodes)
    M = len(coords)

    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(diffs)))

    tangents = _compute_tangents(coords)

    # ---- 内切半径 (来自 STL 表面距离, 用于边界效应过滤) ----
    inscribed_radius = _compute_inscribed_radius_per_point(coords, mesh)

    area = np.zeros(M)
    perimeter = np.zeros(M)

    indices = list(range(0, M, section_step))
    if indices[-1] != M - 1:
        indices.append(M - 1)

    n_success = 0
    n_rejected = 0
    for idx in indices:
        # 局部允许的截面等效直径上限: inscribed_factor × 内切直径
        r_loc = inscribed_radius[idx]
        max_eq_d = (2.0 * r_loc * inscribed_factor) if r_loc > 0.5 else None
        a, p = _compute_cross_section(mesh, coords[idx], tangents[idx],
                                       max_eq_diameter=max_eq_d)
        area[idx] = a
        perimeter[idx] = p
        if a > 0:
            n_success += 1
        elif r_loc > 0.5:
            # 计算成功了但被尺寸/形状过滤掉
            n_rejected += 1

    # ---- 局部一致性后处理 ----
    # 仅在采样点上做异常剔除 (因为只有这些点是真实计算结果, 其余是 0)
    sampled_idx = np.array(indices, dtype=int)
    sampled_area = area[sampled_idx].copy()
    sampled_peri = perimeter[sampled_idx].copy()
    sampled_eq = np.sqrt(4.0 * sampled_area / np.pi)
    sampled_eq[sampled_area <= 0] = 0.0

    sampled_area, sampled_peri, sampled_eq, n_outliers = \
        _remove_local_outliers(sampled_area, sampled_peri, sampled_eq,
                               window=15, mad_factor=3.5)
    # 写回
    area[sampled_idx] = sampled_area
    perimeter[sampled_idx] = sampled_peri

    # 对跳过的点插值 (仅对成功截面插值, 0 值不插)
    if section_step > 1 and n_success >= 2:
        sampled_arc = arc_length[indices]
        for arr in [area, perimeter]:
            sampled = arr[indices]
            valid = sampled > 0
            if np.sum(valid) >= 2:
                f = interp1d(sampled_arc[valid], sampled[valid],
                             kind='linear', bounds_error=False,
                             fill_value=(sampled[valid][0], sampled[valid][-1]))
                arr[:] = np.clip(f(arc_length), 0, None)

    eq_diameter = np.sqrt(4.0 * area / np.pi)
    eq_diameter[area <= 0] = 0.0

    circularity = np.zeros(M)
    valid_mask = (area > 0) & (perimeter > 0)
    circularity[valid_mask] = (4.0 * np.pi * area[valid_mask]) / (perimeter[valid_mask] ** 2)

    curvature = _curvature_sliding_window(coords, curvature_window)

    return {
        'arc_length': arc_length,
        'area': area,
        'perimeter': perimeter,
        'eq_diameter': eq_diameter,
        'circularity': circularity,
        'curvature': curvature,
        'inscribed_radius': inscribed_radius,
        '_n_sampled': len(indices),
        '_n_success': n_success,
        '_n_rejected_oversize': n_rejected,
        '_n_local_outliers': int(n_outliers),
    }


def _apply_endpoint_mask(profile, edge_margin_pct=0.05,
                         edge_margin_mm=8.0):
    """
    将段端点附近的截面值标记为 NaN。

    端点附近的截面常因以下原因失真:
      - STL 在血管末端的收口产生封闭面
      - 分叉点附近切平面穿透到相邻血管

    判定条件 (并集): 满足任一即标记 NaN
      1. 位置百分比 < edge_margin_pct 或 > (1 - edge_margin_pct)
      2. 距段起点弧长 < edge_margin_mm 或距段终点弧长 < edge_margin_mm

    参数:
        profile:           _resample_profile 返回的 dict (100 点剖面)
        edge_margin_pct:   端点保护比例 (默认 0.05 = 前后 5%)
        edge_margin_mm:    端点保护绝对距离 mm (默认 8.0)

    返回:
        修改后的 profile (原地修改)
    """
    if profile is None:
        return profile

    n = len(profile['position'])
    pos = np.array(profile['position'])  # 0..1
    arc = np.array(profile['arc_length_mm'])  # 0..total_length
    total = profile.get('total_length_mm', arc[-1] if len(arc) > 0 else 0)

    # 条件 1: 位置百分比
    pct_mask = (pos < edge_margin_pct) | (pos > 1 - edge_margin_pct)

    # 条件 2: 距段端点的实际距离
    dist_to_start = arc
    dist_to_end = total - arc
    mm_mask = (dist_to_start < edge_margin_mm) | (dist_to_end < edge_margin_mm)

    # 并集
    invalid_mask = pct_mask | mm_mask

    # 标记的 keys (截面相关特征)
    section_keys = ['area', 'perimeter', 'eq_diameter',
                    'circularity', 'inscribed_radius']

    n_masked = int(np.sum(invalid_mask))
    if n_masked > 0:
        for key in section_keys:
            if key in profile:
                values = list(profile[key])
                for i in range(n):
                    if invalid_mask[i]:
                        values[i] = float('nan')
                profile[key] = values

    # 元信息记录
    profile['edge_margin_pct'] = float(edge_margin_pct)
    profile['edge_margin_mm'] = float(edge_margin_mm)
    profile['n_masked_endpoints'] = n_masked

    return profile

def _resample_profile(raw_profile, n_points=100):
    """
    重采样到 n_points (沿弧长均匀)。

    修正: 对面积/周长等截面特征, 只用 area>0 的原始点插值,
          避免未采样点的 0 值污染最大值。
    """
    if raw_profile is None:
        return None

    arc = raw_profile['arc_length']
    total_length = arc[-1]
    if total_length < 1e-6:
        return None

    t_raw = arc / total_length
    t_uniform = np.linspace(0, 1, n_points)

    result = {
        'position': t_uniform.tolist(),
        'arc_length_mm': (t_uniform * total_length).tolist(),
        'total_length_mm': float(total_length),
        'n_raw_points': len(arc),
        'n_section_success': raw_profile.get('_n_success', 0),
    }

    # 哪些 key 需要"只用有效值插值"(截面计算的)
    section_keys = {'area', 'perimeter', 'eq_diameter', 'circularity'}
    # 哪些 key 直接用所有点(中心线本身的几何, 没有 0 值问题)
    geometry_keys = {'curvature', 'inscribed_radius'}

    # 用 area > 0 作为"截面成功"的掩码
    area_arr = np.asarray(raw_profile['area'])
    success_mask = area_arr > 0

    for key in (section_keys | geometry_keys):
        values = np.asarray(raw_profile[key])
        try:
            if key in section_keys:
                # 只用截面成功的原始点插值
                if np.sum(success_mask) >= 2:
                    t_valid = t_raw[success_mask]
                    v_valid = values[success_mask]
                    # 去重 (单调要求)
                    mask = np.concatenate(([True], np.diff(t_valid) > 1e-10))
                    t_c, v_c = t_valid[mask], v_valid[mask]
                    f = interp1d(t_c, v_c, kind='linear',
                                 bounds_error=False,
                                 fill_value=(v_c[0], v_c[-1]))
                    resampled = np.clip(f(t_uniform), 0, None)
                else:
                    resampled = np.zeros(n_points)
            else:
                # 中心线几何特征: 用所有原始点
                mask = np.concatenate(([True], np.diff(t_raw) > 1e-10))
                t_c, v_c = t_raw[mask], values[mask]
                if len(t_c) < 2:
                    resampled = np.zeros(n_points)
                else:
                    f = interp1d(t_c, v_c, kind='linear',
                                 bounds_error=False,
                                 fill_value='extrapolate')
                    resampled = np.clip(f(t_uniform), 0, None)

            if key == 'circularity':
                resampled = np.clip(resampled, 0, 1.5)
            result[key] = resampled.tolist()
        except Exception:
            result[key] = [0.0] * n_points

    return result

# ============================================================
# 主入口 (改为读 JSON 驱动)
# ============================================================
def extract_profiles(stl_path, n_points=100, pitch=0.5,
                     curvature_window=7, section_step=3,
                     edge_margin_pct=0.05,
                     edge_margin_mm=8.0,
                     inscribed_factor=1.8):
    """
    为每个解剖段提取 100 点剖面 (含截面特征)。

    输出:
        <patient_dir>/centerline_pointwise_profiles.json

    参数:
        stl_path:          vessel.stl 路径
        n_points:          重采样点数 (默认 100)
        pitch:             体素化分辨率 mm
        curvature_window:  曲率计算窗口
        section_step:      原始截面采样步长 (每隔 N 个中心线点算一次截面)
        edge_margin_pct:   端点保护比例 (默认 0.05)
        edge_margin_mm:    端点保护绝对距离 mm (默认 8.0)
        inscribed_factor:  截面等效直径相对于内切直径 (2*r) 的最大允许倍数
                           (默认 1.8). 用于过滤穿透到邻近血管的"超大"截面.
    """
    parentdir = os.path.dirname(stl_path)
    seg_path = os.path.join(parentdir, "centerline_profiles.json")
    if not os.path.exists(seg_path):
        print(f"  跳过 (无分段文件): {seg_path}")
        return

    with open(seg_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)

    nodes, _, _ = load_tree(stl_path)
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, 'geometry'):
            mesh = list(mesh.geometry.values())[0]
        else:
            print("  STL 加载失败")
            return

    profiles = {}
    n_total_masked = 0
    n_total_rejected_oversize = 0
    n_total_local_outliers = 0

    for seg_name, seg_info in seg_data['segments'].items():
        if seg_info is None:
            profiles[seg_name] = None
            continue

        try:
            # 计算原始剖面 (沿原始中心线点采样)
            raw_profile = _extract_branch_raw_profile(
                seg_info['path'], nodes, mesh,
                curvature_window=curvature_window,
                section_step=section_step,
                inscribed_factor=inscribed_factor)

            if raw_profile is None:
                profiles[seg_name] = None
                continue

            n_total_rejected_oversize += raw_profile.get(
                '_n_rejected_oversize', 0)
            n_total_local_outliers += raw_profile.get(
                '_n_local_outliers', 0)

            # 重采样到 n_points
            resampled = _resample_profile(raw_profile, n_points=n_points)
            if resampled is None:
                profiles[seg_name] = None
                continue

            # 应用端点掩码
            resampled = _apply_endpoint_mask(
                resampled,
                edge_margin_pct=edge_margin_pct,
                edge_margin_mm=edge_margin_mm)

            # 透传过滤元信息
            resampled['n_rejected_oversize'] = int(
                raw_profile.get('_n_rejected_oversize', 0))
            resampled['n_local_outliers'] = int(
                raw_profile.get('_n_local_outliers', 0))
            resampled['n_section_success'] = int(
                raw_profile.get('_n_success', 0))

            n_total_masked += resampled.get('n_masked_endpoints', 0)
            profiles[seg_name] = resampled

        except Exception as e:
            print(f"    [{seg_name}] 剖面提取失败: {e}")
            profiles[seg_name] = None

    # 元数据
    profiles['_meta'] = {
        'patient_id': seg_data.get('patient_id'),
        'is_post_tips': seg_data.get('is_post_tips'),
        'n_points': n_points,
        'edge_margin_pct': float(edge_margin_pct),
        'edge_margin_mm': float(edge_margin_mm),
        'inscribed_factor': float(inscribed_factor),
        'n_total_masked': int(n_total_masked),
        'n_total_rejected_oversize': int(n_total_rejected_oversize),
        'n_total_local_outliers': int(n_total_local_outliers),
    }

    out_path = os.path.join(parentdir, "centerline_pointwise_profiles.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False, allow_nan=True)

    valid_segs = [k for k, v in profiles.items()
                   if v is not None and not k.startswith('_')]
    print(f"  剖面提取完成: {len(valid_segs)} 个段, "
          f"端点掩码 {n_total_masked} 处, "
          f"形状/内切超限剔除 {n_total_rejected_oversize} 处, "
          f"局部异常剔除 {n_total_local_outliers} 处")
    return profiles

def _diagnose_centerline_mesh(branch_path, nodes, mesh):
    """诊断 MPV 中心线点和 mesh 的对齐情况"""
    if len(branch_path) < 3:
        return

    coords = path_to_coords(branch_path, nodes)
    test_indices = [0, len(branch_path)//4, len(branch_path)//2,
                    3*len(branch_path)//4, len(branch_path)-1]

    mb = mesh.bounds
    print(f"  [诊断] mesh: x=[{mb[0][0]:.1f},{mb[1][0]:.1f}], "
          f"y=[{mb[0][1]:.1f},{mb[1][1]:.1f}], "
          f"z=[{mb[0][2]:.1f},{mb[1][2]:.1f}]")

    n_inside = 0
    for idx in test_indices:
        if mesh.contains([coords[idx]])[0]:
            n_inside += 1
    print(f"  [诊断] MPV 测试点 {n_inside}/{len(test_indices)} 在 mesh 内部")

    mid = len(branch_path) // 2
    pt = coords[mid]
    tangent = (coords[min(mid+1, len(coords)-1)]
               - coords[max(mid-1, 0)])
    tangent /= (np.linalg.norm(tangent) + 1e-15)

    try:
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal=tangent, plane_origin=pt)
        n_segs = len(lines) if lines is not None else 0
        a, p = _section_one(mesh, pt, tangent)
        print(f"  [诊断] MPV 中点截面: {n_segs}线段, "
              f"面积={a:.2f}mm², 周长={p:.2f}mm")
    except Exception as e:
        print(f"  [诊断] 截面测试异常: {e}")


# ============================================================
# 批量
# ============================================================

def batch_extract_profiles(root_folder, n_points=100, pitch=0.5,
                           section_step=3, stl_name="vessel.stl"):
    print(f"\n{'='*60}")
    print(f"批量剖面提取: {root_folder}")
    print(f"{'='*60}")

    subfolders = sorted(
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)))

    success, fail = 0, 0
    for folder in subfolders:
        fp = os.path.join(root_folder, folder)
        stl = os.path.join(fp, stl_name)
        if not os.path.exists(stl):
            continue
        seg_json = os.path.join(fp, "centerline_profiles.json")
        if not os.path.exists(seg_json):
            print(f"  {folder}: 缺少分段 JSON, 跳过")
            fail += 1
            continue
        try:
            extract_profiles(stl, n_points, pitch, section_step=section_step)
            success += 1
        except Exception as e:
            print(f"  {folder}: 失败 ({e})")
            fail += 1

    print(f"\n完成: {success} 成功, {fail} 失败")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        extract_profiles(sys.argv[1])
    else:
        batch_extract_profiles(r"F:\PCG data\dataset\zhengzhou_vkan_qian47")