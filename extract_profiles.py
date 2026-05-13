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


def _pick_polygon_from_geometry(geom, center):
    """从 Polygon/MultiPolygon 中选中心线锚点所属的主多边形。"""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom if geom.is_valid and geom.area > 0 else None
    if not hasattr(geom, 'geoms'):
        return None

    polys = [g for g in geom.geoms
             if getattr(g, 'geom_type', None) == 'Polygon'
             and g.is_valid and g.area > 0]
    if not polys:
        return None

    containing = [p for p in polys if p.covers(center)]
    if containing:
        return max(containing, key=lambda p: p.area)
    return min(polys, key=lambda p: p.distance(center))


def _center_owned_polygon(poly, center, ownership_factor=1.8):
    """
    用中心线锚定的最大内切半径裁剪截面。

    r_anchor 是中心点到截面边界的最短距离。限制圆半径取
    ownership_factor * r_anchor: 圆形血管不受影响, 椭圆血管保留主体,
    分叉污染向外伸出的区域会被裁掉。
    """
    if poly is None or poly.is_empty or not poly.is_valid:
        return None, 0.0, 0.0

    if ownership_factor is None or ownership_factor <= 0:
        return poly, 0.0, 0.0

    if not poly.covers(center):
        return poly, 0.0, 0.0

    anchor_radius = float(poly.boundary.distance(center))
    if anchor_radius <= 1e-6:
        return poly, anchor_radius, 0.0

    owned_radius = float(anchor_radius * ownership_factor)
    limiter = center.buffer(owned_radius, resolution=64)
    owned_geom = poly.intersection(limiter)
    owned_poly = _pick_polygon_from_geometry(owned_geom, center)
    if owned_poly is None or owned_poly.area <= 1e-9:
        return poly, anchor_radius, owned_radius
    return owned_poly, anchor_radius, owned_radius


def _section_one(mesh, point, normal, max_eq_diameter=None,
                 ownership_factor=1.8,
                 return_ring=False, return_metrics=False,
                 return_raw=False, return_extras=False):
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

    额外形状感知量 (return_extras=True, 用于 PVT / 血栓识别):
      - n_components: 切平面下"有效闭合多边形"个数. 正常血管 = 1;
                      血栓把管腔从中间隔断 → 2+; 圆环形血栓 = 1 (仍连通).
      - solidity:     所选多边形面积 / 其凸包面积 ∈ (0, 1].
                      凸截面 (圆/椭圆) = 1; 月牙/凹缺口形 < 1; 越小代表
                      凹缺口越深 (典型 PVT 边缘血栓).

    中心线锚定清洗:
      先取真实截面多边形 P, 再用当前中心线投影点 (0,0) 到 P 边界的
      最短距离作为锚定内切半径 r_anchor。最终用于面积统计的是
      P ∩ circle((0,0), ownership_factor·r_anchor)。这能保留当前血管
      主体, 同时裁掉分叉/汇合处伸向邻近血管的污染区域。

    防止边界效应 (邻近血管"渗透"):
      若 max_eq_diameter (一般取 1.6 ~ 2 倍局部内切直径) 给定,
      且清洗后候选多边形的等效直径 > max_eq_diameter, 视为污染并丢弃.

    参数:
        max_eq_diameter: float 或 None — 等效直径上界 (mm)
        ownership_factor: 中心锚定裁剪圆半径 / 锚定内切半径, 默认 1.8
        return_ring:     是否同时返回 2D 多边形轮廓 (用于可视化)
        return_metrics:  是否同时返回 (aspect_ratio, circularity)
        return_raw:      是否追加原始未裁剪的 area/perimeter 与锚定半径
        return_extras:   是否同时返回 (n_components, solidity)

    返回 (按 flag 组合, extras 永远放在末尾, 不影响既有调用方):
        默认                                            (area, peri)
        return_metrics=True                             (area, peri, AR, circ)
        return_ring=True                                (area, peri, ring_2d)
        return_ring=True, return_metrics=True           (area, peri, AR, circ, ring_2d)
        return_raw=True 时, 再追加 (raw_area, raw_peri, anchor_r, owned_r)
        return_extras=True 时, 最后追加 (n_components, solidity)
        失败时各位置填 0/0/999/0/None/0/0
    """
    base_fail = (0.0, 0.0)
    extras_fail = (0, 0.0)
    if return_ring and return_metrics:
        fail = (0.0, 0.0, 999.0, 0.0, None)
    elif return_metrics:
        fail = (0.0, 0.0, 999.0, 0.0)
    elif return_ring:
        fail = (0.0, 0.0, None)
    else:
        fail = base_fail
    if return_raw:
        fail = fail + (0.0, 0.0, 0.0, 0.0)
    if return_extras:
        fail = fail + extras_fail

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
        # 有效"非微小"多边形 — 用于估计 lumen 连通分量数
        # 阈值 0.1mm² 排除离散化产生的针状碎片
        nontrivial = [p for p in polys
                      if p.is_valid and p.area > 0.1]
        containing = [p for p in nontrivial if p.contains(center)]

        if containing:
            best = min(containing, key=lambda p: p.area)
        else:
            valid_polys = [p for p in polys if p.is_valid and p.area > 0]
            if not valid_polys:
                return fail
            best = min(valid_polys, key=lambda p: p.distance(center))

        raw_area = float(best.area)
        raw_peri = float(best.exterior.length)

        owned, anchor_radius, owned_radius = _center_owned_polygon(
            best, center, ownership_factor=ownership_factor)
        if owned is None:
            return fail

        area = float(owned.area)
        peri = float(owned.exterior.length)

        # 边界效应保护: 若给了内切直径上界, 直接拒绝越界候选
        if max_eq_diameter is not None and area > 0:
            eq_d = float(np.sqrt(4.0 * area / np.pi))
            if eq_d > max_eq_diameter:
                return fail

        # 形状指标
        raw_ring_2d_list = list(best.exterior.coords)
        owned_ring_2d_list = list(owned.exterior.coords)
        if return_metrics:
            aspect_ratio = _polygon_aspect_ratio(owned_ring_2d_list)
            if peri > 1e-6:
                circularity = float(min(1.5, 4.0 * np.pi * area / (peri * peri)))
            else:
                circularity = 0.0

        if return_extras:
            # n_components: 切平面下"有效非微小"多边形数 (lumen 是否被血栓隔断)
            # 不考虑跨血管 — 跨血管候选会在后续 shape filter 中被剔除
            n_components = int(len(nontrivial)) if nontrivial else 1

            # solidity: 所选多边形面积 / 其凸包面积
            # 凸 (圆/椭圆) = 1.0; 月牙/凹缺口 < 1.0
            solidity = 1.0
            try:
                from scipy.spatial import ConvexHull
                ring_arr = np.asarray(owned_ring_2d_list, dtype=float)
                if len(ring_arr) >= 3 and ring_arr.shape[1] >= 2:
                    hull = ConvexHull(ring_arr[:, :2])
                    hull_area = float(hull.volume)  # 2D 下 .volume 即面积
                    if hull_area > 1e-9:
                        solidity = float(min(1.0, area / hull_area))
            except Exception:
                solidity = 1.0
            extras_tuple = (n_components, solidity)

        if return_ring and return_metrics:
            base = (area, peri, aspect_ratio, circularity, raw_ring_2d_list)
        elif return_metrics:
            base = (area, peri, aspect_ratio, circularity)
        elif return_ring:
            base = (area, peri, raw_ring_2d_list)
        else:
            base = (area, peri)
        if return_raw:
            base = base + (raw_area, raw_peri, anchor_radius, owned_radius)
        if return_extras:
            return base + extras_tuple
        return base

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
                           ownership_factor=1.8,
                           max_aspect_ratio=4.0,
                           min_circularity=0.30,
                           return_normal=False,
                           return_raw=False,
                           return_extras=False):
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
        ownership_factor:    中心线锚定裁剪圆半径 / 锚定内切半径.
        max_aspect_ratio:    硬剔除阈值, 默认 4.0.
        min_circularity:     硬剔除阈值, 默认 0.30.
        return_normal:       是否返回所选最佳法线 (供可视化复现).
        return_raw:          是否追加原始未裁剪 area/perimeter 与锚定半径.
        return_extras:       是否额外返回 (n_components, solidity) — PVT/血栓
                             形状指标. 见 _section_one 文档.

    返回:
        默认                   : (area, perimeter)
        return_normal=True     : (area, perimeter, best_normal)
        return_raw=True        : 追加 (raw_area, raw_perimeter, anchor_radius, owned_radius)
        return_extras=True     : 末尾追加 (n_components, solidity)
    """
    normal = normal / (np.linalg.norm(normal) + 1e-15)
    candidates = _generate_normal_candidates(normal, n_perturb, max_angle_deg)

    best_score = float('inf')
    best_area, best_peri, best_normal = 0.0, 0.0, normal
    best_raw_area, best_raw_peri = 0.0, 0.0
    best_anchor_radius, best_owned_radius = 0.0, 0.0
    best_ncomp, best_solidity = 0, 0.0
    for n in candidates:
        if return_extras:
            if return_raw:
                a, p, ar, circ, raw_a, raw_p, anchor_r, owned_r, ncomp, sol = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True,
                    return_raw=True,
                    return_extras=True)
            else:
                a, p, ar, circ, ncomp, sol = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True,
                    return_extras=True)
                raw_a, raw_p, anchor_r, owned_r = a, p, 0.0, 0.0
        else:
            if return_raw:
                a, p, ar, circ, raw_a, raw_p, anchor_r, owned_r = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True,
                    return_raw=True)
            else:
                a, p, ar, circ = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True)
                raw_a, raw_p, anchor_r, owned_r = a, p, 0.0, 0.0
            ncomp, sol = 0, 0.0
        if a <= 0:
            continue
        # 形状硬过滤
        if ar > max_aspect_ratio or circ < min_circularity:
            continue
        score = _shape_score(a, ar, circ)
        if score < best_score:
            best_score = score
            best_area, best_peri, best_normal = a, p, n
            best_raw_area, best_raw_peri = raw_a, raw_p
            best_anchor_radius, best_owned_radius = anchor_r, owned_r
            best_ncomp, best_solidity = ncomp, sol

    if best_score == float('inf'):
        # 所有候选均不合格 → 该点截面无效
        base = (0.0, 0.0)
        if return_normal:
            base = base + (normal,)
        if return_raw:
            base = base + (0.0, 0.0, 0.0, 0.0)
        if return_extras:
            base = base + (0, 0.0)
        return base

    base = (best_area, best_peri)
    if return_normal:
        base = base + (best_normal,)
    if return_raw:
        base = base + (best_raw_area, best_raw_peri,
                       best_anchor_radius, best_owned_radius)
    if return_extras:
        base = base + (int(best_ncomp), float(best_solidity))
    return base


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


def _remove_rate_outliers(area, perimeter, eq_diameter, arc_length,
                          max_rate_per_mm=0.5):
    """
    沿管轴"变化速率"过滤: 真实血管的直径沿管轴是缓变的, 哪怕在缩窄/狭窄处,
    每 mm 的相对直径变化也很少超过 50% (max_rate_per_mm=0.5).
    单点出现急剧塌陷/急剧膨胀 → 截面渗透到邻近血管 / 沿轴薄片切 / 分叉伪影.

    判定: 对每个有效采样点 i, 找其最近的左右有效邻居 j ∈ {prev, next},
    计算相对变化率
        r_j = |D[i] − D[j]| / (mean_D · Δs_ij)        单位 1/mm
    - 若两侧邻居均给出 r_j > max_rate_per_mm → 孤立尖峰, 剔除
    - 若只有一侧邻居 (段端), 该侧 r_j > 2·max_rate_per_mm 才剔除
      (段端单边判据更严, 避免误伤端点处的真实收口)

    与 `_remove_local_outliers` (MAD) 互补:
      MAD : 适合捕捉与"局部分布"显著偏离的点 (含成簇异常)
      rate: 适合捕捉单点"突变 / 阶梯", 含图像 2.png / 3.png 中 MPV 沿轴
            单点截面塌陷 (大血管中突现 1.8mm² 极小值) 这类伪影.

    参数:
        max_rate_per_mm: 允许的相对直径变化率上限 (1/mm), 默认 0.5

    返回:
        (area, perimeter, eq_diameter, n_removed) — 原地修改
    """
    M = len(area)
    if M < 3 or max_rate_per_mm <= 0:
        return area, perimeter, eq_diameter, 0

    valid = eq_diameter > 0
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 3:
        return area, perimeter, eq_diameter, 0

    flagged = np.zeros(M, dtype=bool)

    for k, i in enumerate(valid_idx):
        rates = []
        neighbors = []
        if k > 0:
            neighbors.append(valid_idx[k - 1])
        if k < len(valid_idx) - 1:
            neighbors.append(valid_idx[k + 1])
        for j in neighbors:
            ds = abs(arc_length[i] - arc_length[j])
            if ds < 1e-6:
                continue
            mean_d = 0.5 * (eq_diameter[i] + eq_diameter[j])
            if mean_d < 1e-6:
                continue
            rates.append(abs(eq_diameter[i] - eq_diameter[j]) / (mean_d * ds))

        if not rates:
            continue
        if len(rates) >= 2:
            # 两侧都有邻居: 两侧均超阈才剔除 (剔除孤立尖峰)
            if all(r > max_rate_per_mm for r in rates):
                flagged[i] = True
        else:
            # 段端单边: 阈值加倍, 更保守
            if rates[0] > 2.0 * max_rate_per_mm:
                flagged[i] = True

    n_removed = int(np.sum(flagged))
    if n_removed > 0:
        area[flagged] = 0.0
        perimeter[flagged] = 0.0
        eq_diameter[flagged] = 0.0
    return area, perimeter, eq_diameter, n_removed


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


def _torsion_sliding_window(coords, arc_length, smooth_sigma=2.0,
                             min_curvature_for_torsion=1e-3):
    """
    Frenet-Serret 挠率 τ (1/mm), 描述中心线在 3D 空间的"扭转"程度.

    曲率 κ 描述"弯不弯"; 挠率 τ 描述"扭不扭". 平面曲线 τ=0; 螺旋线 τ>0.
    门静脉海绵样变 / 重度迂曲的代偿血管 → τ 显著升高.

    公式 (对弧长 s 的导数):
        τ = ((P' × P'') · P''') / |P' × P''|²

    数值实现:
      1. 用 Gaussian 平滑坐标 (σ=smooth_sigma 个点), 抑制离散噪声
      2. np.gradient 对弧长求一/二/三阶导数
      3. 直线段 (|P' × P''| 几乎 0, ≡ κ ≈ 0) 数值不稳定 → 置 NaN

    参数:
        coords:                    (N, 3) 中心线坐标
        arc_length:                (N,) 累积弧长 (单调递增)
        smooth_sigma:              坐标平滑核宽 (点数), 默认 2
        min_curvature_for_torsion: 曲率低于此值的点上挠率置 NaN
                                    (避免直线段的 0/0 数值噪声)

    返回:
        (N,) 挠率数组, 不可信处为 NaN.
    """
    N = len(coords)
    if N < 5:
        return np.full(N, np.nan)
    try:
        from scipy.ndimage import gaussian_filter1d
        coords_s = gaussian_filter1d(coords, sigma=smooth_sigma,
                                      axis=0, mode='nearest')
    except Exception:
        coords_s = np.asarray(coords, dtype=float)

    s = np.asarray(arc_length, dtype=float)
    # 弧长退化 (重复点) 时 np.gradient 会发散
    if not np.all(np.diff(s) > 1e-8):
        return np.full(N, np.nan)

    p1 = np.gradient(coords_s, s, axis=0)
    p2 = np.gradient(p1, s, axis=0)
    p3 = np.gradient(p2, s, axis=0)

    cross_12 = np.cross(p1, p2)                   # (N, 3)
    denom = np.sum(cross_12 ** 2, axis=1)         # |P'×P''|²

    numer = np.einsum('ij,ij->i', cross_12, p3)   # (P'×P'') · P'''
    torsion = numer / (denom + 1e-12)

    # 在曲率近 0 处, 数值不稳定 → NaN
    curv = np.sqrt(np.maximum(denom, 0.0)) / (
        np.linalg.norm(p1, axis=1) ** 3 + 1e-12)
    bad = (curv < min_curvature_for_torsion) | ~np.isfinite(torsion)
    torsion[bad] = np.nan
    return torsion


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
                                inscribed_factor=1.8,
                                ownership_factor=1.8,
                                max_diameter_rate_per_mm=0.5):
    """
    沿一段中心线提取逐点剖面.

    inscribed_factor: 等效直径相对于内切直径 (2*r) 的最大允许倍数.
        越界则该位置截面记为 0 (后续会变 NaN).
        默认 1.8: 真实截面通常 1.0~1.4 倍 (圆形=1, 椭圆稍大).
        分叉点处穿透到邻近血管时, 比值会显著 > 2.

    ownership_factor: 中心线锚定清洗半径倍数.
        clean_area = raw_section ∩ circle(center, ownership_factor*r_anchor).
        默认 1.8, 保留椭圆主体并裁掉分叉污染外伸区域.

    max_diameter_rate_per_mm: 沿管轴允许的等效直径相对变化率 (1/mm).
        超过此速率的孤立点视为伪影 (单点塌陷/膨胀), 见 _remove_rate_outliers.
        默认 0.5 = 每 mm 最多 50% 相对变化.

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

    area = np.zeros(M)           # clean/owned area, downstream default
    perimeter = np.zeros(M)      # clean/owned perimeter
    raw_area = np.zeros(M)       # original STL section before owned clipping
    raw_perimeter = np.zeros(M)
    anchor_radius = np.zeros(M)
    owned_radius = np.zeros(M)
    solidity = np.zeros(M)            # (新) area / convex_hull_area
    n_components = np.zeros(M, dtype=np.int16)  # (新) lumen 连通分量数

    indices = list(range(0, M, section_step))
    if indices[-1] != M - 1:
        indices.append(M - 1)

    n_success = 0
    n_rejected = 0
    for idx in indices:
        # 局部允许的截面等效直径上限: inscribed_factor × 内切直径
        r_loc = inscribed_radius[idx]
        max_eq_d = (2.0 * r_loc * inscribed_factor) if r_loc > 0.5 else None
        a, p, raw_a, raw_p, anchor_r, owned_r, ncomp, sol = _compute_cross_section(
            mesh, coords[idx], tangents[idx],
            max_eq_diameter=max_eq_d,
            ownership_factor=ownership_factor,
            return_raw=True,
            return_extras=True)
        area[idx] = a
        perimeter[idx] = p
        raw_area[idx] = raw_a
        raw_perimeter[idx] = raw_p
        anchor_radius[idx] = anchor_r
        owned_radius[idx] = owned_r
        solidity[idx] = sol
        n_components[idx] = ncomp
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
    sampled_arc = arc_length[sampled_idx]

    # (1) MAD 局部异常 (与局部分布偏离)
    sampled_area, sampled_peri, sampled_eq, n_outliers = \
        _remove_local_outliers(sampled_area, sampled_peri, sampled_eq,
                               window=15, mad_factor=3.5)
    # (2) 沿管轴变化速率过滤 (单点突变/塌陷, 与 MAD 互补)
    sampled_area, sampled_peri, sampled_eq, n_rate_outliers = \
        _remove_rate_outliers(sampled_area, sampled_peri, sampled_eq,
                              sampled_arc,
                              max_rate_per_mm=max_diameter_rate_per_mm)
    # 写回
    area[sampled_idx] = sampled_area
    perimeter[sampled_idx] = sampled_peri
    # 同步: 被剔除的位置 (area=0) 把 solidity / n_components 也清掉, 防止
    # 后续 resample 把无效残留传到 100 点输出.
    zeroed = sampled_area <= 0
    if np.any(zeroed):
        raw_area[sampled_idx[zeroed]] = 0.0
        raw_perimeter[sampled_idx[zeroed]] = 0.0
        anchor_radius[sampled_idx[zeroed]] = 0.0
        owned_radius[sampled_idx[zeroed]] = 0.0
        solidity[sampled_idx[zeroed]] = 0.0
        n_components[sampled_idx[zeroed]] = 0

    # 对跳过的点插值 (仅对成功截面插值, 0 值不插)
    if section_step > 1 and n_success >= 2:
        sampled_arc = arc_length[indices]
        for arr in [area, perimeter, raw_area, raw_perimeter,
                    anchor_radius, owned_radius, solidity]:
            sampled = arr[indices]
            valid = sampled > 0
            if np.sum(valid) >= 2:
                f = interp1d(sampled_arc[valid], sampled[valid],
                             kind='linear', bounds_error=False,
                             fill_value=(sampled[valid][0], sampled[valid][-1]))
                arr[:] = np.clip(f(arc_length), 0, None)

    eq_diameter = np.sqrt(4.0 * area / np.pi)
    eq_diameter[area <= 0] = 0.0
    raw_eq_diameter = np.sqrt(4.0 * raw_area / np.pi)
    raw_eq_diameter[raw_area <= 0] = 0.0

    circularity = np.zeros(M)
    valid_mask = (area > 0) & (perimeter > 0)
    circularity[valid_mask] = (4.0 * np.pi * area[valid_mask]) / (perimeter[valid_mask] ** 2)

    # ---- 形状/水力派生通道 ----
    # 水力直径 D_h = 4 A / P (适用于任意非圆截面)
    hydraulic_diameter = np.zeros(M)
    hydraulic_diameter[valid_mask] = (4.0 * area[valid_mask]
                                      / perimeter[valid_mask])
    # 瓶颈比 = 2·r_inscribed / D_eq ∈ (0, 1]
    # 圆形 ≈ 1; 月牙/血栓挤压 → << 1 (真实通道宽 比 乐观估计窄)
    r_insc_to_r_eq_ratio = np.zeros(M)
    eq_valid = eq_diameter > 1e-6
    r_insc_to_r_eq_ratio[eq_valid] = np.clip(
        (2.0 * inscribed_radius[eq_valid]) / eq_diameter[eq_valid],
        0.0, 1.5)
    # solidity 已经在采样点直接得到, 用 0 标记缺失. circularity 同步,
    # 在 _resample_profile 里会按 area>0 做插值.

    # 曲率 + 挠率 (中心线本身的几何, 不受截面有效性影响)
    curvature = _curvature_sliding_window(coords, curvature_window)
    torsion = _torsion_sliding_window(coords, arc_length)

    return {
        'arc_length': arc_length,
        'area': area,
        'perimeter': perimeter,
        'eq_diameter': eq_diameter,
        'raw_area': raw_area,
        'raw_perimeter': raw_perimeter,
        'raw_eq_diameter': raw_eq_diameter,
        'anchor_radius': anchor_radius,
        'owned_radius': owned_radius,
        'hydraulic_diameter': hydraulic_diameter,
        'circularity': circularity,
        'solidity': solidity,
        'n_components': n_components.astype(float),  # 便于和其它通道共用插值
        'r_insc_to_r_eq_ratio': r_insc_to_r_eq_ratio,
        'curvature': curvature,
        'torsion': torsion,
        'inscribed_radius': inscribed_radius,
        '_n_sampled': len(indices),
        '_n_success': n_success,
        '_n_rejected_oversize': n_rejected,
        '_n_local_outliers': int(n_outliers),
        '_n_rate_outliers': int(n_rate_outliers),
    }


def _copy_section_values(profile, src_idx, dst_mask, keys):
    """把一个可信截面的主通道复制到一组目标点。"""
    n = len(profile['position'])
    dst_idx = np.where(dst_mask)[0]
    if len(dst_idx) == 0:
        return
    for key in keys:
        if key not in profile:
            continue
        values = list(profile[key])
        if src_idx < 0 or src_idx >= len(values):
            continue
        src_val = values[src_idx]
        for i in dst_idx:
            if 0 <= i < n:
                values[i] = src_val
        profile[key] = values


def _refresh_dA_ds_norm(profile):
    """根据当前 area 重新计算归一化面积变化率。"""
    try:
        area = np.asarray(profile.get('area', []), dtype=float)
        arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
        if len(area) != len(arc) or len(area) < 3 or not np.all(np.diff(arc) > 0):
            return
        if np.sum(np.isfinite(area) & (area > 0)) < 3:
            profile['dA_ds_norm'] = [float('nan')] * len(area)
            return
        grad = np.gradient(area, arc)
        with np.errstate(divide='ignore', invalid='ignore'):
            dA_ds = grad / np.where(area > 1e-6, area, np.nan)
        dA_ds[(area <= 0) | ~np.isfinite(area)] = np.nan
        profile['dA_ds_norm'] = dA_ds.tolist()
    except Exception:
        return


def _apply_endpoint_mask(profile, edge_margin_pct=0.05,
                         edge_margin_mm=8.0,
                         branchpoint_arcs=None,
                         terminal_start=True,
                         terminal_end=True,
                         junction_policy='min_valid'):
    """
    处理段端点/交叉点附近的截面值。

    真实血管末端附近仍标记为 NaN, 避免 STL 开口/收口伪影。
    分叉/交叉点附近不再丢弃: 默认用该段可信区域的最小 clean area
    对应截面替换这些点, 让它们参与平均面积等统计, 但不再可能成为
    错误的最大截面。

    判定:
      - 真实末端: 距起/终点 < edge_margin_mm 或落在端点百分比保护带
      - 交叉点: 距 branchpoint_arcs 任一弧长 < junction_margin

    junction_policy:
      - 'min_valid': 用非末端、非交叉保护区中 clean area 最小的截面替换
      - 'cap_min':  只把交叉区中大于最小可信面积的点封顶到最小可信截面
      - 'keep':     交叉区不处理

    参数:
        profile:           _resample_profile 返回的 dict (100 点剖面)
        edge_margin_pct:   端点保护比例 (默认 0.05 = 前后 5%)
        edge_margin_mm:    端点保护绝对距离 mm (默认 8.0)
        branchpoint_arcs:   当前段路径上所有分叉点的弧长位置(mm)
        terminal_start/end: 当前段首/尾是否真实血管末端; 分叉点端不是末端

    返回:
        修改后的 profile (原地修改)
    """
    if profile is None:
        return profile

    n = len(profile['position'])
    pos = np.array(profile['position'])  # 0..1
    arc = np.array(profile['arc_length_mm'])  # 0..total_length
    total = profile.get('total_length_mm', arc[-1] if len(arc) > 0 else 0)

    # 真实末端保护: 只对非分叉的开口/末端置 NaN.
    start_pct_mask = pos < edge_margin_pct
    end_pct_mask = pos > 1 - edge_margin_pct

    dist_to_start = arc
    dist_to_end = total - arc
    start_mm_mask = dist_to_start < edge_margin_mm
    end_mm_mask = dist_to_end < edge_margin_mm

    terminal_mask = np.zeros(n, dtype=bool)
    if terminal_start:
        terminal_mask |= start_pct_mask | start_mm_mask
    if terminal_end:
        terminal_mask |= end_pct_mask | end_mm_mask

    # 交叉点保护: 不丢弃, 用本段可信最小截面替换/封顶.
    junction_mask = np.zeros(n, dtype=bool)
    branchpoint_arcs = branchpoint_arcs or []
    junction_margin = max(float(edge_margin_mm), float(total) * edge_margin_pct)
    for bp_arc in branchpoint_arcs:
        try:
            bp_arc = float(bp_arc)
        except Exception:
            continue
        junction_mask |= np.abs(arc - bp_arc) < junction_margin
    junction_mask &= ~terminal_mask

    # 标记的 keys (截面相关特征 + 新增形状/水力派生)
    section_keys = ['area', 'perimeter', 'eq_diameter',
                    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
                    'anchor_radius', 'owned_radius',
                    'circularity', 'inscribed_radius',
                    'hydraulic_diameter', 'solidity',
                    'r_insc_to_r_eq_ratio', 'n_components',
                    'dA_ds_norm']

    n_masked = int(np.sum(terminal_mask))
    if n_masked > 0:
        for key in section_keys:
            if key in profile:
                values = list(profile[key])
                for i in range(n):
                    if terminal_mask[i]:
                        values[i] = float('nan')
                profile[key] = values

    n_junction = int(np.sum(junction_mask))
    n_junction_replaced = 0
    area = np.asarray(profile.get('area', []), dtype=float)
    trusted_mask = (
        np.isfinite(area) & (area > 0)
        & ~terminal_mask & ~junction_mask
    )
    if n_junction > 0 and junction_policy in ('min_valid', 'cap_min'):
        reference_mask = trusted_mask
        if not np.any(reference_mask):
            # 短段可能几乎全在交叉保护区内; 这时退回到整段有效最小值,
            # 仍然避免交叉区异常大截面成为最大截面.
            reference_mask = np.isfinite(area) & (area > 0) & ~terminal_mask
        if np.any(reference_mask):
            trusted_idx = np.where(reference_mask)[0]
            min_idx = int(trusted_idx[np.argmin(area[trusted_idx])])
            main_keys = ['area', 'perimeter', 'eq_diameter',
                         'hydraulic_diameter', 'circularity', 'solidity',
                         'r_insc_to_r_eq_ratio', 'n_components']
            if junction_policy == 'min_valid':
                replace_mask = junction_mask
            else:
                replace_mask = junction_mask & np.isfinite(area) & (
                    area > area[min_idx])
            _copy_section_values(profile, min_idx, replace_mask, main_keys)
            n_junction_replaced = int(np.sum(replace_mask))

    # 标记哪些点来自交叉区替换, 便于可视化和诊断.
    marker = [0.0] * n
    for i in np.where(junction_mask)[0]:
        marker[int(i)] = 1.0
    profile['junction_replaced'] = marker
    _refresh_dA_ds_norm(profile)

    # 元信息记录
    profile['edge_margin_pct'] = float(edge_margin_pct)
    profile['edge_margin_mm'] = float(edge_margin_mm)
    profile['n_masked_endpoints'] = n_masked
    profile['n_junction_protected'] = n_junction
    profile['n_junction_replaced'] = n_junction_replaced
    profile['junction_policy'] = junction_policy

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

    # 哪些 key 需要"只用有效值插值"(截面计算的, 0 值代表缺失)
    section_keys = {'area', 'perimeter', 'eq_diameter', 'circularity',
                    'hydraulic_diameter', 'solidity',
                    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
                    'anchor_radius', 'owned_radius'}
    # 哪些 key 直接用所有点(中心线本身的几何, 没有 0 值问题)
    geometry_keys = {'curvature', 'inscribed_radius', 'r_insc_to_r_eq_ratio'}
    # 整数离散 (lumen 分量数), 用最近邻
    integer_keys = {'n_components'}
    # 含 NaN 的几何 (挠率), 单独处理 — NaN 不参与插值
    nanable_keys = {'torsion'}

    # 用 area > 0 作为"截面成功"的掩码
    area_arr = np.asarray(raw_profile['area'])
    success_mask = area_arr > 0

    available_keys = section_keys | geometry_keys | integer_keys | nanable_keys
    available_keys = {k for k in available_keys if k in raw_profile}

    for key in available_keys:
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
            elif key in integer_keys:
                # 离散整数: 用最近邻插值 (取整) + 端点延拓
                if np.sum(success_mask) >= 2:
                    t_valid = t_raw[success_mask]
                    v_valid = values[success_mask]
                    mask = np.concatenate(([True], np.diff(t_valid) > 1e-10))
                    t_c, v_c = t_valid[mask], v_valid[mask]
                    f = interp1d(t_c, v_c, kind='nearest',
                                 bounds_error=False,
                                 fill_value=(v_c[0], v_c[-1]))
                    resampled = np.clip(f(t_uniform), 0, None)
                else:
                    resampled = np.zeros(n_points)
            elif key in nanable_keys:
                # NaN-aware: 跳过 NaN 做线性插值, 不可信处仍保留 NaN
                finite = np.isfinite(values)
                if np.sum(finite) >= 2:
                    t_c, v_c = t_raw[finite], values[finite]
                    mask = np.concatenate(([True], np.diff(t_c) > 1e-10))
                    t_c, v_c = t_c[mask], v_c[mask]
                    f = interp1d(t_c, v_c, kind='linear',
                                 bounds_error=False,
                                 fill_value=np.nan)
                    resampled = f(t_uniform)
                else:
                    resampled = np.full(n_points, np.nan)
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
            if key == 'solidity':
                resampled = np.clip(resampled, 0, 1.0)
            if key == 'r_insc_to_r_eq_ratio':
                resampled = np.clip(resampled, 0, 1.5)
            result[key] = resampled.tolist()
        except Exception:
            result[key] = ([float('nan')] * n_points
                           if key in nanable_keys else [0.0] * n_points)

    # ---- dA/ds 归一化变化率 (沿重采样均匀点计算, 数值稳定) ----
    try:
        area_uniform = np.asarray(result.get('area', [0.0] * n_points),
                                   dtype=float)
        arc_uniform = np.asarray(result['arc_length_mm'], dtype=float)
        if np.sum(area_uniform > 0) >= 3 and np.all(np.diff(arc_uniform) > 0):
            grad = np.gradient(area_uniform, arc_uniform)
            with np.errstate(divide='ignore', invalid='ignore'):
                dA_ds = grad / np.where(area_uniform > 1e-6,
                                         area_uniform, np.nan)
            # 缺失区段置 NaN, 不污染下游
            dA_ds[area_uniform <= 0] = np.nan
            result['dA_ds_norm'] = dA_ds.tolist()
        else:
            result['dA_ds_norm'] = [float('nan')] * n_points
    except Exception:
        result['dA_ds_norm'] = [float('nan')] * n_points

    return result

# ============================================================
# 主入口 (改为读 JSON 驱动)
# ============================================================
def _branchpoint_arcs_for_path(seg_path, nodes, branchpoint_ids):
    """返回当前段路径上所有分叉点的弧长位置。"""
    if not seg_path or not branchpoint_ids:
        return []
    coords = path_to_coords(seg_path, nodes)
    if len(coords) != len(seg_path):
        return []
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(diffs)))
    return [float(arc[i]) for i, nid in enumerate(seg_path)
            if int(nid) in branchpoint_ids]


def extract_profiles(stl_path, n_points=100, pitch=0.5,
                     curvature_window=7, section_step=3,
                     edge_margin_pct=0.05,
                     edge_margin_mm=8.0,
                     inscribed_factor=1.8,
                     ownership_factor=1.8,
                     junction_policy='min_valid',
                     max_diameter_rate_per_mm=0.5):
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
        ownership_factor:  中心线锚定清洗半径倍数 (默认 1.8).
                           用于保留当前血管主体并裁剪分叉污染区域.
        junction_policy:   分叉/交叉点保护策略:
                           'min_valid' 用本段可信最小截面替换交叉区;
                           'cap_min' 只封顶异常大截面;
                           'keep' 保留 clean area, 不替换。
        max_diameter_rate_per_mm: 沿管轴允许的等效直径相对变化率 (1/mm),
                                  默认 0.5 = 每 mm 最多 50% 相对变化.
                                  超阈孤立点视为伪影截面 (单点塌陷/膨胀).
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
    n_total_junction_protected = 0
    n_total_junction_replaced = 0
    n_total_rejected_oversize = 0
    n_total_local_outliers = 0
    n_total_rate_outliers = 0
    branchpoint_ids = {
        int(bp['id']) for bp in seg_data.get('branch_points', [])
        if isinstance(bp, dict) and 'id' in bp
    }

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
                inscribed_factor=inscribed_factor,
                ownership_factor=ownership_factor,
                max_diameter_rate_per_mm=max_diameter_rate_per_mm)

            if raw_profile is None:
                profiles[seg_name] = None
                continue

            n_total_rejected_oversize += raw_profile.get(
                '_n_rejected_oversize', 0)
            n_total_local_outliers += raw_profile.get(
                '_n_local_outliers', 0)
            n_total_rate_outliers += raw_profile.get(
                '_n_rate_outliers', 0)

            # 重采样到 n_points
            resampled = _resample_profile(raw_profile, n_points=n_points)
            if resampled is None:
                profiles[seg_name] = None
                continue

            seg_path_ids = [int(nid) for nid in seg_info['path']]
            branchpoint_arcs = _branchpoint_arcs_for_path(
                seg_path_ids, nodes, branchpoint_ids)
            terminal_start = seg_path_ids[0] not in branchpoint_ids
            terminal_end = seg_path_ids[-1] not in branchpoint_ids

            # 应用真实末端掩码 + 交叉区最小截面替换/封顶
            resampled = _apply_endpoint_mask(
                resampled,
                edge_margin_pct=edge_margin_pct,
                edge_margin_mm=edge_margin_mm,
                branchpoint_arcs=branchpoint_arcs,
                terminal_start=terminal_start,
                terminal_end=terminal_end,
                junction_policy=junction_policy)

            # 透传过滤元信息
            resampled['n_rejected_oversize'] = int(
                raw_profile.get('_n_rejected_oversize', 0))
            resampled['n_local_outliers'] = int(
                raw_profile.get('_n_local_outliers', 0))
            resampled['n_rate_outliers'] = int(
                raw_profile.get('_n_rate_outliers', 0))
            resampled['n_section_success'] = int(
                raw_profile.get('_n_success', 0))

            n_total_masked += resampled.get('n_masked_endpoints', 0)
            n_total_junction_protected += resampled.get(
                'n_junction_protected', 0)
            n_total_junction_replaced += resampled.get(
                'n_junction_replaced', 0)
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
        'ownership_factor': float(ownership_factor),
        'junction_policy': junction_policy,
        'max_diameter_rate_per_mm': float(max_diameter_rate_per_mm),
        'n_total_masked': int(n_total_masked),
        'n_total_junction_protected': int(n_total_junction_protected),
        'n_total_junction_replaced': int(n_total_junction_replaced),
        'n_total_rejected_oversize': int(n_total_rejected_oversize),
        'n_total_local_outliers': int(n_total_local_outliers),
        'n_total_rate_outliers': int(n_total_rate_outliers),
        # 新增逐点通道清单 (便于训练侧统一索引)
        'pointwise_channels': [
            'position', 'arc_length_mm',
            'area', 'perimeter', 'eq_diameter',
            'raw_area', 'raw_perimeter', 'raw_eq_diameter',
            'anchor_radius', 'owned_radius',
            'hydraulic_diameter',        # 4A/P, 非圆截面有效直径
            'circularity',
            'solidity',                  # A / 凸包面积, ∈ (0,1], 1=凸
            'r_insc_to_r_eq_ratio',      # 2r_insc / D_eq, 瓶颈程度
            'n_components',              # lumen 分量数 (1=正常, 2+=被血栓隔断)
            'junction_replaced',         # 1=交叉区使用可信最小截面替换/封顶
            'curvature',
            'torsion',                   # Frenet 挠率, 中心线 3D 扭转 (NaN 友好)
            'dA_ds_norm',                # (dA/ds)/A, 局部锥度 (NaN 友好)
            'inscribed_radius',
        ],
    }

    out_path = os.path.join(parentdir, "centerline_pointwise_profiles.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False, allow_nan=True)

    valid_segs = [k for k, v in profiles.items()
                   if v is not None and not k.startswith('_')]
    print(f"  剖面提取完成: {len(valid_segs)} 个段, "
          f"端点掩码 {n_total_masked} 处, "
          f"交叉区保护 {n_total_junction_protected} 处 "
          f"(替换/封顶 {n_total_junction_replaced} 处), "
          f"形状/内切超限剔除 {n_total_rejected_oversize} 处, "
          f"局部异常剔除 {n_total_local_outliers} 处, "
          f"变化率剔除 {n_total_rate_outliers} 处")
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
