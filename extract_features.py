"""
门静脉中心线统计特征提取（v4 - NaN-aware, 适配端点掩码剖面）
====================================================
不再自己做解剖识别, 直接读 segment_vessels.py 写入的
centerline_profiles.json, 按段跑特征。

支持的段:
    MPV, SV, SMV, LPV, RPV, TIPS, LGV, PGV
对每段计算:
    长度、曲折度、平均/最大曲率、平均/最大直径、平均面积、面积变异系数、圆度
另含全局特征:
    total_centerline_length, sv_smv_diameter_ratio, sv_smv_angle (若可计算)

v4 变更:
    - 截面统计跳过 NaN (端点掩码区), 用 nanmean/nanmax/nanstd
    - 与 extract_profiles.py 的 _apply_endpoint_mask 配合

输出:
    portal_vein_features.json
"""

import os
import json
import numpy as np

from utils import (load_tree, path_to_coords, node_distance,
                   path_physical_length)
from system_features import (compute_system_features,
                              SYSTEM_FEATURE_NAMES,
                              SYSTEM_FEATURE_LABELS_CN)


# ============================================================
# 常量
# ============================================================

# 所有可能出现的段名 (顺序固定, 决定输出 JSON 的键顺序)
ALL_SEG_NAMES = ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'tips', 'lgv', 'pgv']

# 每段的统计特征键 (与 correlation_analysis.PER_SEG_FEATURES 一致)
PER_SEG_FEATURE_KEYS = [
    'length', 'tortuosity',
    'mean_curvature', 'max_curvature',
    'mean_diameter', 'max_diameter',
    'mean_area', 'area_cv',
    'mean_circularity',
]

# 全局 (单个数值) 特征键
GLOBAL_FEATURE_KEYS = [
    'total_centerline_length',
    'sv_smv_diameter_ratio',
    'sv_smv_angle',
    'has_lgv', 'has_pgv', 'has_compensation_vessel', 'has_tips',
]


# ============================================================
# NaN 安全的统计工具
# ============================================================

def _safe_nanmean(arr):
    """跳过 NaN 求均值, 全 NaN/空数组返回 None。"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmean(arr))


def _safe_nanmax(arr):
    """跳过 NaN 求最大值, 全 NaN/空数组返回 None。"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmax(arr))


def _safe_nanstd(arr):
    """跳过 NaN 求标准差, 全 NaN/空数组返回 None。"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanstd(arr))


# ============================================================
# 几何工具
# ============================================================

def _tortuosity_arc_over_chord(coords):
    """曲折度: 弧长/弦长 (≥ 1.0, 越大越弯)。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return 1.0
    chord = float(np.linalg.norm(coords[-1] - coords[0]))
    arclen = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    return arclen / chord if chord > 1e-8 else 1.0


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


def _interior_curvature_stats(coords, window=7):
    """返回 (mean_curv, max_curv) 对内部点的统计。"""
    if len(coords) < 3:
        return 0.0, 0.0
    curvatures = _curvature_sliding_window(coords, window)
    half = window // 2
    interior = curvatures[half:-half] if len(curvatures) > window else curvatures
    interior = interior[interior > 0]
    if len(interior) == 0:
        return 0.0, 0.0
    return float(np.mean(interior)), float(np.max(interior))


def _direction_at_start(coords, sample_dist=8.0):
    """从首端出发的单位方向 (沿弧长 sample_dist mm)。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return None
    cumlen = 0.0
    sample_pt = coords[-1]
    for i in range(1, len(coords)):
        cumlen += np.linalg.norm(coords[i] - coords[i - 1])
        if cumlen >= sample_dist:
            sample_pt = coords[i]
            break
    direction = sample_pt - coords[0]
    n = np.linalg.norm(direction)
    return direction / n if n > 1e-6 else None


# ============================================================
# 段级几何特征
# ============================================================

def _seg_length_features(seg_info):
    """从 JSON segment 的 length_mm 字段直接取(已是物理长度)。"""
    return seg_info['length_mm'] if seg_info else None


def _seg_tortuosity_features(seg_info):
    """从 JSON 取 tortuosity (注: JSON 里存的是 1-chord/arc, 需转换为 arc/chord)。"""
    if seg_info is None:
        return None
    # JSON 中 tortuosity 定义为 1 - chord/arc (越大越弯, 0~1)
    # 这里转换为 arc/chord (≥ 1, 与旧版一致)
    t = seg_info['tortuosity']
    if t >= 1.0 - 1e-8:
        return float('inf')
    return 1.0 / (1.0 - t)


def _seg_curvature_features(coords, window=7):
    return _interior_curvature_stats(coords, window)


# ============================================================
# 截面特征 (v4: 优先从 pointwise JSON 读取, 跳过 NaN)
# ============================================================

def _seg_section_features_from_profile(profile):
    """
    从已有的 pointwise 剖面 (含端点 NaN 掩码) 计算段级统计。
    优先用这个, 因为可以直接复用 _apply_endpoint_mask 的结果。

    输入 profile: dict, 含 area, eq_diameter, perimeter, circularity 等列表
                  (端点掩码位置的值已是 NaN)
    返回: dict
    """
    if profile is None:
        return {'mean_diameter': None, 'max_diameter': None,
                'mean_area': None, 'area_cv': None,
                'mean_circularity': None}

    areas = np.asarray(profile.get('area', []), dtype=float)
    diameters = np.asarray(profile.get('eq_diameter', []), dtype=float)
    circularities = np.asarray(profile.get('circularity', []), dtype=float)

    mean_a = _safe_nanmean(areas)
    std_a = _safe_nanstd(areas)
    if mean_a is not None and mean_a > 1e-8 and std_a is not None:
        cv = std_a / mean_a
    else:
        cv = None

    return {
        'mean_diameter': _safe_nanmean(diameters),
        'max_diameter': _safe_nanmax(diameters),
        'mean_area': mean_a,
        'area_cv': cv,
        'mean_circularity': _safe_nanmean(circularities),
    }


def _seg_section_features_from_mesh(coords, mesh, sample_step=3):
    """
    沿一段中心线计算截面统计 (回退方案: 现场计算, 无端点掩码)。
    仅在 pointwise JSON 缺失时使用。
    """
    if len(coords) < 2 or mesh is None:
        return {'mean_diameter': None, 'max_diameter': None,
                'mean_area': None, 'area_cv': None,
                'mean_circularity': None}

    from extract_profiles import _compute_cross_section, _compute_tangents

    tangents = _compute_tangents(coords)
    M = len(coords)

    indices = list(range(0, M, sample_step))
    if indices[-1] != M - 1:
        indices.append(M - 1)

    areas, perimeters = [], []
    for idx in indices:
        a, p = _compute_cross_section(mesh, coords[idx], tangents[idx])
        if a > 0:
            areas.append(a)
            perimeters.append(p)

    if not areas:
        return {'mean_diameter': None, 'max_diameter': None,
                'mean_area': None, 'area_cv': None,
                'mean_circularity': None}

    areas = np.asarray(areas)
    perimeters = np.asarray(perimeters)

    eq_diameters = np.sqrt(4.0 * areas / np.pi)

    valid = (areas > 0) & (perimeters > 0)
    if np.any(valid):
        circ = (4.0 * np.pi * areas[valid]) / (perimeters[valid] ** 2)
        mean_circ = float(np.mean(np.clip(circ, 0, 1.5)))
    else:
        mean_circ = None

    mean_area = float(np.mean(areas))
    return {
        'mean_diameter': float(np.mean(eq_diameters)),
        'max_diameter': float(np.max(eq_diameters)),
        'mean_area': mean_area,
        'area_cv': float(np.std(areas) / mean_area) if mean_area > 1e-8 else None,
        'mean_circularity': mean_circ,
    }


# ============================================================
# SV-SMV 夹角 (基于 JSON 分段)
# ============================================================

def _compute_sv_smv_angle_from_segments(seg_dict, nodes,
                                         n_fit_points=10,
                                         sample_dist=None):
    """
    SV / SMV 段的起点应为 SV-SMV 汇合点 (confluence)。
    用每段从起点出发的方向向量计算夹角。
    """
    sv_info = seg_dict.get('sv')
    smv_info = seg_dict.get('smv')

    if sv_info is None or smv_info is None:
        return None, "SV 或 SMV 段缺失"

    sv_path = sv_info['path']
    smv_path = smv_info['path']
    if len(sv_path) < 2 or len(smv_path) < 2:
        return None, "SV / SMV 段点数不足"

    sv_start = sv_path[0]
    smv_start = smv_path[0]
    if sv_start != smv_start:
        # 不在同一 bp, 不能算严格夹角
        return None, (f"SV 起点({sv_start}) ≠ SMV 起点({smv_start}), "
                      f"无共同 confluence")

    sv_coords = path_to_coords(sv_path[:min(n_fit_points + 1, len(sv_path))],
                                nodes)
    smv_coords = path_to_coords(smv_path[:min(n_fit_points + 1, len(smv_path))],
                                 nodes)

    def _fit_dir(coords):
        d = np.mean(coords[1:] - coords[0], axis=0)
        n = np.linalg.norm(d)
        return d / n if n > 1e-8 else None

    d1, d2 = _fit_dir(sv_coords), _fit_dir(smv_coords)
    if d1 is None or d2 is None:
        return None, "方向向量退化"

    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1))))
    conf_node = nodes[sv_start]
    return {
        'angle_degrees': round(angle_deg, 2),
        'confluence_point_physical': [round(conf_node['x'], 2),
                                       round(conf_node['y'], 2),
                                       round(conf_node['z'], 2)],
        'confluence_node_id': int(sv_start),
        'branch1_direction': [round(float(v), 4) for v in d1],
        'branch2_direction': [round(float(v), 4) for v in d2],
        'n_fit_points': n_fit_points,
        '_branch1_coords': sv_coords.tolist(),
        '_branch2_coords': smv_coords.tolist(),
    }, None


# ============================================================
# 单段特征汇总
# ============================================================

def _features_for_one_segment(seg_name, seg_info, nodes, mesh,
                               curvature_window, sample_step,
                               pointwise_data=None):
    """
    对单段算所有几何特征, 返回 flat dict, 键带前缀 seg_name_。

    pointwise_data: 若提供 centerline_pointwise_profiles.json 的字典,
                    截面特征优先从该数据读取(已含端点 NaN 掩码)。
                    否则回退到从 mesh 现场计算(无掩码)。
    """
    prefix = seg_name + '_'

    if seg_info is None:
        # 段缺失, 输出 None
        return {
            prefix + 'length':          None,
            prefix + 'tortuosity':      None,
            prefix + 'mean_curvature':  None,
            prefix + 'max_curvature':   None,
            prefix + 'mean_diameter':   None,
            prefix + 'max_diameter':    None,
            prefix + 'mean_area':       None,
            prefix + 'area_cv':         None,
            prefix + 'mean_circularity':None,
        }

    coords = path_to_coords(seg_info['path'], nodes)

    # 长度 + 曲折度: 直接从 JSON 取
    length = _seg_length_features(seg_info)
    tort = _seg_tortuosity_features(seg_info)

    # 曲率: 现算 (基于 path 上的离散点)
    mean_curv, max_curv = _seg_curvature_features(coords, curvature_window)

    # 截面特征: 优先用 pointwise JSON (含 NaN 掩码), 否则现场算
    sec = None
    if pointwise_data is not None:
        profile = pointwise_data.get(seg_name)
        if profile is not None:
            sec = _seg_section_features_from_profile(profile)
    if sec is None:
        sec = _seg_section_features_from_mesh(coords, mesh, sample_step)

    return {
        prefix + 'length':          float(length) if length is not None else None,
        prefix + 'tortuosity':      float(tort)   if tort   is not None else None,
        prefix + 'mean_curvature':  float(mean_curv),
        prefix + 'max_curvature':   float(max_curv),
        prefix + 'mean_diameter':   sec['mean_diameter'],
        prefix + 'max_diameter':    sec['max_diameter'],
        prefix + 'mean_area':       sec['mean_area'],
        prefix + 'area_cv':         sec['area_cv'],
        prefix + 'mean_circularity':sec['mean_circularity'],
    }


# ============================================================
# 全局特征
# ============================================================

def _global_features(nodes, adj, all_seg_features, seg_dict=None):
    """全局/跨段衍生特征。

    seg_dict: 若提供, 加入代偿血管/TIPS 存在性二值特征。
    """
    # 总中心线长度
    visited_edges = set()
    total = 0.0
    for nid in nodes:
        for nb in adj[nid]:
            edge = (min(nid, nb), max(nid, nb))
            if edge not in visited_edges:
                visited_edges.add(edge)
                total += node_distance(nodes[nid], nodes[nb])

    sv_d = all_seg_features.get('sv_mean_diameter')
    smv_d = all_seg_features.get('smv_mean_diameter')
    sv_smv_ratio = (sv_d / smv_d) if (sv_d and smv_d and smv_d > 1e-8) else None

    out = {
        'total_centerline_length': float(total),
        'sv_smv_diameter_ratio': sv_smv_ratio,
    }

    # 二值: 代偿血管/TIPS 存在性 (即使段统计是 None, 存在性也是 0/1)
    if seg_dict is not None:
        out['has_lgv'] = 1 if seg_dict.get('lgv') is not None else 0
        out['has_pgv'] = 1 if seg_dict.get('pgv') is not None else 0
        out['has_compensation_vessel'] = 1 if (
            seg_dict.get('lgv') is not None
            or seg_dict.get('pgv') is not None) else 0
        out['has_tips'] = 1 if seg_dict.get('tips') is not None else 0

    return out


# ============================================================
# 主入口
# ============================================================

def extract_all_features(stl_path, n_fit_points=10,
                          curvature_window=7, sample_step=3,
                          pitch=0.5,
                          write_unified=True,
                          write_legacy=True):
    """
    从中心线树 + 分段 JSON + 剖面 JSON 计算所有统计特征 + 系统特征,
    写入:
      - portal_vein_features.json  (扁平字段, 旧版 schema, 供旧的 correlation 工具)
      - unified_features.json      (新, 单文件统一格式; 推荐用于训练)

    若同目录下存在 centerline_pointwise_profiles.json,
    截面统计优先从其中读取 (含端点 NaN 掩码, 跳过端点不可信值)。
    否则回退到从 STL mesh 现场计算 (无掩码)。

    参数:
        stl_path:          STL 文件路径
        n_fit_points:      SV-SMV 夹角拟合点数
        curvature_window:  曲率滑窗大小
        sample_step:       截面采样步长(每隔几个中心线点做一次截面, 仅回退用)
        pitch:             保留参数兼容旧接口(目前不用)
        write_unified:     是否输出 unified_features.json (默认 True)
        write_legacy:      是否输出 portal_vein_features.json (默认 True, 兼容老分析脚本)

    返回:
        all_features: 扁平 dict (含统计 + 系统 + 全局特征)
    """
    parentdir = os.path.dirname(stl_path)
    print(f"\n{'='*60}")
    print(f"特征提取: {os.path.basename(stl_path)}")
    print(f"{'='*60}")

    # ---------- 1. 加载中心线树 ----------
    print("[1/5] 加载中心线树...")
    try:
        nodes, adj, _ = load_tree(stl_path)
    except FileNotFoundError as e:
        print(f"  ✗ 中心线缺失: {e}")
        return {}

    # ---------- 2. 加载分段 JSON ----------
    print("[2/5] 加载分段 JSON (centerline_profiles.json)...")
    seg_json_path = os.path.join(parentdir, "centerline_profiles.json")
    if not os.path.exists(seg_json_path):
        print(f"  ✗ 分段 JSON 不存在, 请先运行 segment_vessels.py")
        return {}

    with open(seg_json_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)
    seg_dict = seg_data.get('segments', {})

    loaded = [n for n in ALL_SEG_NAMES if seg_dict.get(n) is not None]
    print(f"  分段类型: {'POST-TIPS' if seg_data.get('is_post_tips') else 'PRE-TIPS'}"
          f"{', 代偿='+seg_data['compensation_type'] if seg_data.get('compensation_type') else ''}")
    print(f"  含血管段: {loaded}")

    # ---------- 3. 加载 pointwise 剖面 JSON (优先源) ----------
    print("[3/5] 加载剖面 JSON (centerline_pointwise_profiles.json)...")
    pw_json_path = os.path.join(parentdir, "centerline_pointwise_profiles.json")
    pointwise_data = None
    if os.path.exists(pw_json_path):
        try:
            with open(pw_json_path, 'r', encoding='utf-8') as f:
                pointwise_data = json.load(f)
            n_masked = pointwise_data.get('_meta', {}).get('n_total_masked', 0)
            print(f"  ✓ 剖面已加载 (端点掩码 {n_masked} 处)")
        except Exception as e:
            print(f"  ✗ 剖面 JSON 解析失败: {e}, 将回退到 mesh 计算")
            pointwise_data = None
    else:
        print(f"  剖面 JSON 不存在, 将回退到 mesh 计算 (无端点掩码)")

    # ---------- 4. 加载 STL 网格 (回退用) ----------
    mesh = None
    if pointwise_data is None:
        print("[4/5] 加载 STL 网格 (回退用)...")
        try:
            import trimesh
            mesh = trimesh.load(stl_path)
            if not isinstance(mesh, trimesh.Trimesh):
                if hasattr(mesh, 'geometry'):
                    mesh = list(mesh.geometry.values())[0]
                else:
                    mesh = None
            if mesh is not None:
                print(f"  STL: {len(mesh.vertices)}顶点, {len(mesh.faces)}面")
        except Exception as e:
            print(f"  ✗ STL 加载失败: {e}")
            mesh = None
    else:
        print("[4/5] STL 网格加载跳过 (使用剖面 JSON)")

    # ---------- 5. 逐段算特征 ----------
    print("[5/5] 逐段计算特征...")
    all_features = {}

    for seg_name in ALL_SEG_NAMES:
        seg_info = seg_dict.get(seg_name)
        feats = _features_for_one_segment(
            seg_name, seg_info, nodes, mesh,
            curvature_window, sample_step,
            pointwise_data=pointwise_data)
        all_features.update(feats)

        if seg_info is not None:
            L = feats[seg_name + '_length']
            D = feats[seg_name + '_mean_diameter']
            t = feats[seg_name + '_tortuosity']
            d_str = f"{D:5.2f}mm" if D is not None else "  N/A"
            print(f"  [{seg_name.upper():4s}] L={L:6.1f}mm  "
                  f"D={d_str}  arc/chord={t:.3f}")

    # 全局特征 (含代偿/TIPS 二值)
    all_features.update(_global_features(nodes, adj, all_features, seg_dict))
    print(f"  total_centerline_length: {all_features['total_centerline_length']:.2f}mm")
    print(f"  has_lgv={all_features['has_lgv']}  "
          f"has_pgv={all_features['has_pgv']}  "
          f"has_compensation={all_features['has_compensation_vessel']}  "
          f"has_tips={all_features['has_tips']}")

    # SV-SMV 夹角
    angle_result, angle_err = _compute_sv_smv_angle_from_segments(
        seg_dict, nodes, n_fit_points=n_fit_points)
    if angle_result is not None:
        all_features['sv_smv_angle'] = angle_result['angle_degrees']
        # 角度详情单独保存 (保留旧文件以兼容)
        angle_json = os.path.join(parentdir, "sv_smv_angle.json")
        save_data = {k: v for k, v in angle_result.items()
                     if not k.startswith('_')}
        with open(angle_json, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"  sv_smv_angle: {angle_result['angle_degrees']:.1f}°")
    else:
        all_features['sv_smv_angle'] = None
        save_data = None
        print(f"  sv_smv_angle: None ({angle_err})")

    # ---------- 6. 系统 / 联合特征 (新增) ----------
    print("[6/6] 计算系统/联合特征...")
    branch_points = [bp['id'] for bp in seg_data.get('branch_points', [])]
    sys_feats = compute_system_features(
        seg_dict, all_features, pointwise_data, nodes, branch_points)
    all_features.update(sys_feats)

    n_sys_valid = sum(1 for k in SYSTEM_FEATURE_NAMES
                      if all_features.get(k) is not None)
    print(f"  有效系统特征: {n_sys_valid}/{len(SYSTEM_FEATURE_NAMES)}")
    # 选择性打印重点系统特征
    for k in ('confluence_murray3_ratio', 'splenic_dominance_index',
              'inflow_resistance_asymmetry', 'mpv_effective_radius',
              'collateral_burden_score'):
        v = all_features.get(k)
        if v is None:
            continue
        print(f"    {k}: {v:.4f}")

    # 元信息
    all_features['_meta'] = {
        'patient_id': seg_data.get('patient_id'),
        'is_post_tips': seg_data.get('is_post_tips'),
        'has_compensation': seg_data.get('has_compensation'),
        'compensation_type': seg_data.get('compensation_type'),
        'used_pointwise_profiles': pointwise_data is not None,
    }

    # ---------- 保存 ----------
    if write_legacy:
        # 旧版扁平 schema, 兼容 correlation_analysis.py / profile_correlation.py
        output_path = os.path.join(parentdir, "portal_vein_features.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, indent=2, ensure_ascii=False, allow_nan=True)
        n_valid = sum(1 for k, v in all_features.items()
                      if k != '_meta' and v is not None)
        n_null = sum(1 for k, v in all_features.items()
                     if k != '_meta' and v is None)
        print(f"\n[Legacy] 扁平特征已保存: {output_path}")
        print(f"  有效 {n_valid}, 为 None {n_null}")

    if write_unified:
        unified_path = os.path.join(parentdir, "unified_features.json")
        unified = build_unified_features(
            all_features, pointwise_data, seg_data,
            angle_detail=save_data)
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(unified, f, indent=2, ensure_ascii=False, allow_nan=True)
        print(f"[Unified] 统一特征已保存: {unified_path}")

    return all_features


# ============================================================
# 统一 JSON 组装
# ============================================================

UNIFIED_SCHEMA_VERSION = "v1"


def _nested_per_seg(flat, prefix, keys):
    """从 flat dict 抽出 {prefix}_<key> 组装为 {key: value} 嵌套字典."""
    out = {k: flat.get(f"{prefix}_{k}") for k in keys}
    if all(v is None for v in out.values()):
        return None
    return out


def build_unified_features(flat_features, pointwise_data, seg_data,
                            angle_detail=None):
    """
    把各路输出聚合为单一 JSON 结构, 便于训练时一次加载.

    顶层结构:
        _meta            病人 ID / TIPS / 代偿
        _index           各字段块的说明 (字段使用文档)
        statistical      每段 9 个标量特征 {seg: {key: value}}
        system           系统 / 联合特征 (扁平)
        global           全局/树级 (扁平, 含 has_*)
        sv_smv_angle     夹角详细信息 (向量, 汇合点等)
        pointwise        逐点剖面 (沿用 extract_profiles 的格式)
        segments_meta    每段路径长度 / 节点 id (来自 centerline_profiles)
    """
    flat = dict(flat_features)
    flat.pop('_meta', None)

    # ---- statistical: per-segment ----
    statistical = {}
    for seg_name in ALL_SEG_NAMES:
        block = _nested_per_seg(flat, seg_name, PER_SEG_FEATURE_KEYS)
        if block is not None:
            statistical[seg_name] = block

    # ---- system features ----
    system = {k: flat.get(k) for k in SYSTEM_FEATURE_NAMES}

    # ---- global ----
    global_block = {k: flat.get(k) for k in GLOBAL_FEATURE_KEYS}

    # ---- pointwise (剥掉 _meta 单独处理, 内部有 inscribed_radius 等) ----
    pointwise_block = {}
    pointwise_meta = {}
    if pointwise_data:
        for k, v in pointwise_data.items():
            if k == '_meta':
                pointwise_meta = v
            elif v is not None:
                pointwise_block[k] = v

    # ---- segments_meta: 每段的 path / 长度 / 起止节点 ----
    seg_meta_block = {}
    for nm, info in (seg_data.get('segments') or {}).items():
        if info is None:
            continue
        seg_meta_block[nm] = {
            'length_mm': info.get('length_mm'),
            'tortuosity': info.get('tortuosity'),
            'mean_curvature': info.get('mean_curvature'),
            'n_points': info.get('n_points'),
            'endpoints_id': info.get('endpoints_id'),
            'endpoints_coord': info.get('endpoints_coord'),
        }

    # ---- _index: 文档说明 ----
    index = {
        'statistical': {
            'description': '每段 9 个标量统计特征 (从 centerline_pointwise_profiles 派生, 含端点 NaN 掩码)',
            'segments': list(statistical.keys()),
            'feature_keys': PER_SEG_FEATURE_KEYS,
            'flat_key_pattern': '<seg>_<feature>  e.g. mpv_mean_diameter',
        },
        'system': {
            'description': '系统 / 联合特征 (跨血管几何关系, 文献先验, 与 PPG/HVPG 相关)',
            'feature_names': SYSTEM_FEATURE_NAMES,
            'groups': {
                'A_angles': [n for n in SYSTEM_FEATURE_NAMES if n.startswith('angle_') or 'planarity' in n],
                'B_diameter_area_ratio': [
                    'sv_smv_diameter_asymmetry', 'sv_mpv_diameter_ratio',
                    'smv_mpv_diameter_ratio',
                    'confluence_murray3_ratio', 'confluence_murray3_deviation',
                    'confluence_area_ratio',
                    'mpv_bifurc_murray3_ratio', 'mpv_bifurc_murray3_deviation',
                    'mpv_bifurc_area_ratio',
                    'lpv_rpv_diameter_asymmetry',
                    'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
                    'splenic_dominance_index'],
                'C_length_tortuosity': [
                    'splenoportal_path_chord_ratio',
                    'collateral_length_mpv_ratio',
                    'diameter_weighted_tortuosity'],
                'D_hydraulic': [
                    'mpv_resistance_integral', 'sv_resistance_integral',
                    'smv_resistance_integral', 'lpv_resistance_integral',
                    'rpv_resistance_integral', 'tips_resistance_integral',
                    'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
                    'mpv_effective_radius', 'tips_inflow_resistance_ratio'],
                'E_topology': [
                    'collateral_burden_score', 'n_collaterals_detected',
                    'branchpoint_density_per_cm',
                    'mpv_taper_coefficient',
                    'mpv_proximal_diameter', 'mpv_distal_diameter',
                    'mpv_min_max_diameter_ratio',
                    'tree_area_conservation_mean_dev'],
            },
            'labels_cn': SYSTEM_FEATURE_LABELS_CN,
        },
        'global': {
            'description': '全局/树级标量, 含侧支/TIPS 存在性二值',
            'feature_keys': GLOBAL_FEATURE_KEYS,
        },
        'sv_smv_angle': {
            'description': 'SV-SMV 汇合处的几何细节 (汇合点坐标 + 两支单位向量)',
        },
        'pointwise': {
            'description': '逐点剖面 (重采样到 n_points), 含端点 NaN 掩码',
            'segments': list(pointwise_block.keys()),
            'feature_keys': ['position', 'arc_length_mm', 'total_length_mm',
                             'area', 'eq_diameter', 'perimeter', 'circularity',
                             'curvature', 'inscribed_radius',
                             'edge_margin_pct', 'edge_margin_mm',
                             'n_masked_endpoints', 'n_rejected_oversize',
                             'n_section_success'],
            'mask_explanation': (
                'area / eq_diameter / perimeter / circularity / inscribed_radius '
                '在端点保护带 (edge_margin_pct + edge_margin_mm) 内为 NaN; '
                '另对超过内切直径×inscribed_factor 的截面也丢弃 (防止穿透到邻近血管).'
            ),
        },
        'segments_meta': {
            'description': '每段路径的几何概览 (来自 centerline_profiles.json)',
        },
    }

    return {
        '_schema_version': UNIFIED_SCHEMA_VERSION,
        '_meta': {
            'patient_id': seg_data.get('patient_id'),
            'is_post_tips': seg_data.get('is_post_tips'),
            'has_compensation': seg_data.get('has_compensation'),
            'compensation_type': seg_data.get('compensation_type'),
        },
        '_index': index,
        'statistical': statistical,
        'system': system,
        'global': global_block,
        'sv_smv_angle': angle_detail,
        'segments_meta': seg_meta_block,
        'pointwise': pointwise_block,
        'pointwise_meta': pointwise_meta,
    }


if __name__ == '__main__':
    import sys, time
    p = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    t0 = time.time()
    extract_all_features(p)
    print(f"\n耗时: {time.time() - t0:.2f}s")