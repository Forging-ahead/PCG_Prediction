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
        description_path = os.path.join(parentdir, FEATURE_DESCRIPTION_FILENAME)
        unified = build_unified_features(
            all_features, pointwise_data, seg_data,
            angle_detail=save_data)
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(unified, f, indent=2, ensure_ascii=False, allow_nan=True)
        with open(description_path, 'w', encoding='utf-8') as f:
            json.dump(build_feature_description(), f, indent=2,
                      ensure_ascii=False, allow_nan=True)
        print(f"[Unified] 统一特征已保存: {unified_path}")
        print(f"[Unified] 特征说明已保存: {description_path}")

    return all_features


# ============================================================
# 统一 JSON 组装
# ============================================================

UNIFIED_SCHEMA_VERSION = "v2"
FEATURE_DESCRIPTION_FILENAME = "feature_description.json"

POINTWISE_CORE_VALID_KEYS = [
    'area',
    'eq_diameter',
    'perimeter',
]

SYSTEM_FEATURE_GROUPS = {
    'A_angles': [
        'angle_sv_smv', 'angle_mpv_lpv', 'angle_mpv_rpv',
        'angle_lpv_rpv', 'angle_mpv_bifurc_total',
        'mpv_bifurc_planarity_deg', 'angle_mpv_tips',
    ],
    'B_diameter_area_ratio': [
        'sv_smv_diameter_asymmetry', 'sv_mpv_diameter_ratio',
        'smv_mpv_diameter_ratio', 'confluence_murray3_ratio',
        'confluence_murray3_deviation', 'confluence_area_ratio',
        'mpv_bifurc_murray3_ratio', 'mpv_bifurc_murray3_deviation',
        'mpv_bifurc_area_ratio', 'lpv_rpv_diameter_asymmetry',
        'lgv_mpv_diameter_ratio', 'pgv_mpv_diameter_ratio',
        'splenic_dominance_index',
    ],
    'C_length_tortuosity': [
        'splenoportal_path_chord_ratio',
        'collateral_length_mpv_ratio',
        'diameter_weighted_tortuosity',
    ],
    'D_hydraulic': [
        'mpv_resistance_integral', 'sv_resistance_integral',
        'smv_resistance_integral', 'lpv_resistance_integral',
        'rpv_resistance_integral', 'tips_resistance_integral',
        'inflow_parallel_resistance', 'inflow_resistance_asymmetry',
        'mpv_effective_radius', 'tips_inflow_resistance_ratio',
    ],
    'E_topology': [
        'collateral_burden_score', 'n_collaterals_detected',
        'branchpoint_density_per_cm', 'mpv_taper_coefficient',
        'mpv_proximal_diameter', 'mpv_distal_diameter',
        'mpv_min_max_diameter_ratio', 'tree_area_conservation_mean_dev',
    ],
    'F_clinical': [
        'sv_max_to_mpv_max_diam_ratio', 'mpv_trunk_length_mm',
        'max_tortuosity_index', 'mean_tortuosity_index',
        'max_collateral_diameter_mm',
        'area_conservation_bifurc_deviation',
        'tips_stent_diameter_mm', 'tips_stent_length_mm',
        'pvt_severity_grade', 'min_lumen_area_to_max_ratio_mpv',
        'cavernous_transformation_flag',
    ],
}

SYSTEM_FEATURE_DEPENDENCIES = {
    'angle_sv_smv': {'required_vessels': ['sv', 'smv']},
    'angle_mpv_lpv': {'required_vessels': ['mpv', 'lpv']},
    'angle_mpv_rpv': {'required_vessels': ['mpv', 'rpv']},
    'angle_lpv_rpv': {'required_vessels': ['lpv', 'rpv']},
    'angle_mpv_bifurc_total': {'required_vessels': ['mpv', 'lpv', 'rpv']},
    'mpv_bifurc_planarity_deg': {'required_vessels': ['mpv', 'lpv', 'rpv']},
    'angle_mpv_tips': {'required_vessels': ['mpv', 'tips']},
    'sv_smv_diameter_asymmetry': {
        'required_vessels': ['sv', 'smv'],
        'source_features': ['sv_mean_diameter', 'smv_mean_diameter'],
    },
    'sv_mpv_diameter_ratio': {
        'required_vessels': ['sv', 'mpv'],
        'source_features': ['sv_mean_diameter', 'mpv_mean_diameter'],
    },
    'smv_mpv_diameter_ratio': {
        'required_vessels': ['smv', 'mpv'],
        'source_features': ['smv_mean_diameter', 'mpv_mean_diameter'],
    },
    'confluence_murray3_ratio': {
        'required_vessels': ['mpv', 'sv', 'smv'],
        'source_features': [
            'mpv_mean_diameter', 'sv_mean_diameter', 'smv_mean_diameter'],
    },
    'confluence_murray3_deviation': {
        'required_vessels': ['mpv', 'sv', 'smv'],
        'source_features': ['confluence_murray3_ratio'],
    },
    'confluence_area_ratio': {
        'required_vessels': ['mpv', 'sv', 'smv'],
        'source_features': ['mpv_mean_area', 'sv_mean_area', 'smv_mean_area'],
    },
    'mpv_bifurc_murray3_ratio': {
        'required_vessels': ['mpv', 'lpv', 'rpv'],
        'source_features': [
            'mpv_mean_diameter', 'lpv_mean_diameter', 'rpv_mean_diameter'],
    },
    'mpv_bifurc_murray3_deviation': {
        'required_vessels': ['mpv', 'lpv', 'rpv'],
        'source_features': ['mpv_bifurc_murray3_ratio'],
    },
    'mpv_bifurc_area_ratio': {
        'required_vessels': ['mpv', 'lpv', 'rpv'],
        'source_features': ['mpv_mean_area', 'lpv_mean_area', 'rpv_mean_area'],
    },
    'lpv_rpv_diameter_asymmetry': {
        'required_vessels': ['lpv', 'rpv'],
        'source_features': ['lpv_mean_diameter', 'rpv_mean_diameter'],
    },
    'lgv_mpv_diameter_ratio': {
        'required_vessels': ['lgv', 'mpv'],
        'source_features': ['lgv_mean_diameter', 'mpv_mean_diameter'],
    },
    'pgv_mpv_diameter_ratio': {
        'required_vessels': ['pgv', 'mpv'],
        'source_features': ['pgv_mean_diameter', 'mpv_mean_diameter'],
    },
    'splenic_dominance_index': {
        'required_vessels': ['sv', 'smv'],
        'source_features': ['sv_mean_diameter', 'smv_mean_diameter'],
    },
    'splenoportal_path_chord_ratio': {'required_vessels': ['sv', 'mpv']},
    'collateral_length_mpv_ratio': {
        'required_vessels': ['mpv'],
        'required_any_vessels': ['lgv', 'pgv'],
        'source_features': ['mpv_length', 'lgv_length', 'pgv_length'],
    },
    'diameter_weighted_tortuosity': {
        'required_min_present_vessels': {
            'vessels': ['mpv', 'sv', 'smv', 'lpv', 'rpv'],
            'min_count': 2,
        },
    },
    'mpv_resistance_integral': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
        'source_features': ['mpv_length'],
    },
    'sv_resistance_integral': {
        'required_vessels': ['sv'], 'requires_pointwise': True,
        'source_features': ['sv_length'],
    },
    'smv_resistance_integral': {
        'required_vessels': ['smv'], 'requires_pointwise': True,
        'source_features': ['smv_length'],
    },
    'lpv_resistance_integral': {
        'required_vessels': ['lpv'], 'requires_pointwise': True,
        'source_features': ['lpv_length'],
    },
    'rpv_resistance_integral': {
        'required_vessels': ['rpv'], 'requires_pointwise': True,
        'source_features': ['rpv_length'],
    },
    'tips_resistance_integral': {
        'required_vessels': ['tips'], 'requires_pointwise': True,
        'source_features': ['tips_length'],
    },
    'inflow_parallel_resistance': {
        'required_vessels': ['sv', 'smv'], 'requires_pointwise': True,
        'source_features': ['sv_resistance_integral',
                            'smv_resistance_integral'],
    },
    'inflow_resistance_asymmetry': {
        'required_vessels': ['sv', 'smv'], 'requires_pointwise': True,
        'source_features': ['sv_resistance_integral',
                            'smv_resistance_integral'],
    },
    'mpv_effective_radius': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
        'source_features': ['mpv_length', 'mpv_resistance_integral'],
    },
    'tips_inflow_resistance_ratio': {
        'required_vessels': ['tips', 'sv', 'smv'],
        'requires_pointwise': True,
        'source_features': ['tips_resistance_integral',
                            'inflow_parallel_resistance'],
    },
    'collateral_burden_score': {
        'required_vessels': ['mpv'],
        'optional_vessels': ['lgv', 'pgv'],
        'source_features': ['mpv_length', 'mpv_mean_diameter'],
    },
    'n_collaterals_detected': {'optional_vessels': ['lgv', 'pgv']},
    'branchpoint_density_per_cm': {
        'required_vessels': ['mpv'],
        'source_features': ['mpv_length'],
    },
    'mpv_taper_coefficient': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
        'source_features': ['mpv_length'],
    },
    'mpv_proximal_diameter': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
        'source_features': ['mpv_length'],
    },
    'mpv_distal_diameter': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
        'source_features': ['mpv_length'],
    },
    'mpv_min_max_diameter_ratio': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
        'source_features': ['mpv_length'],
    },
    'tree_area_conservation_mean_dev': {
        'required_vessel_sets_any': [
            ['mpv', 'sv', 'smv'], ['mpv', 'lpv', 'rpv']],
        'source_features': [
            'mpv_mean_area', 'sv_mean_area', 'smv_mean_area',
            'lpv_mean_area', 'rpv_mean_area'],
    },
    'sv_max_to_mpv_max_diam_ratio': {
        'required_vessels': ['sv', 'mpv'],
        'source_features': ['sv_max_diameter', 'mpv_max_diameter'],
    },
    'mpv_trunk_length_mm': {
        'required_vessels': ['mpv'], 'source_features': ['mpv_length'],
    },
    'max_tortuosity_index': {
        'required_min_present_vessels': {
            'vessels': ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'lgv', 'pgv',
                        'tips'],
            'min_count': 1,
        },
    },
    'mean_tortuosity_index': {
        'required_min_present_vessels': {
            'vessels': ['mpv', 'sv', 'smv', 'lpv', 'rpv', 'lgv', 'pgv',
                        'tips'],
            'min_count': 1,
        },
    },
    'max_collateral_diameter_mm': {'optional_vessels': ['lgv', 'pgv']},
    'area_conservation_bifurc_deviation': {
        'required_vessels': ['mpv', 'lpv', 'rpv'],
        'source_features': ['mpv_mean_area', 'lpv_mean_area', 'rpv_mean_area'],
    },
    'tips_stent_diameter_mm': {
        'required_vessels': ['tips'], 'source_features': ['tips_mean_diameter'],
    },
    'tips_stent_length_mm': {
        'required_vessels': ['tips'], 'source_features': ['tips_length'],
    },
    'pvt_severity_grade': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
    },
    'min_lumen_area_to_max_ratio_mpv': {
        'required_vessels': ['mpv'], 'requires_pointwise': True,
    },
    'cavernous_transformation_flag': {
        'optional_vessels': ['mpv', 'lgv', 'pgv'],
        'source_features': ['mpv_max_diameter', 'branchpoint_density_per_cm'],
    },
}


def _nested_per_seg(flat, prefix, keys):
    """从 flat dict 抽出 {prefix}_<key> 组装为 {key: value} 嵌套字典."""
    out = {k: flat.get(f"{prefix}_{k}") for k in keys}
    if all(v is None for v in out.values()):
        return None
    return out


def _finite_positive_count(values):
    """统计数组中有限且 >0 的元素数。"""
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return 0
    if arr.size == 0:
        return 0
    return int(np.sum(np.isfinite(arr) & (arr > 0)))


def _pointwise_diag(profile):
    """提取 pointwise profile 的简短诊断信息。"""
    if profile is None:
        return {}
    return {
        'n_section_success': profile.get('n_section_success'),
        'n_masked_endpoints': profile.get('n_masked_endpoints'),
        'n_rejected_oversize': profile.get('n_rejected_oversize'),
        'n_local_outliers': profile.get('n_local_outliers'),
        'n_rate_outliers': profile.get('n_rate_outliers'),
        'n_junction_protected': profile.get('n_junction_protected'),
        'n_junction_replaced': profile.get('n_junction_replaced'),
        'valid_area_points': _finite_positive_count(profile.get('area', [])),
        'valid_diameter_points': _finite_positive_count(
            profile.get('eq_diameter', [])),
    }


def _is_missing_json_value(value):
    """JSON 输出前统一判断 None / NaN / inf 这类不可训练值。"""
    if value is None:
        return True
    try:
        if isinstance(value, (float, np.floating)):
            return not np.isfinite(float(value))
    except Exception:
        return False
    return False


def _list_like(value):
    return isinstance(value, (list, tuple))


def _infer_point_count(profile):
    lengths = [
        len(v) for v in profile.values()
        if _list_like(v) and not isinstance(v, (str, bytes))
    ]
    if not lengths:
        return 0
    return max(set(lengths), key=lengths.count)


def _clean_scalar_for_json(value):
    if _is_missing_json_value(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clean_pointwise_profile_for_unified(profile):
    """
    写 unified_features.json 前清洗逐点剖面。

    extract_profiles 为了保持 100 点对齐, 会在端点/交叉保护区把截面值写成
    NaN。统一特征文件用于训练时不再保留这些占位点: 只要核心截面通道
    area/eq_diameter/perimeter 任一无效, 就删除该位置在所有逐点通道中的值。
    """
    if not isinstance(profile, dict):
        return None

    n_points = _infer_point_count(profile)
    if n_points <= 0:
        return dict(profile)

    core_keys = [
        k for k in POINTWISE_CORE_VALID_KEYS
        if _list_like(profile.get(k)) and len(profile.get(k)) == n_points
    ]
    point_feature_keys = [
        k for k, v in profile.items()
        if _list_like(v) and len(v) == n_points
    ]

    keep_mask = []
    for i in range(n_points):
        keep = True
        for key in point_feature_keys:
            if _is_missing_json_value(profile[key][i]):
                keep = False
                break
        if not keep:
            keep_mask.append(False)
            continue
        for key in core_keys:
            value = profile[key][i]
            if _is_missing_json_value(value):
                keep = False
                break
            try:
                if float(value) <= 0:
                    keep = False
                    break
            except Exception:
                keep = False
                break
        keep_mask.append(keep)

    kept_indices = [i for i, keep in enumerate(keep_mask) if keep]
    cleaned = {}
    for key, value in profile.items():
        if _list_like(value) and len(value) == n_points:
            cleaned[key] = [
                _clean_scalar_for_json(value[i])
                for i in kept_indices
            ]
        else:
            cleaned[key] = _clean_scalar_for_json(value)

    cleaned['_point_filter'] = {
        'original_n_points': int(n_points),
        'kept_n_points': int(len(kept_indices)),
        'removed_n_points': int(n_points - len(kept_indices)),
        'removed_reason': (
            '任一逐点通道存在 None/NaN/inf, 或核心截面通道 '
            'area/eq_diameter/perimeter <=0, '
            '该逐点位置已从 unified pointwise 中删除。'
        ),
        'validity_keys': point_feature_keys,
        'positive_core_keys': core_keys,
    }
    return cleaned


def _seg_missing_reason(seg_name, seg_info, profile, feature_key,
                        pointwise_data):
    """解释单段统计特征为什么为 None。"""
    label = seg_name.upper()
    if seg_info is None:
        return (f"{label} 段在 centerline_profiles.json 中为 None: "
                "解剖上可能不存在、该期别不需要该段, 或 segment_vessels 未识别到。")

    section_keys = {
        'mean_diameter', 'max_diameter', 'mean_area',
        'area_cv', 'mean_circularity'
    }
    if feature_key in section_keys:
        if pointwise_data is not None:
            if profile is None:
                return (f"{label} 段存在, 但 centerline_pointwise_profiles.json "
                        "中该段剖面为 None 或缺失; 通常是 extract_profiles "
                        "该段失败、路径太短或截面全部无效。")
            if _finite_positive_count(profile.get('area', [])) == 0:
                return (f"{label} 段剖面存在, 但 area 没有有效正值; "
                        "可能全部被端点掩码、内切半径/形状过滤、局部异常过滤剔除。")
            if feature_key == 'area_cv':
                return (f"{label} 的面积均值无效或接近 0, 无法计算 area_cv。")
            return (f"{label} 的 {feature_key} 缺失: 对应 pointwise 通道 "
                    "没有有限有效值。")
        return (f"{label} 截面统计缺失: pointwise JSON 不存在且 mesh 回退计算 "
                "没有得到有效截面。")

    if feature_key in {'length', 'tortuosity'}:
        return (f"{label} 段存在, 但分段 JSON 中长度/曲折度字段缺失或退化。")
    if feature_key in {'mean_curvature', 'max_curvature'}:
        return (f"{label} 段中心线点数不足或几何退化, 曲率无法稳定计算。")
    return f"{label} 的 {feature_key} 无法计算, 请检查分段和剖面输入。"


def _deps_missing(deps, flat):
    return [d for d in deps if flat.get(d) is None]


def _system_missing_reason(name, flat, seg_dict, pointwise_data,
                           branch_points, angle_detail):
    """解释系统/全局特征为什么为 None。"""
    deps_by_feature = {
        'sv_smv_diameter_asymmetry': ['sv_mean_diameter', 'smv_mean_diameter'],
        'sv_mpv_diameter_ratio': ['sv_mean_diameter', 'mpv_mean_diameter'],
        'smv_mpv_diameter_ratio': ['smv_mean_diameter', 'mpv_mean_diameter'],
        'confluence_murray3_ratio': [
            'mpv_mean_diameter', 'sv_mean_diameter', 'smv_mean_diameter'],
        'confluence_murray3_deviation': ['confluence_murray3_ratio'],
        'confluence_area_ratio': ['mpv_mean_area', 'sv_mean_area', 'smv_mean_area'],
        'mpv_bifurc_murray3_ratio': [
            'mpv_mean_diameter', 'lpv_mean_diameter', 'rpv_mean_diameter'],
        'mpv_bifurc_murray3_deviation': ['mpv_bifurc_murray3_ratio'],
        'mpv_bifurc_area_ratio': ['mpv_mean_area', 'lpv_mean_area', 'rpv_mean_area'],
        'lpv_rpv_diameter_asymmetry': ['lpv_mean_diameter', 'rpv_mean_diameter'],
        'lgv_mpv_diameter_ratio': ['lgv_mean_diameter', 'mpv_mean_diameter'],
        'pgv_mpv_diameter_ratio': ['pgv_mean_diameter', 'mpv_mean_diameter'],
        'splenic_dominance_index': ['sv_mean_diameter', 'smv_mean_diameter'],
        'collateral_length_mpv_ratio': ['mpv_length'],
        'diameter_weighted_tortuosity': [],
        'mpv_resistance_integral': ['mpv_length'],
        'sv_resistance_integral': ['sv_length'],
        'smv_resistance_integral': ['smv_length'],
        'lpv_resistance_integral': ['lpv_length'],
        'rpv_resistance_integral': ['rpv_length'],
        'tips_resistance_integral': ['tips_length'],
        'inflow_parallel_resistance': [
            'sv_resistance_integral', 'smv_resistance_integral'],
        'inflow_resistance_asymmetry': [
            'sv_resistance_integral', 'smv_resistance_integral'],
        'mpv_effective_radius': ['mpv_length', 'mpv_resistance_integral'],
        'tips_inflow_resistance_ratio': [
            'tips_resistance_integral', 'inflow_parallel_resistance'],
        'collateral_burden_score': ['mpv_length', 'mpv_mean_diameter'],
        'branchpoint_density_per_cm': ['mpv_length'],
        'mpv_taper_coefficient': ['mpv_length'],
        'mpv_proximal_diameter': ['mpv_length'],
        'mpv_distal_diameter': ['mpv_length'],
        'mpv_min_max_diameter_ratio': ['mpv_length'],
        'tree_area_conservation_mean_dev': [],
        'sv_max_to_mpv_max_diam_ratio': ['sv_max_diameter', 'mpv_max_diameter'],
        'mpv_trunk_length_mm': ['mpv_length'],
        'area_conservation_bifurc_deviation': [
            'mpv_mean_area', 'lpv_mean_area', 'rpv_mean_area'],
        'tips_stent_diameter_mm': ['tips_mean_diameter'],
        'tips_stent_length_mm': ['tips_length'],
        'min_lumen_area_to_max_ratio_mpv': [],
    }

    angle_deps = {
        'angle_sv_smv': ['sv', 'smv'],
        'angle_mpv_lpv': ['mpv', 'lpv'],
        'angle_mpv_rpv': ['mpv', 'rpv'],
        'angle_lpv_rpv': ['lpv', 'rpv'],
        'angle_mpv_bifurc_total': ['mpv', 'lpv', 'rpv'],
        'mpv_bifurc_planarity_deg': ['mpv', 'lpv', 'rpv'],
        'angle_mpv_tips': ['mpv', 'tips'],
    }
    if name in angle_deps:
        missing = [s.upper() for s in angle_deps[name]
                   if seg_dict.get(s) is None]
        if missing:
            return f"依赖血管段缺失: {', '.join(missing)}。"
        return "依赖段存在, 但方向向量退化或分叉几何无法稳定拟合。"

    if name == 'sv_smv_angle':
        if angle_detail is None:
            return ("SV-SMV 夹角无法计算: SV/SMV 段缺失、起点不是同一汇合点, "
                    "或方向向量退化。")

    if name.startswith(('mpv_', 'sv_', 'smv_', 'lpv_', 'rpv_', 'tips_')):
        seg = name.split('_', 1)[0]
        if seg in ALL_SEG_NAMES and seg_dict.get(seg) is None:
            return f"{seg.upper()} 段缺失, 因此该段派生特征无法计算。"

    deps = deps_by_feature.get(name, [])
    missing_deps = _deps_missing(deps, flat)
    if missing_deps:
        return "依赖特征为 None: " + ', '.join(missing_deps) + "。"

    if 'resistance' in name or name in {
            'mpv_effective_radius', 'min_lumen_area_to_max_ratio_mpv',
            'pvt_severity_grade'}:
        if pointwise_data is None:
            return "需要 centerline_pointwise_profiles.json, 但剖面数据缺失。"
        if name.startswith('mpv') and not pointwise_data.get('mpv'):
            return "需要 MPV pointwise 剖面, 但该剖面缺失或为 None。"
        return "剖面有效点不足、半径非正或积分分母退化。"

    if name == 'tree_area_conservation_mean_dev':
        return "汇合或分叉面积守恒所需的 MPV/SV/SMV 或 MPV/LPV/RPV 平均面积不完整。"
    if name == 'diameter_weighted_tortuosity':
        return "可用段少于 2 条, 或所有段缺少平均直径/曲折度。"
    if name == 'collateral_length_mpv_ratio':
        return "没有检测到 LGV/PGV 侧支, 或 MPV 长度缺失。"
    if name == 'cavernous_transformation_flag':
        return "海绵样变标志未能计算, 请检查 MPV 最大直径和分叉点密度。"

    if branch_points is None:
        return "分叉点列表缺失。"
    return "依赖条件不足或几何退化, 具体依赖请检查同名输入段/pointwise 剖面。"


def _system_group_for_feature(name):
    for group_name, names in SYSTEM_FEATURE_GROUPS.items():
        if name in names:
            return group_name
    return None


def _dependency_vessels_from_spec(spec):
    vessels = []
    vessels.extend(spec.get('required_vessels', []))
    vessels.extend(spec.get('required_any_vessels', []))
    vessels.extend(spec.get('optional_vessels', []))
    for vessel_set in spec.get('required_vessel_sets_any', []):
        vessels.extend(vessel_set)
    min_present = spec.get('required_min_present_vessels')
    if min_present:
        vessels.extend(min_present.get('vessels', []))
    return sorted(set(vessels), key=ALL_SEG_NAMES.index)


def _missing_vessels_for_feature(name, seg_dict):
    spec = SYSTEM_FEATURE_DEPENDENCIES.get(name, {})
    missing = [
        v for v in spec.get('required_vessels', [])
        if seg_dict.get(v) is None
    ]

    any_vessels = spec.get('required_any_vessels', [])
    if any_vessels and not any(seg_dict.get(v) is not None
                              for v in any_vessels):
        missing.extend(any_vessels)

    vessel_sets = spec.get('required_vessel_sets_any', [])
    if vessel_sets and not any(
            all(seg_dict.get(v) is not None for v in vessel_set)
            for vessel_set in vessel_sets):
        for vessel_set in vessel_sets:
            missing.extend([v for v in vessel_set if seg_dict.get(v) is None])

    min_present = spec.get('required_min_present_vessels')
    if min_present:
        vessels = min_present.get('vessels', [])
        min_count = int(min_present.get('min_count', 1))
        present_count = sum(1 for v in vessels if seg_dict.get(v) is not None)
        if present_count < min_count:
            missing.extend([v for v in vessels if seg_dict.get(v) is None])

    return sorted(set(missing), key=ALL_SEG_NAMES.index)


def _pointwise_missing_segments_for_feature(name, seg_dict, pointwise_data):
    spec = SYSTEM_FEATURE_DEPENDENCIES.get(name, {})
    if not spec.get('requires_pointwise'):
        return []

    target_vessels = spec.get('required_vessels', [])
    if not target_vessels and spec.get('required_vessel_sets_any'):
        target_vessels = _dependency_vessels_from_spec(spec)

    target_vessels = [
        v for v in target_vessels
        if v in ALL_SEG_NAMES and seg_dict.get(v) is not None
    ]
    if pointwise_data is None:
        return target_vessels
    return [v for v in target_vessels if pointwise_data.get(v) is None]


def _system_unavailable_detail(name, flat, seg_dict, pointwise_data,
                               branch_points, angle_detail):
    spec = SYSTEM_FEATURE_DEPENDENCIES.get(name, {})
    missing_vessels = _missing_vessels_for_feature(name, seg_dict)
    missing_pointwise = _pointwise_missing_segments_for_feature(
        name, seg_dict, pointwise_data)
    source_features = spec.get('source_features', [])
    missing_source_features = _deps_missing(source_features, flat)

    if missing_vessels:
        unavailable_due_to = 'vessel_absent'
        reason_category = 'vessel_absent'
    elif missing_pointwise:
        unavailable_due_to = 'extraction_failed'
        reason_category = 'pointwise_missing_or_failed'
    elif missing_source_features:
        unavailable_due_to = 'extraction_failed'
        reason_category = 'source_feature_missing'
    else:
        unavailable_due_to = 'extraction_failed'
        reason_category = 'geometry_or_quality_failed'

    return {
        'value': None,
        'unavailable_due_to': unavailable_due_to,
        'reason_category': reason_category,
        'reason': _system_missing_reason(
            name, flat, seg_dict, pointwise_data, branch_points,
            angle_detail),
        'dependent_vessels': _dependency_vessels_from_spec(spec),
        'missing_vessels': missing_vessels,
        'requires_pointwise': bool(spec.get('requires_pointwise')),
        'missing_pointwise_segments': missing_pointwise,
        'source_features': source_features,
        'missing_source_features': missing_source_features,
    }


def _split_system_features(system_values, flat, seg_data, pointwise_data,
                           angle_detail):
    seg_dict = seg_data.get('segments') or {}
    branch_points = seg_data.get('branch_points')
    available = {}
    unavailable = {}

    for name in SYSTEM_FEATURE_NAMES:
        value = system_values.get(name)
        if value is None:
            unavailable[name] = _system_unavailable_detail(
                name, flat, seg_dict, pointwise_data, branch_points,
                angle_detail)
        else:
            available[name] = value

    return {
        'available': available,
        'unavailable': unavailable,
        'all_values': system_values,
    }


def _build_vessel_presence(seg_data, pointwise_data, statistical):
    seg_dict = seg_data.get('segments') or {}
    out = {}
    for seg_name in ALL_SEG_NAMES:
        seg_info = seg_dict.get(seg_name)
        profile = pointwise_data.get(seg_name) if pointwise_data else None
        present = seg_info is not None

        if not present:
            pointwise_status = 'segment_absent'
        elif pointwise_data is None:
            pointwise_status = 'pointwise_file_missing_or_unreadable'
        elif profile is None:
            pointwise_status = 'pointwise_profile_missing_or_failed'
        else:
            pointwise_status = 'available'

        out[seg_name] = {
            'present': bool(present),
            'has_statistical_features': bool(seg_name in statistical),
            'has_pointwise_profile': bool(profile is not None),
            'pointwise_status': pointwise_status,
            'pointwise_diag': _pointwise_diag(profile),
        }
    return out


def build_feature_description():
    """生成独立的特征说明 JSON, 不含任何患者样本值。"""
    statistical_features = {
        key: {
            'depends_on_vessels': '<segment>',
            'flat_key_pattern': f'<segment>_{key}',
            'missing_rule': (
                '如果该 segment 血管不存在, 对应统计特征不可计算; '
                '如果血管存在但截面/中心线提取失败, 值也可能为 None。'
            ),
        }
        for key in PER_SEG_FEATURE_KEYS
    }

    system_features = {}
    for name in SYSTEM_FEATURE_NAMES:
        spec = SYSTEM_FEATURE_DEPENDENCIES.get(name, {})
        system_features[name] = {
            'label_cn': SYSTEM_FEATURE_LABELS_CN.get(name, name),
            'group': _system_group_for_feature(name),
            'depends_on_vessels': _dependency_vessels_from_spec(spec),
            'required_vessels': spec.get('required_vessels', []),
            'required_any_vessels': spec.get('required_any_vessels', []),
            'required_vessel_sets_any': spec.get(
                'required_vessel_sets_any', []),
            'required_min_present_vessels': spec.get(
                'required_min_present_vessels'),
            'optional_vessels': spec.get('optional_vessels', []),
            'requires_pointwise': bool(spec.get('requires_pointwise')),
            'source_features': spec.get('source_features', []),
        }

    return {
        '_schema_version': UNIFIED_SCHEMA_VERSION,
        'description': (
            '特征说明文件。这里记录字段含义、分组、依赖血管和缺失规则; '
            '患者样本的具体数值保存在 unified_features.json。'
        ),
        'vessels': {
            'names': ALL_SEG_NAMES,
            'labels_cn': {
                'mpv': '门静脉主干',
                'sv': '脾静脉',
                'smv': '肠系膜上静脉',
                'lpv': '左门静脉',
                'rpv': '右门静脉',
                'tips': 'TIPS 支架/分流道',
                'lgv': '胃左静脉侧支',
                'pgv': '胃后静脉侧支',
            },
        },
        'missing_value_policy': {
            'vessel_absent': (
                '样本没有检测到该特征依赖的血管, 该特征按解剖/分割结构不可计算。'
            ),
            'extraction_failed': (
                '依赖血管存在, 但中心线、截面、pointwise 或几何拟合质量不足。'
            ),
            'pointwise_missing_or_failed': (
                '特征需要 centerline_pointwise_profiles.json 中的逐点剖面, '
                '但文件或对应血管剖面缺失。'
            ),
            'source_feature_missing': (
                '上游统计特征为 None, 因此下游 system 特征无法计算。'
            ),
        },
        'statistical': {
            'description': '每段血管的 9 个标量统计特征。',
            'segments': ALL_SEG_NAMES,
            'feature_keys': PER_SEG_FEATURE_KEYS,
            'features': statistical_features,
        },
        'system': {
            'description': '跨血管系统特征, 每个字段列出依赖血管。',
            'feature_names': SYSTEM_FEATURE_NAMES,
            'groups': SYSTEM_FEATURE_GROUPS,
            'labels_cn': SYSTEM_FEATURE_LABELS_CN,
            'features': system_features,
        },
        'global': {
            'description': '样本/树级全局特征与血管存在性标志。',
            'feature_keys': GLOBAL_FEATURE_KEYS,
            'dependencies': {
                'total_centerline_length': [],
                'sv_smv_diameter_ratio': ['sv', 'smv'],
                'sv_smv_angle': ['sv', 'smv'],
                'has_lgv': ['lgv'],
                'has_pgv': ['pgv'],
                'has_compensation_vessel': ['lgv', 'pgv'],
                'has_tips': ['tips'],
            },
        },
        'pointwise': {
            'description': (
                '逐点剖面, 每个 segment 依赖同名血管存在且 extract_profiles 成功。'
            ),
            'segments': ALL_SEG_NAMES,
            'feature_keys': [
                'position', 'arc_length_mm', 'total_length_mm',
                'area', 'eq_diameter', 'perimeter',
                'raw_area', 'raw_eq_diameter', 'raw_perimeter',
                'anchor_radius', 'owned_radius', 'hydraulic_diameter',
                'circularity', 'solidity', 'r_insc_to_r_eq_ratio',
                'n_components', 'junction_replaced', 'curvature',
                'torsion', 'dA_ds_norm', 'inscribed_radius',
                'edge_margin_pct', 'edge_margin_mm', 'n_masked_endpoints',
                'n_junction_protected', 'n_junction_replaced',
                'n_rejected_oversize', 'n_section_success',
            ],
        },
    }


def _build_missing_report(flat, statistical, system, global_block,
                          pointwise_block, pointwise_data, seg_data,
                          angle_detail):
    """生成 unified_features.json 的 None 诊断块。"""
    seg_dict = seg_data.get('segments') or {}
    branch_points = seg_data.get('branch_points')

    report = {
        'summary': {},
        'statistical': {},
        'system': {},
        'global': {},
        'sv_smv_angle': None,
        'pointwise': {},
    }

    for seg_name in ALL_SEG_NAMES:
        seg_info = seg_dict.get(seg_name)
        profile = pointwise_data.get(seg_name) if pointwise_data else None
        block = statistical.get(seg_name)
        if block is None:
            report['statistical'][seg_name] = {
                '_segment': _seg_missing_reason(
                    seg_name, seg_info, profile, 'length', pointwise_data),
                '_pointwise_diag': _pointwise_diag(profile),
            }
            continue
        missing = {}
        for key, value in block.items():
            if value is None:
                missing[key] = {
                    'reason': _seg_missing_reason(
                        seg_name, seg_info, profile, key, pointwise_data),
                    'pointwise_diag': _pointwise_diag(profile),
                }
        if missing:
            report['statistical'][seg_name] = missing

    for key, value in system.items():
        if value is None:
            report['system'][key] = _system_missing_reason(
                key, flat, seg_dict, pointwise_data, branch_points,
                angle_detail)

    for key, value in global_block.items():
        if value is None:
            report['global'][key] = _system_missing_reason(
                key, flat, seg_dict, pointwise_data, branch_points,
                angle_detail)

    if angle_detail is None:
        report['sv_smv_angle'] = _system_missing_reason(
            'sv_smv_angle', flat, seg_dict, pointwise_data, branch_points,
            angle_detail)

    for seg_name in ALL_SEG_NAMES:
        if seg_name not in pointwise_block:
            seg_info = seg_dict.get(seg_name)
            profile = pointwise_data.get(seg_name) if pointwise_data else None
            if seg_info is None:
                reason = (f"{seg_name.upper()} 段未识别或解剖上不存在, "
                          "因此没有 pointwise 剖面。")
            elif pointwise_data is None:
                reason = "centerline_pointwise_profiles.json 缺失或解析失败。"
            elif profile is None:
                reason = "该段 pointwise 剖面为 None, extract_profiles 该段失败。"
            else:
                reason = "该段 pointwise 剖面未写入 unified, 请检查 JSON 结构。"
            report['pointwise'][seg_name] = reason

    n_stat_missing = sum(
        len(v) if isinstance(v, dict) else 1
        for v in report['statistical'].values())
    report['summary'] = {
        'statistical_missing_items': int(n_stat_missing),
        'system_missing_items': int(len(report['system'])),
        'global_missing_items': int(len(report['global'])),
        'pointwise_missing_segments': int(len(report['pointwise'])),
        'note': '此块只解释 None/缺失来源, 不改变原始特征字段。'
    }
    return report


def build_unified_features(flat_features, pointwise_data, seg_data,
                            angle_detail=None):
    """
    把各路输出聚合为单一 JSON 结构, 便于训练时一次加载.

    顶层结构:
        _meta            病人 ID / TIPS / 代偿
        _feature_description_file  独立特征说明 JSON 文件名
        vessel_presence 当前样本的血管存在性与 pointwise 状态
        statistical      每段 9 个标量特征 {seg: {key: value}}
        system           系统 / 联合特征, 拆成 available / unavailable / all_values
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
    system_values = {k: flat.get(k) for k in SYSTEM_FEATURE_NAMES}
    system = _split_system_features(
        system_values, flat, seg_data, pointwise_data, angle_detail)

    # ---- global ----
    global_block = {k: flat.get(k) for k in GLOBAL_FEATURE_KEYS}

    # ---- pointwise (剥掉 _meta 单独处理, 内部有 inscribed_radius 等) ----
    pointwise_block = {}
    pointwise_meta = {}
    if pointwise_data is not None:
        for k, v in pointwise_data.items():
            if k == '_meta':
                pointwise_meta = v
            elif v is not None:
                cleaned_profile = _clean_pointwise_profile_for_unified(v)
                if cleaned_profile is not None:
                    pointwise_block[k] = cleaned_profile

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

    vessel_presence = _build_vessel_presence(
        seg_data, pointwise_data, statistical)

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
                'F_clinical': [
                    'sv_max_to_mpv_max_diam_ratio',
                    'mpv_trunk_length_mm',
                    'max_tortuosity_index', 'mean_tortuosity_index',
                    'max_collateral_diameter_mm',
                    'area_conservation_bifurc_deviation',
                    'tips_stent_diameter_mm', 'tips_stent_length_mm',
                    'pvt_severity_grade',
                    'min_lumen_area_to_max_ratio_mpv',
                    'cavernous_transformation_flag'],
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
            'description': '逐点剖面 (重采样到 n_points), 真实末端 NaN 掩码; 交叉区用可信最小截面替换/封顶',
            'segments': list(pointwise_block.keys()),
            'feature_keys': ['position', 'arc_length_mm', 'total_length_mm',
                             'area', 'eq_diameter', 'perimeter',
                             'raw_area', 'raw_eq_diameter', 'raw_perimeter',
                             'anchor_radius', 'owned_radius',
                             'hydraulic_diameter',       # 4A/P, 非圆截面用
                             'circularity',
                             'solidity',                  # A / 凸包面积 (PVT)
                             'r_insc_to_r_eq_ratio',      # 瓶颈程度
                             'n_components',              # lumen 分量数
                             'junction_replaced',         # 1=交叉区已替换/封顶
                             'curvature',
                             'torsion',                   # 中心线 3D 扭转
                             'dA_ds_norm',                # 局部锥度
                             'inscribed_radius',
                             'edge_margin_pct', 'edge_margin_mm',
                             'n_masked_endpoints',
                             'n_junction_protected', 'n_junction_replaced',
                             'n_rejected_oversize',
                             'n_section_success'],
            'channel_labels_cn': {
                'area': '清洗/归属截面面积 mm²',
                'eq_diameter': '清洗/归属等效直径 mm',
                'perimeter': '清洗/归属截面周长 mm',
                'raw_area': '原始STL切面面积 mm²',
                'raw_eq_diameter': '原始STL切面等效直径 mm',
                'raw_perimeter': '原始STL切面周长 mm',
                'anchor_radius': '中心线锚定内切半径 mm',
                'owned_radius': '归属限制圆半径 mm',
                'hydraulic_diameter': '水力直径 4A/P (任意形状)',
                'circularity': '圆度 4πA/P²',
                'solidity': '凸包实心度 (PVT指标)',
                'r_insc_to_r_eq_ratio': '内切/等效半径比 (瓶颈)',
                'n_components': 'lumen 连通分量数',
                'junction_replaced': '交叉区替换/封顶标记',
                'curvature': '曲率 1/mm',
                'torsion': '挠率 1/mm (NaN 可信度)',
                'dA_ds_norm': '面积归一化变化率 (1/mm)',
                'inscribed_radius': '内切球半径 mm',
            },
            'mask_explanation': (
                '真实血管末端保护带内仍为 NaN; 分叉/交叉点保护带内不再丢弃, '
                '默认用该段非交叉可信区域的最小 clean area 对应截面替换, '
                '并用 junction_replaced=1 标记。raw_* 保留原始STL切面, '
                'area/eq_diameter/perimeter 用于统计与训练。'
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
        '_feature_description_file': FEATURE_DESCRIPTION_FILENAME,
        'vessel_presence': vessel_presence,
        '_missing': _build_missing_report(
            flat, statistical, system_values, global_block, pointwise_block,
            pointwise_data, seg_data, angle_detail),
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
