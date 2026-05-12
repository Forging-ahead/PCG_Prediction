"""
血管分段可视化导出模块
======================
为每个患者生成两个可视化文件:
  vis_overview.png       — 多角度静态拼图
  vis_interactive.html   — Plotly 交互式 3D 可视化 (浏览器打开)

成功样本: 完整渲染 (STL + 分段中心线 + 最大截面圈 + 标签 + 图例)
失败样本: 降级渲染 (STL + 原始中心线, 红色 "分段失败" 标题)

新增 (vs v1):
  - 每条血管段的最大截面位置可视化:
    * 实线彩色环 = STL 真实截面轮廓
    * 虚线圆 = 等效圆 (radius = √(A/π))
    * 中心点标记 + 悬停显示面积/直径
  - 段信息表格附加 Amax 数值

依赖:
  pip install plotly trimesh numpy pillow kaleido
"""

import os
import json
import numpy as np
import trimesh

import plotly.graph_objects as go


# ============================================================
# 配色 (与 visualize_segments.py 保持一致)
# ============================================================

SEGMENT_COLORS = {
    'mpv':  '#ff3333',  'sv':   '#3380ff',  'smv':  '#ff9933',
    'lpv':  '#b34dff',  'rpv':  '#33e666',  'tips': '#00e6e6',
    'lgv':  '#ffe633',  'pgv':  '#ff4dee',
}

SEGMENT_LABELS = {
    'mpv': 'MPV', 'sv': 'SV', 'smv': 'SMV',
    'lpv': 'LPV', 'rpv': 'RPV', 'tips': 'TIPS',
    'lgv': 'LGV', 'pgv': 'PGV',
}


# ============================================================
# 数据加载
# ============================================================

def _load_segments_json(parentdir):
    """加载分段 JSON, 不存在返回 None。"""
    path = os.path.join(parentdir, "centerline_profiles.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_pointwise_profiles(parentdir):
    """加载逐点剖面 JSON, 不存在返回 None。"""
    path = os.path.join(parentdir, "centerline_pointwise_profiles.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_centerline_txt(parentdir, prefer_smooth=True):
    """加载中心线 txt, 返回 nodes dict。"""
    if prefer_smooth:
        candidates = ["newCenterlist.txt", "CenterlinePoints.txt"]
    else:
        candidates = ["CenterlinePoints.txt", "newCenterlist.txt"]

    for name in candidates:
        path = os.path.join(parentdir, name)
        if os.path.exists(path):
            nodes = {}
            try:
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 4:
                            continue
                        while len(parts) < 7:
                            parts.append('-1')
                        nid = int(parts[0])
                        nodes[nid] = {
                            'id': nid,
                            'x': float(parts[1]), 'y': float(parts[2]),
                            'z': float(parts[3]),
                            'parent': int(parts[4]),
                            'left': int(parts[5]),
                            'right': int(parts[6])
                        }
                return nodes
            except Exception:
                continue
    return None


def _load_stl_mesh(stl_path):
    """加载 STL, 返回 trimesh.Trimesh 或 None。"""
    try:
        mesh = trimesh.load(stl_path)
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'geometry'):
                mesh = list(mesh.geometry.values())[0]
            else:
                return None
        return mesh
    except Exception as e:
        print(f"  STL 加载失败: {e}")
        return None


def _build_full_centerline_segments(nodes):
    """
    从 nodes 构造所有 (parent, child) 边的坐标列表, 用于画灰色参考中心线。
    每段后插入 None 实现断线。
    """
    xs, ys, zs = [], [], []
    for nid, n in nodes.items():
        for child in [n['left'], n['right']]:
            if child >= 0 and child in nodes:
                c = nodes[child]
                xs.extend([n['x'], c['x'], None])
                ys.extend([n['y'], c['y'], None])
                zs.extend([n['z'], c['z'], None])
    return xs, ys, zs


# ============================================================
# Plotly 基础 Trace 构建 (STL / 中心线 / 分段 / 标签 / 分支点)
# ============================================================

def _build_mesh_trace(vertices, faces, opacity=0.2):
    """STL 半透明网格 trace。"""
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=opacity,
        color='#c0c0d8',
        name='Vessel mesh',
        showlegend=True,
        legendgroup='mesh',
        flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1),
        hoverinfo='skip',
    )


def _build_centerline_trace(nodes):
    """整条中心线 (灰色细线), 默认隐藏。"""
    xs, ys, zs = _build_full_centerline_segments(nodes)
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color='#888888', width=2),
        name='Centerline (raw)',
        legendgroup='centerline',
        visible='legendonly',
        hoverinfo='skip',
    )


def _build_segment_trace(seg_name, seg_info, nodes):
    """单条解剖段 trace (粗彩色线)。"""
    color = SEGMENT_COLORS.get(seg_name, '#888888')
    label = SEGMENT_LABELS.get(seg_name, seg_name.upper())
    coords = np.array([
        [nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
        for nid in seg_info['path']
    ])

    L = seg_info.get('length_mm', 0)
    tort = seg_info.get('tortuosity', 0)
    hover = (f"<b>{label}</b><br>"
             f"长度: {L:.1f} mm<br>"
             f"曲折度: {tort:.3f}<br>"
             f"点数: {seg_info.get('n_points', len(coords))}<br>"
             f"<extra></extra>")

    return go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='lines',
        line=dict(color=color, width=8),
        name=f"{label} ({L:.1f}mm)",
        legendgroup=f'seg_{seg_name}',
        hovertemplate=hover,
    )


def _build_label_trace(seg_name, seg_info, nodes):
    """段名 3D 文字标签 (放在段中点上方)。"""
    coords = np.array([
        [nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
        for nid in seg_info['path']
    ])
    mid = coords[len(coords) // 2]
    label = SEGMENT_LABELS.get(seg_name, seg_name.upper())
    color = SEGMENT_COLORS.get(seg_name, '#888888')

    return go.Scatter3d(
        x=[mid[0] + 1.5], y=[mid[1] + 1.5], z=[mid[2] + 1.5],
        mode='text',
        text=[f"<b>{label}</b>"],
        textfont=dict(size=16, color=color),
        name=f"{label} label",
        legendgroup=f'seg_{seg_name}',
        showlegend=False,
        hoverinfo='skip',
    )


def _build_branch_point_trace(branch_points):
    """所有分支点 (默认隐藏)。"""
    if not branch_points:
        return None
    xs = [bp['coord'][0] for bp in branch_points]
    ys = [bp['coord'][1] for bp in branch_points]
    zs = [bp['coord'][2] for bp in branch_points]
    ids = [bp['id'] for bp in branch_points]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=6, color='#222222', symbol='circle',
                    line=dict(color='white', width=1)),
        name='Branch points',
        legendgroup='bps',
        visible='legendonly',
        text=[f"BP id={i}" for i in ids],
        hovertemplate='%{text}<extra></extra>',
    )


# ============================================================
# 最大截面可视化
# ============================================================
def _find_max_section(profile, edge_margin=0.05):
    """
    在剖面 100 点里找最大面积位置 (跳过 NaN)。

    NaN 来自 _apply_endpoint_mask 标记的端点保护区。
    edge_margin 这层是 _find_max_section 自己的"显示用"端点保护,
    与 extract_profiles 里的 edge_margin_pct 是独立的两层防护。

    返回 dict 或 None
    """
    if profile is None or 'area' not in profile:
        return None
    areas = np.array(profile['area'], dtype=float)

    # 用 nanmax 跳过端点掩码的 NaN
    valid_mask = np.isfinite(areas) & (areas > 0)
    if not np.any(valid_mask):
        return None

    n = len(areas)
    lo = int(n * edge_margin)
    hi = int(n * (1 - edge_margin))

    middle = areas[lo:hi].copy()
    middle_valid = np.isfinite(middle) & (middle > 0)

    if np.any(middle_valid):
        # 中间区间找峰值, 用 nanargmax 跳过 NaN
        # 先把无效位置设为 -inf 避免被选中
        masked_middle = np.where(middle_valid, middle, -np.inf)
        idx = int(np.argmax(masked_middle)) + lo
    else:
        # 中间无有效值, 整段找
        masked_all = np.where(valid_mask, areas, -np.inf)
        idx = int(np.argmax(masked_all))

    return {
        'pos_index': idx,
        'area': float(areas[idx]),
        'eq_diameter': float(profile['eq_diameter'][idx]),
        'perimeter': float(profile['perimeter'][idx])
                     if 'perimeter' in profile else 0.0,
        'position_pct': idx,
    }
def _interp_centerline_at_pos(seg_path, nodes, pos_idx, n_total=100):
    """
    在分段路径上按归一化位置 (pos_idx / (n_total-1)) 取 3D 坐标和切线。

    seg_path:  list of node id (来自 centerline_profiles.json segments[name].path)
    pos_idx:   0-99
    返回:
        (point_3d ndarray(3,), tangent_unit ndarray(3,)) 或 (None, None)
    """
    if len(seg_path) < 2:
        return None, None

    coords = np.array([
        [nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
        for nid in seg_path
    ])

    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(diffs)))
    total = arc[-1]
    if total < 1e-6:
        return None, None

    target_arc = total * pos_idx / (n_total - 1)

    seg_idx = np.searchsorted(arc, target_arc) - 1
    seg_idx = max(0, min(len(coords) - 2, seg_idx))

    a0, a1 = arc[seg_idx], arc[seg_idx + 1]
    t = (target_arc - a0) / (a1 - a0) if a1 > a0 else 0.0

    point = coords[seg_idx] + t * (coords[seg_idx + 1] - coords[seg_idx])

    # 切线: 用相邻几个点的差分
    lo = max(0, seg_idx - 1)
    hi = min(len(coords) - 1, seg_idx + 2)
    tangent = coords[hi] - coords[lo]
    norm = np.linalg.norm(tangent)
    tangent = tangent / norm if norm > 1e-6 else np.array([0, 0, 1])

    return point, tangent


def _compute_real_cross_section_ring(stl_mesh, point, normal,
                                      target_area=None,
                                      max_eq_diameter=None,
                                      max_aspect_ratio=4.0,
                                      min_circularity=0.30):
    """
    计算 STL 在给定点 + 法线下的真实截面闭合 3D 轮廓.

    与 extract_profiles._compute_cross_section 形状感知评分完全一致:
      1. 扰动法线 (n_perturb=12, ±15°)
      2. 形状硬过滤: aspect_ratio > max_aspect_ratio 或 circularity < min_circularity 剔除
      3. 综合评分: area × elongation_pen × irregularity_pen, 选最小者
      这样视觉上画出的环 与 portal_vein_features.json 中的面积来源同一候选.

    若给定 target_area (训练时记录的面积), 用它做最终一致性兜底:
      在通过形状过滤的候选里, 选 area 最接近 target_area 的那个.
      若所有候选都被过滤掉, 退回到全局 score 最小者 (比 target 不一致好过没图).

    参数:
        max_eq_diameter:    等效直径上限 (用于过滤穿透到邻近血管的截面)
        target_area:        期望面积 (mm²); 给定时作为最终一致性兜底
        max_aspect_ratio:   形状硬过滤阈值 (与 _compute_cross_section 同步)
        min_circularity:    形状硬过滤阈值

    返回:
        ring_3d: (N, 3) ndarray, 闭合 (首尾相连) 或 None
    """
    if stl_mesh is None:
        return None

    try:
        from extract_profiles import (_make_orthonormal_basis,
                                       _section_one,
                                       _generate_normal_candidates,
                                       _shape_score)

        normal = np.asarray(normal, dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-15)
        candidates = _generate_normal_candidates(normal,
                                                  n_perturb=12,
                                                  max_angle_deg=15)

        # 收集所有通过形状过滤的候选
        passes = []   # list of (score, area, ring_2d, normal)
        all_valid = []  # 兜底: 有面积的所有候选 (即使形状不达标)
        for n in candidates:
            a, p, ar, circ, ring_2d = _section_one(
                stl_mesh, point, n,
                max_eq_diameter=max_eq_diameter,
                return_metrics=True,
                return_ring=True)
            if a <= 0 or ring_2d is None:
                continue
            score = _shape_score(a, ar, circ)
            all_valid.append((score, a, ring_2d, n))
            if ar > max_aspect_ratio or circ < min_circularity:
                continue
            passes.append((score, a, ring_2d, n))

        pool = passes if passes else all_valid
        if not pool:
            return None

        if target_area is not None and target_area > 0:
            chosen = min(pool, key=lambda x: abs(x[1] - target_area))
        else:
            chosen = min(pool, key=lambda x: x[0])

        _, area_used, ring_2d, n_used = chosen
        u_use, v_use = _make_orthonormal_basis(n_used)
        ring_3d = np.array([
            point + x2 * u_use + y2 * v_use for (x2, y2) in ring_2d])
        return ring_3d
    except Exception as e:
        print(f"    [warn] cross-section ring 计算失败: {e}")
        return None


def _build_equivalent_circle_3d(point, normal, radius, n_pts=64):
    """构造 3D 等效圆 (用虚线显示)。"""
    from extract_profiles import _make_orthonormal_basis
    u, v = _make_orthonormal_basis(normal)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.array([
        point + radius * (np.cos(t) * u + np.sin(t) * v)
        for t in theta
    ])
    return circle


def _build_max_section_traces(seg_data, pointwise_profiles, nodes, stl_mesh):
    """
    为每条段构造最大截面相关的 traces。

    返回 list of plotly traces:
        - 实线真实截面轮廓 (粗线, 同段色)
        - 虚线等效圆 (粗虚线, 同段色)
        - 中心点钻石标记
        - 文字标签 (单独 trace, 偏移避免重叠)
    """
    traces = []
    if seg_data is None or pointwise_profiles is None or nodes is None:
        return traces

    if stl_mesh is None:
        print("    [Warn] STL mesh 不可用, 跳过最大截面绘制")
        return traces

    print(f"    构建最大截面 traces:")

    for seg_name, seg_info in seg_data.get('segments', {}).items():
        if seg_info is None:
            continue
        profile = pointwise_profiles.get(seg_name)
        if profile is None:
            continue

        max_info = _find_max_section(profile)
        if max_info is None:
            print(f"      [{seg_name.upper()}] 无有效截面值, 跳过")
            continue

        point, tangent = _interp_centerline_at_pos(
            seg_info['path'], nodes, max_info['pos_index'], n_total=100)
        if point is None:
            continue

        color = SEGMENT_COLORS.get(seg_name, '#888888')
        label = SEGMENT_LABELS.get(seg_name, seg_name.upper())

        # ---- 真实截面 3D 轮廓 (粗实线) ----
        # 用与 extract_profiles 一致的扰动+内切上限策略, 并指定 target_area
        # 让选到的扰动法线对应的面积接近剖面记录的最大值, 这样圆环和数值匹配.
        local_r = None
        if 'inscribed_radius' in profile:
            try:
                ir_arr = np.asarray(profile['inscribed_radius'], dtype=float)
                idx_safe = max(0, min(len(ir_arr) - 1, max_info['pos_index']))
                v = float(ir_arr[idx_safe])
                if np.isfinite(v) and v > 0.5:
                    local_r = v
            except Exception:
                local_r = None
        max_eq_d = (1.8 * 2.0 * local_r) if local_r is not None else None
        ring_3d = _compute_real_cross_section_ring(
            stl_mesh, point, tangent,
            target_area=max_info['area'],
            max_eq_diameter=max_eq_d)
        ring_ok = ring_3d is not None and len(ring_3d) >= 3

        if ring_ok:
            traces.append(go.Scatter3d(
                x=ring_3d[:, 0], y=ring_3d[:, 1], z=ring_3d[:, 2],
                mode='lines',
                line=dict(color=color, width=10),  # 加粗
                name=f"{label} max-section (A={max_info['area']:.1f}mm²)",
                legendgroup=f'maxsec_{seg_name}',
                showlegend=True,
                hovertemplate=(
                    f"<b>{label} 最大截面 (真实)</b><br>"
                    f"位置: {max_info['position_pct']}%<br>"
                    f"面积: {max_info['area']:.2f} mm²<br>"
                    f"周长: {max_info['perimeter']:.2f} mm<br>"
                    f"等效直径: {max_info['eq_diameter']:.2f} mm"
                    "<extra></extra>"),
            ))
            print(f"      [{label}] @ {max_info['position_pct']}%, "
                  f"A={max_info['area']:.2f}mm², 真实轮廓 {len(ring_3d)} 点 ✓")
        else:
            print(f"      [{label}] @ {max_info['position_pct']}%, "
                  f"A={max_info['area']:.2f}mm², 真实轮廓计算失败 ✗")

        # ---- 等效圆 (粗虚线) ----
        eq_radius = max_info['eq_diameter'] / 2.0
        if eq_radius > 1e-6:
            eq_circle = _build_equivalent_circle_3d(
                point, tangent, eq_radius, n_pts=64)
            traces.append(go.Scatter3d(
                x=eq_circle[:, 0], y=eq_circle[:, 1], z=eq_circle[:, 2],
                mode='lines',
                line=dict(color=color, width=6, dash='dash'),  # 加粗
                name=f"{label} eq circle (r={eq_radius:.2f}mm)",
                legendgroup=f'maxsec_{seg_name}',
                showlegend=False,
                opacity=0.85,
                hovertemplate=(
                    f"<b>{label} 等效圆</b><br>"
                    f"半径: {eq_radius:.2f} mm<br>"
                    f"等效面积: {np.pi * eq_radius**2:.2f} mm²"
                    "<extra></extra>"),
            ))

        # ---- 中心钻石标记 (无文字, 文字单独画避免遮挡) ----
        traces.append(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='diamond',
                         line=dict(color='white', width=2)),
            name=f"{label} max marker",
            legendgroup=f'maxsec_{seg_name}',
            showlegend=False,
            hovertemplate=(
                f"<b>{label} 最大截面位置</b><br>"
                f"沿段位置: {max_info['position_pct']}%<br>"
                f"截面积: {max_info['area']:.2f} mm²<br>"
                f"等效直径: {max_info['eq_diameter']:.2f} mm<br>"
                f"坐标: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})"
                "<extra></extra>"),
        ))

        # ---- 文字标签: 以等效半径偏移, 避免与段名/钻石重叠 ----
        # 文字偏移方向: 沿切向法线之一, 距离 = 1.5 * 半径 + 5mm
        from extract_profiles import _make_orthonormal_basis
        u, v = _make_orthonormal_basis(tangent)
        offset_distance = max(eq_radius * 1.5 + 5, 8)
        text_pos = point + offset_distance * u

        traces.append(go.Scatter3d(
            x=[text_pos[0]], y=[text_pos[1]], z=[text_pos[2]],
            mode='text',
            text=[f"<b style='background-color:white;padding:2px'>"
                  f"{label}: A={max_info['area']:.1f}mm²</b>"],
            textposition='middle center',
            textfont=dict(size=12, color=color, family='Arial Black'),
            name=f"{label} max text",
            legendgroup=f'maxsec_{seg_name}',
            showlegend=False,
            hoverinfo='skip',
        ))

    return traces

# ============================================================
# 主函数: 构建 Plotly Figure
# ============================================================

def _build_figure(stl_path):
    """
    构建 Plotly figure。

    返回:
        fig: plotly Figure
        seg_data: 分段 JSON (失败时 None)
        is_success: 是否完整渲染
    """
    parentdir = os.path.dirname(stl_path)
    folder_name = os.path.basename(parentdir)

    seg_data = _load_segments_json(parentdir)
    nodes = _load_centerline_txt(parentdir, prefer_smooth=True)
    stl_mesh = _load_stl_mesh(stl_path)

    fig = go.Figure()
    is_success = (seg_data is not None and nodes is not None)

    # ---- STL 网格 ----
    if stl_mesh is not None:
        fig.add_trace(_build_mesh_trace(stl_mesh.vertices, stl_mesh.faces))

    # ---- 原始中心线 (灰色, 默认隐藏) ----
    if nodes is not None:
        fig.add_trace(_build_centerline_trace(nodes))

    # ---- 分段彩色中心线 + 段名标签 ----
    loaded_segs = []
    if is_success:
        for seg_name, seg_info in seg_data.get('segments', {}).items():
            if seg_info is None:
                continue
            fig.add_trace(_build_segment_trace(seg_name, seg_info, nodes))
            fig.add_trace(_build_label_trace(seg_name, seg_info, nodes))
            loaded_segs.append((seg_name, seg_info))

    # ---- 最大截面圈 (新增) ----
    pointwise = None
    if is_success:
        pointwise = _load_pointwise_profiles(parentdir)
        if pointwise is not None and stl_mesh is not None:
            max_section_traces = _build_max_section_traces(
                seg_data, pointwise, nodes, stl_mesh)
            for tr in max_section_traces:
                fig.add_trace(tr)

    # ---- 分支点 ----
    if is_success:
        bp_trace = _build_branch_point_trace(seg_data.get('branch_points', []))
        if bp_trace is not None:
            fig.add_trace(bp_trace)

    # ---- 标题 ----
    if is_success:
        title_main = f"Patient: {folder_name}"
        is_post = seg_data.get('is_post_tips', False)
        title_sub = (f"{'POST-TIPS' if is_post else 'PRE-TIPS'}"
                     f"{' | Compensation: '+seg_data['compensation_type'] if seg_data.get('has_compensation') else ''}")
        title_color = '#1e293b'
    else:
        title_main = f"⚠ {folder_name} —— 分段失败"
        if seg_data is None and nodes is None:
            title_sub = "中心线和分段文件均缺失"
        elif seg_data is None:
            title_sub = "分段失败 (centerline_profiles.json 缺失). 仅显示 STL 与原始中心线"
        else:
            title_sub = "渲染异常"
        title_color = '#dc2626'

    # ---- 段信息表格 (右上角, 含最大截面) ----
    info_lines = []
    if loaded_segs:
        info_lines.append(f"<b>Loaded segments ({len(loaded_segs)})</b>")
        for nm, info in loaded_segs:
            lb = SEGMENT_LABELS.get(nm, nm.upper())
            color = SEGMENT_COLORS.get(nm, '#888888')
            L = info.get('length_mm', 0)
            t = info.get('tortuosity', 0)
            line = (f"<span style='color:{color}'>● {lb}</span> "
                    f"L={L:.1f}mm  τ={t:.3f}")
            # 追加最大截面信息
            if pointwise and pointwise.get(nm):
                ms = _find_max_section(pointwise[nm])
                if ms:
                    line += (f"  Amax={ms['area']:.1f}mm² "
                             f"(@{ms['position_pct']}%)")
            info_lines.append(line)
    info_html = "<br>".join(info_lines) if info_lines else ""

    # ---- 布局 ----
    fig.update_layout(
        title=dict(
            text=f"<b style='color:{title_color}'>{title_main}</b>"
                 f"<br><span style='font-size:13px;color:#475569'>{title_sub}</span>",
            x=0.02, xanchor='left', y=0.97, yanchor='top',
        ),
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='#f8fafc',
                       gridcolor='#cbd5e1', showgrid=True),
            yaxis=dict(title='Y (mm)', backgroundcolor='#f8fafc',
                       gridcolor='#cbd5e1', showgrid=True),
            zaxis=dict(title='Z (mm)', backgroundcolor='#f8fafc',
                       gridcolor='#cbd5e1', showgrid=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0),
                        up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(
            x=0.01, y=0.85, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cbd5e1', borderwidth=1,
            font=dict(size=11),
            itemsizing='constant',
        ),
        annotations=[dict(
            text=info_html,
            xref='paper', yref='paper',
            x=0.99, y=0.97, xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cbd5e1', borderwidth=1,
            font=dict(family='Courier New, monospace', size=11),
            align='left',
        )] if info_html else [],
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor='white',
    )

    return fig, seg_data, is_success


# ============================================================
# 导出 PNG (多角度拼图)
# ============================================================

def _export_overview_png(fig, output_path, success=True):
    """
    导出 8 角度拼图 PNG。
    使用 plotly 的 to_image 在不同 camera 下截图, 然后用 PIL 拼接。
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        print("  PIL 未安装, 跳过 PNG 拼图 (pip install pillow)")
        return False

    cameras = [
        ('Front',     dict(x=0,    y=-2.5, z=0)),
        ('Back',      dict(x=0,    y=2.5,  z=0)),
        ('Left',      dict(x=-2.5, y=0,    z=0)),
        ('Right',     dict(x=2.5,  y=0,    z=0)),
        ('Top',       dict(x=0,    y=0,    z=2.5)),
        ('Bottom',    dict(x=0,    y=0,    z=-2.5)),
        ('Iso 1',     dict(x=1.5,  y=1.5,  z=1.0)),
        ('Iso 2',     dict(x=-1.5, y=-1.5, z=1.0)),
    ]

    sub_images = []
    sub_w, sub_h = 600, 500

    for view_name, eye in cameras:
        fig.update_layout(scene_camera=dict(
            eye=eye, up=dict(x=0, y=0, z=1)))
        try:
            img_bytes = fig.to_image(format='png', width=sub_w, height=sub_h,
                                     engine='kaleido')
            img = Image.open(io.BytesIO(img_bytes))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            draw.rectangle([5, 5, 100, 32], fill='white', outline='#cbd5e1')
            draw.text((10, 8), view_name, fill='#1e293b', font=font)
            sub_images.append(img)
        except Exception as e:
            placeholder = Image.new('RGB', (sub_w, sub_h), 'white')
            sub_images.append(placeholder)
            print(f"  视角 {view_name} 截图失败: {e}")

    n_cols, n_rows = 4, 2
    total_w = sub_w * n_cols
    total_h = sub_h * n_rows + 60

    canvas = Image.new('RGB', (total_w, total_h), 'white')

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font_big = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font_big = ImageFont.load_default()
        title_color = '#1e293b' if success else '#dc2626'
        prefix = "" if success else "⚠ FAILED  "
        draw.text((20, 15), f"{prefix}{os.path.basename(os.path.dirname(output_path))}",
                  fill=title_color, font=font_big)
    except Exception:
        pass

    for idx, img in enumerate(sub_images):
        r, c = idx // n_cols, idx % n_cols
        canvas.paste(img, (c * sub_w, r * sub_h + 60))

    canvas.save(output_path, format='PNG', optimize=True)
    return True


# ============================================================
# 对外主入口
# ============================================================

def export_patient_visualization(stl_path,
                                  export_html=True, export_png=True,
                                  verbose=True):
    """
    为单个患者导出可视化文件。

    输出:
        <patient_dir>/vis_interactive.html   (Plotly 交互, 含最大截面圈)
        <patient_dir>/vis_overview.png       (8 角度拼图)

    参数:
        export_html: 是否导出 HTML
        export_png:  是否导出 PNG (依赖 kaleido)

    返回:
        dict {'html': bool, 'png': bool, 'success': bool}
    """
    parentdir = os.path.dirname(stl_path)
    folder_name = os.path.basename(parentdir)
    result = {'html': False, 'png': False, 'success': False}

    if verbose:
        print(f"  [Vis Export] {folder_name}")

    try:
        fig, seg_data, is_success = _build_figure(stl_path)
        result['success'] = is_success
    except Exception as e:
        print(f"  ✗ 构建 figure 失败: {e}")
        import traceback
        traceback.print_exc()
        return result

    if export_html:
        html_path = os.path.join(parentdir, "vis_interactive.html")
        try:
            fig.write_html(
                html_path,
                include_plotlyjs='cdn',
                full_html=True,
                config={'displaylogo': False, 'responsive': True})
            result['html'] = True
            if verbose:
                size_mb = os.path.getsize(html_path) / 1024 / 1024
                print(f"    HTML: {html_path}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ HTML 导出失败: {e}")

    if export_png:
        png_path = os.path.join(parentdir, "vis_overview.png")
        try:
            ok = _export_overview_png(fig, png_path, success=is_success)
            result['png'] = ok
            if ok and verbose:
                size_mb = os.path.getsize(png_path) / 1024 / 1024
                print(f"    PNG:  {png_path}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ PNG 导出失败: {e}")
            print(f"     (确保已安装 kaleido: pip install kaleido)")

    return result


# ============================================================
# 批量入口 (独立调用时用)
# ============================================================

def batch_export(root_folder, stl_name="vessel.stl"):
    """对 root_folder 下所有有 STL 的文件夹批量导出可视化。"""
    print(f"\n{'='*60}")
    print(f"批量导出可视化: {root_folder}")
    print(f"{'='*60}")

    n_html, n_png, n_total = 0, 0, 0
    for folder in sorted(os.listdir(root_folder)):
        fp = os.path.join(root_folder, folder)
        stl = os.path.join(fp, stl_name)
        if not os.path.isdir(fp) or not os.path.exists(stl):
            continue
        n_total += 1
        result = export_patient_visualization(stl)
        if result['html']: n_html += 1
        if result['png']:  n_png  += 1

    print(f"\n完成: {n_total} 个样本, "
          f"HTML {n_html}, PNG {n_png}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        export_patient_visualization(sys.argv[1])
    else:
        batch_export(r"F:\PCG data\dataset\zhengzhou_vkan_qian47")