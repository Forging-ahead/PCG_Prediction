"""
中心线提取模块（STL输入版 v2 - 增强剪枝）
==========================================
流程:
  STL → 体素化 → 距离变换 → 3D骨架化(Lee94) → 图构建
      → 增强剪枝(物理长度+半径判据+近邻分支点合并)
      → BFS建树

新增剪枝策略:
  (-1) **硬阈值**: 末端分支极短 (< absolute_min_branch_length_mm), 或
       "短且极细" → 视为噪声毛刺. 极细但足够长的分支会被保留, 避免真实
       细血管被半径阈值误删.
  (0) **保护门 (新)**: 末端分支最大半径 / 父主干半径 ≥ keep_radius_ratio
       → 视为真实分支, 跳过所有长度/相对长度判据.
       目的: 避免短-但-粗 的真实小分支被长度阈值误剪
       (Murray 定律下子血管半径 ≈ 0.6~0.8 × 父血管, 远高于噪声毛刺 0.1~0.2)
       注: branch_max_radius 计算**必须排除 junction 节点**, 否则距离变换
       在分叉点处会把主干体积算进去, 使保护门对所有毛刺都误触发.
  (1) 物理长度阈值: 末端分支弧长 < min_branch_length_mm 且半径占比低时剪除
  (2) 相对长度阈值: 末端分支弧长 < total_length × min_relative_length 且半径占比低时剪除
  (3) 半径判据: 半径占比低只作为毛刺证据, 需结合短分支/相对短分支才剪除
  (4) 近邻 bp 合并: 两个 bp 距离 < merge_bp_distance_mm 时合并
  (5) 迭代到稳定

依赖: numpy, trimesh, scipy, scikit-image, networkx
"""

import os
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as skeletonize_3d
import networkx as nx
from collections import deque

from utils import voxelize_stl, voxel_to_physical, save_tree


def extract_centerline(stl_path, output_txt_path=None,
                       pitch=0.5,
                       min_branch_length_mm=10.0,
                       min_relative_length=0.05,
                       min_radius_ratio=0.4,
                       keep_radius_ratio=0.55,
                       absolute_min_branch_length_mm=3.0,
                       absolute_min_radius_mm=0.5,
                       merge_bp_distance_mm=6.0,
                       max_prune_iterations=20,
                       regularize_binary=True,
                       auto_refine_thin=True,
                       verbose=True):
    """
    从STL文件提取中心线 (含增强剪枝)。

    参数:
        stl_path:               STL 文件路径
        output_txt_path:        输出路径, 默认 CenterlinePoints.txt
        pitch:                  体素化分辨率(mm)
        min_branch_length_mm:   末端分支最小物理长度阈值(mm)
                                小于此值且半径占比低的末端分支被视为噪声剪除
        min_relative_length:    末端分支相对总长度的最小比例
                                小于此比例且半径占比低的末端分支也被剪除
        min_radius_ratio:       末端分支最大半径 / 主干半径 的最小比例
                                小于此比例时作为表面凸起证据之一
                                (区分真细血管 vs 表面凸起)
        keep_radius_ratio:      "保护门"阈值. 末端分支最大半径 / 主干半径
                                ≥ 该值时, 跳过所有长度判据直接保留.
                                目的: 保护短-但-粗 的真实分支 (常见于
                                LPV/RPV/LGV/PGV 末端). 默认 0.55:
                                noise spur 通常 < 0.3, 真分支多在 0.6+.
        absolute_min_branch_length_mm:
                                **硬剪阈值**, 默认 3.0mm. 末端分支弧长 < 此值
                                必为骨架毛刺 (1~5 体素的表面凸起), 直接剪除,
                                **跳过保护门**. 真血管分支不会短于 3mm.
        absolute_min_radius_mm: 极细毛刺参考阈值. 只在分支也很短时参与硬剪,
                                不再单独删除长的真实细血管。
        merge_bp_distance_mm:   两个分支点距离小于此值时合并
        max_prune_iterations:   剪枝最大迭代次数
        regularize_binary:      骨架化前做轻量体素正则化, 去掉单体素毛刺并补小孔。
        auto_refine_thin:       骨架为空/剪枝后为空时自动用更小 pitch 重试。
        verbose:                打印进度

    返回:
        centerline_tree: list of [ID, x, y, z, parentID, leftChildID, rightChildID]
    """
    last_error = None
    attempt_pitches = _centerline_pitch_attempts(pitch, auto_refine_thin)

    for attempt_idx, attempt_pitch in enumerate(attempt_pitches):
        retry_note = "" if attempt_idx == 0 else " (细血管兜底重试)"
        try:
            centerline_tree = _extract_centerline_once(
                stl_path,
                output_txt_path=output_txt_path,
                pitch=attempt_pitch,
                min_branch_length_mm=min_branch_length_mm,
                min_relative_length=min_relative_length,
                min_radius_ratio=min_radius_ratio,
                keep_radius_ratio=keep_radius_ratio,
                absolute_min_branch_length_mm=absolute_min_branch_length_mm,
                absolute_min_radius_mm=absolute_min_radius_mm,
                merge_bp_distance_mm=merge_bp_distance_mm,
                max_prune_iterations=max_prune_iterations,
                regularize_binary=regularize_binary,
                verbose=verbose,
                retry_note=retry_note)
            return centerline_tree
        except ValueError as e:
            last_error = e
            if (not auto_refine_thin or
                    attempt_idx == len(attempt_pitches) - 1 or
                    not _is_retryable_centerline_error(e)):
                raise
            if verbose:
                print(f"       [warn] {e}; 改用 pitch={attempt_pitches[attempt_idx+1]:.3f}mm 重试")

    if last_error is not None:
        raise last_error
    raise ValueError("中心线提取失败")


def _extract_centerline_once(stl_path, output_txt_path=None,
                             pitch=0.5,
                             min_branch_length_mm=10.0,
                             min_relative_length=0.05,
                             min_radius_ratio=0.4,
                             keep_radius_ratio=0.55,
                             absolute_min_branch_length_mm=3.0,
                             absolute_min_radius_mm=0.5,
                             merge_bp_distance_mm=6.0,
                             max_prune_iterations=20,
                             regularize_binary=True,
                             verbose=True,
                             retry_note=""):
    """执行一次中心线提取；外层负责失败重试。"""
    if verbose:
        print(f"\n[1/6] 体素化STL: {os.path.basename(stl_path)}{retry_note}")
    binary, origin, pitch = voxelize_stl(stl_path, pitch)
    if np.sum(binary) == 0:
        raise ValueError("体素化结果为空")

    if regularize_binary:
        binary = _regularize_binary_for_skeleton(binary, verbose=verbose)

    if verbose:
        print("[2/6] 距离变换...")
    dist_map = ndimage.distance_transform_edt(binary, sampling=pitch)

    if verbose:
        print("[3/6] 3D骨架化 (Lee94)...")
    skeleton = skeletonize_3d(binary).astype(np.uint8)
    skel_points = np.argwhere(skeleton > 0)
    if verbose:
        print(f"       骨架点: {len(skel_points)}")
    if len(skel_points) == 0:
        raise ValueError("骨架化结果为空")

    if verbose:
        print("[4/6] 构建骨架图...")
    G, id_to_point, id_to_radius = _build_skeleton_graph(
        skel_points, dist_map, pitch)

    components = list(nx.connected_components(G))
    if len(components) > 1:
        if verbose:
            print(f"       {len(components)} 个连通分量, 保留最大的")
        largest_cc = max(components, key=len)
        G = G.subgraph(largest_cc).copy()
    if verbose:
        print(f"       图节点数: {G.number_of_nodes()}")
    if G.number_of_nodes() == 0:
        raise ValueError("骨架图为空")

    if verbose:
        print(f"[5/6] 增强剪枝 (min_L={min_branch_length_mm}mm, "
              f"radius_ratio={min_radius_ratio})...")
    G = _enhanced_prune(
        G, id_to_point, id_to_radius, pitch,
        min_branch_length_mm=min_branch_length_mm,
        min_relative_length=min_relative_length,
        min_radius_ratio=min_radius_ratio,
        keep_radius_ratio=keep_radius_ratio,
        absolute_min_branch_length_mm=absolute_min_branch_length_mm,
        absolute_min_radius_mm=absolute_min_radius_mm,
        merge_bp_distance_mm=merge_bp_distance_mm,
        max_iterations=max_prune_iterations,
        verbose=verbose)

    if verbose:
        print(f"       剪枝后节点数: {G.number_of_nodes()}")
        eps = sum(1 for n in G.nodes() if G.degree(n) == 1)
        bps = sum(1 for n in G.nodes() if G.degree(n) >= 3)
        print(f"       端点: {eps}, 分支点: {bps}")
    if G.number_of_nodes() == 0:
        raise ValueError("剪枝后中心线为空")

    # ----- 第6步: BFS 建树, 输出物理坐标 -----
    if verbose:
        print("[6/6] BFS建树...")
    centerline_tree = _build_tree_from_graph(
        G, id_to_point, dist_map, origin, pitch)

    if output_txt_path is None:
        parentdir = os.path.dirname(stl_path)
        output_txt_path = os.path.join(parentdir, "CenterlinePoints.txt")
    save_tree(centerline_tree, output_txt_path)

    # ---- 缓存一致性保护 ----
    # 下游 utils.load_tree() 默认优先读 newCenterlist.txt. 如果用户:
    #   1) 调了剪枝参数, 2) 重跑 Step 1, 3) 没重跑 Step 2 (clean_old=False
    #   或 smooth_centerline 关闭), 旧的 newCenterlist.txt 会"屏蔽"新结果,
    #   导致下游/可视化与新参数完全脱节. 这里主动把过期的平滑文件干掉,
    #   保证 load_tree() 必然落到刚写的 CenterlinePoints.txt 上.
    stale_smooth = os.path.join(os.path.dirname(output_txt_path),
                                 "newCenterlist.txt")
    if os.path.exists(stale_smooth):
        try:
            os.remove(stale_smooth)
            if verbose:
                print(f"       已清理过期平滑文件: {stale_smooth} "
                      f"(避免下游读到旧中心线)")
        except Exception as e:
            if verbose:
                print(f"       [warn] 无法删除 {stale_smooth}: {e}")

    n_ep = sum(1 for r in centerline_tree if r[5] == -1 and r[6] == -1)
    n_br = sum(1 for r in centerline_tree if r[5] != -1 and r[6] != -1)
    if verbose:
        print(f"       总点数: {len(centerline_tree)}, 端点: {n_ep}, "
              f"分支点: {n_br}")
        print(f"       已保存: {output_txt_path}")

    return centerline_tree


def _centerline_pitch_attempts(pitch, auto_refine_thin):
    """生成中心线提取的 pitch 尝试序列。"""
    base = float(pitch)
    attempts = [base]
    if auto_refine_thin:
        for factor in (0.75, 0.5):
            p = round(base * factor, 4)
            if p > 0 and all(abs(p - old) > 1e-6 for old in attempts):
                attempts.append(p)
    return attempts


def _is_retryable_centerline_error(error):
    """判断是否值得用更小 pitch 重试。"""
    msg = str(error)
    retry_markers = (
        "体素化结果为空",
        "骨架化结果为空",
        "骨架图为空",
        "剪枝后中心线为空",
    )
    return any(marker in msg for marker in retry_markers)


def _regularize_binary_for_skeleton(binary, verbose=True):
    """
    骨架化前的轻量体素正则化。

    先闭运算/补孔, 降低单体素缺口对拓扑的影响; 再尝试一次 6 邻域开运算
    去掉表面小凸起。若开运算会破坏太多体素, 说明模型本身偏细, 自动退回到
    只闭运算的结果, 避免把真实细血管抹掉。
    """
    binary_bool = binary.astype(bool)
    original_count = int(np.sum(binary_bool))
    if original_count == 0:
        return binary

    structure = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(binary_bool, structure=structure, iterations=1)
    closed = ndimage.binary_fill_holes(closed)

    opened = ndimage.binary_opening(closed, structure=structure, iterations=1)
    closed_count = int(np.sum(closed))
    opened_count = int(np.sum(opened))
    retained = opened_count / max(closed_count, 1)

    if opened_count > 0 and retained >= 0.85:
        result = opened
        action = "闭运算+保守开运算"
    else:
        result = closed
        action = "闭运算"

    if verbose:
        print(f"       体素正则化: {action}, "
              f"{original_count} -> {int(np.sum(result))} 体素")
    return result.astype(np.uint8)


# ============================================================
# 骨架图构建（带半径信息）
# ============================================================

def _build_skeleton_graph(skel_points, dist_map, pitch):
    """
    构建 26 邻域骨架图。
    每个节点附带 radius (来自距离变换, mm)。
    """
    skel_set = set(map(tuple, skel_points))
    point_to_id = {}
    id_to_point = {}
    id_to_radius = {}
    for idx, pt in enumerate(skel_points):
        key = tuple(pt)
        point_to_id[key] = idx
        id_to_point[idx] = key
        id_to_radius[idx] = float(dist_map[pt[0], pt[1], pt[2]])

    offsets = [(di, dj, dk)
               for di in [-1, 0, 1] for dj in [-1, 0, 1] for dk in [-1, 0, 1]
               if not (di == 0 and dj == 0 and dk == 0)]

    G = nx.Graph()
    G.add_nodes_from(range(len(skel_points)))
    for idx, pt in enumerate(skel_points):
        i, j, k = pt
        for di, dj, dk in offsets:
            neighbor = (i + di, j + dj, k + dk)
            if neighbor in skel_set:
                nid = point_to_id[neighbor]
                if nid > idx:
                    w = pitch * np.sqrt(di**2 + dj**2 + dk**2)
                    G.add_edge(idx, nid, weight=w)

    return G, id_to_point, id_to_radius


# ============================================================
# 增强剪枝
# ============================================================

def _enhanced_prune(G, id_to_point, id_to_radius, pitch,
                    min_branch_length_mm,
                    min_relative_length,
                    min_radius_ratio,
                    keep_radius_ratio,
                    absolute_min_branch_length_mm,
                    absolute_min_radius_mm,
                    merge_bp_distance_mm,
                    max_iterations,
                    verbose):
    """
    迭代式增强剪枝。

    每轮:
      a) 找所有末端分支(从端点沿度=2 节点走到下一个度!=2 节点)
      b) **硬阈值**: 极短或短且极细分支剪除, 跳过保护门
      c) **保护门**: 若分支最大半径 / 主干半径 ≥ keep_radius_ratio,
                     视为真实分支, 跳过该轮所有剪枝判据
      d) 剪掉满足短分支 + 低半径占比判据的分支
      e) 合并近邻分支点
    直到稳定或达到最大迭代。
    """
    n_pruned_total = 0
    n_merged_total = 0
    n_kept_thick_total = 0

    # 先估计总长度作为相对长度参考
    total_edge_length = sum(d['weight'] for _, _, d in G.edges(data=True))

    for iteration in range(max_iterations):
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        if not endpoints:
            break

        n_pruned_this_round = 0
        n_kept_thick_this_round = 0
        prune_logs = []

        for ep in list(endpoints):
            if ep not in G or G.degree(ep) != 1:
                continue  # 已被前面剪掉

            branch_path, junction = _trace_branch_from_endpoint(G, ep)
            if branch_path is None or junction is None:
                continue
            if G.degree(junction) < 3:
                continue  # 不是真分支点, 不剪

            # 物理弧长
            branch_length = _path_arc_length(G, branch_path)

            # 末端分支最大半径
            # *关键*: 必须排除 junction 节点 — 距离变换在分叉点处会"看到"
            # 主干体积, 给 junction 一个 = 主干半径的大值. 若把 junction
            # 算进 max, 任何毛刺都会得到 ratio=1.0, 保护门对所有毛刺误触发.
            spur_only_nodes = [n for n in branch_path if n != junction]
            if not spur_only_nodes:
                continue
            branch_radii = np.asarray(
                [id_to_radius[n] for n in spur_only_nodes], dtype=float)
            # Ignore the junction-side tail when estimating branch thickness:
            # those voxels often inherit the main trunk radius and can protect
            # skeletonization spurs that are otherwise thin.
            core_count = max(1, int(np.ceil(0.8 * len(branch_radii))))
            branch_core_radii = branch_radii[:core_count]
            branch_max_radius = float(np.max(branch_core_radii))
            branch_median_radius = float(np.median(branch_core_radii))

            # 父主干半径(取分支点处的半径)
            junction_radius = id_to_radius[junction]

            radius_ratio = (branch_median_radius / junction_radius
                            if junction_radius > 1e-6 else 1.0)

            # ---- (-1) 硬阈值: 极短, 或"短且极细" → 噪声毛刺 ----
            # 旧逻辑只要半径低于 absolute_min_radius_mm 就强剪, 会把 LGV/PGV
            # 等真实细血管一起删掉。这里把半径阈值改为毛刺证据之一:
            # 细血管可以细, 但不应同时短到只像表面凸起。
            absolute_short = branch_length < absolute_min_branch_length_mm
            absolute_thin = branch_max_radius < absolute_min_radius_mm
            short_thin_spur = (
                absolute_thin and
                branch_length < max(min_branch_length_mm,
                                    2.0 * absolute_min_branch_length_mm)
            )
            radius_too_small = radius_ratio < min_radius_ratio

            # ---- (0) 保护门: 半径占比足够大 → 视为真实分支, 不剪 ----
            # 即使分支较短, 只要其管径相对于分叉处足够粗, 物理上不可能是
            # 表面噪声毛刺 (skeletonization spur 的 r_max 通常 ≤ 1~2 个体素),
            # 真分支应当保留以免后续解剖分段缺失.
            # 注: 硬阈值 (极短/短且极细) 优先, 不进保护门.
            if (not absolute_short and not short_thin_spur and
                    junction_radius > 1e-6 and
                    radius_ratio >= keep_radius_ratio):
                n_kept_thick_this_round += 1
                continue

            # ---- 三个判据 + 硬阈值 ----
            should_prune = False
            reason = ""

            if absolute_short:
                should_prune = True
                reason = (f"极短(L={branch_length:.2f}mm, "
                          f"硬阈值 {absolute_min_branch_length_mm}mm)")
            elif short_thin_spur:
                should_prune = True
                reason = (f"短且极细(L={branch_length:.1f}mm, "
                          f"r={branch_max_radius:.2f}mm)")
            elif branch_length < min_branch_length_mm and radius_too_small:
                should_prune = True
                reason = (f"短且细(L={branch_length:.1f}mm, "
                          f"ratio={radius_ratio:.2f})")
            elif (branch_length < total_edge_length * min_relative_length and
                  radius_too_small):
                should_prune = True
                reason = (f"相对短(L={branch_length:.1f}mm < "
                          f"{100*min_relative_length:.0f}%, "
                          f"ratio={radius_ratio:.2f})")

            if should_prune:
                # 剪除分支(不剪 junction 本身)
                for n in branch_path:
                    if n != junction and n in G:
                        G.remove_node(n)
                        n_pruned_this_round += 1
                if verbose and len(prune_logs) < 5:
                    prune_logs.append(reason)

        # 合并近邻分支点
        n_merged = _merge_nearby_branchpoints(
            G, id_to_point, id_to_radius, pitch,
            merge_bp_distance_mm)

        n_pruned_total += n_pruned_this_round
        n_merged_total += n_merged
        n_kept_thick_total += n_kept_thick_this_round

        if verbose and (n_pruned_this_round > 0 or n_merged > 0):
            sample = ("; ".join(prune_logs[:3]) +
                      ("..." if len(prune_logs) > 3 else ""))
            kept_note = (f", 保护粗分支 {n_kept_thick_this_round}"
                         if n_kept_thick_this_round > 0 else "")
            print(f"       iter {iteration+1}: 剪除{n_pruned_this_round}点, "
                  f"合并bp{n_merged}个{kept_note}  例: {sample}")

        if n_pruned_this_round == 0 and n_merged == 0:
            break

    if verbose:
        print(f"       剪枝合计: 移除 {n_pruned_total} 点, "
              f"合并 {n_merged_total} 个 bp, "
              f"保护粗分支 {n_kept_thick_total} 次")
    return G


def _trace_branch_from_endpoint(G, endpoint):
    """
    从一个端点沿度=2 节点走, 直到遇到度!=2 节点(分支点或另一端点)。

    返回:
        path:     [endpoint, ..., last_deg2_node, junction]
                  (含 junction)
        junction: 分支点 ID
        若找不到合适路径, 返回 (None, None)
    """
    if G.degree(endpoint) != 1:
        return None, None

    path = [endpoint]
    prev = None
    current = endpoint
    visited = {endpoint}
    while True:
        neighbors = [n for n in G.neighbors(current) if n != prev]
        if len(neighbors) != 1:
            break
        nxt = neighbors[0]
        if nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        if G.degree(nxt) != 2:
            return path, nxt
        prev = current
        current = nxt

    return None, None


def _path_arc_length(G, path):
    """计算 path 的物理弧长, 使用图边权重。"""
    L = 0.0
    for i in range(len(path) - 1):
        if G.has_edge(path[i], path[i + 1]):
            L += G[path[i]][path[i + 1]]['weight']
    return L


def _merge_nearby_branchpoints(G, id_to_point, id_to_radius, pitch,
                                merge_distance_mm):
    """
    合并距离极近的两个分支点。

    若两个分支点 bp1, bp2 在图中通过 ≤ merge_distance_mm 的路径相连,
    且都是真 bp (度≥3), 把 bp2 的所有邻居改接到 bp1, 删除 bp2。
    """
    if merge_distance_mm <= 0:
        return 0

    n_merged = 0
    bps = [n for n in G.nodes() if G.degree(n) >= 3]
    bps_set = set(bps)

    for bp1 in list(bps):
        if bp1 not in G or G.degree(bp1) < 3:
            continue
        # 找 bp1 附近 (BFS, 按弧长 ≤ merge_distance_mm) 的另一个 bp
        merged_one = False
        visited = {bp1: 0.0}
        queue = deque([(bp1, 0.0)])
        while queue:
            cur, dist = queue.popleft()
            if dist > merge_distance_mm:
                continue
            if cur != bp1 and cur in bps_set and G.degree(cur) >= 3:
                # 合并 cur 到 bp1
                _merge_two_bps(G, bp1, cur)
                bps_set.discard(cur)
                n_merged += 1
                merged_one = True
                break
            for nb in G.neighbors(cur):
                w = G[cur][nb]['weight']
                if nb not in visited or dist + w < visited[nb]:
                    visited[nb] = dist + w
                    if dist + w <= merge_distance_mm:
                        queue.append((nb, dist + w))
        if merged_one:
            # 重新检查 bp1, 可能还有更多近邻 bp
            continue

    return n_merged


def _merge_two_bps(G, keep, drop):
    """
    把 drop 的所有邻居重接到 keep, 删除 drop 及 keep-drop 之间的桥接路径。

    实际操作:
      - 找 keep → drop 的最短路径 (理论上很短)
      - 删除路径上 keep 和 drop 之间所有中间节点 + drop
      - 把 drop 的非桥接邻居挂到 keep
    """
    if keep == drop or drop not in G or keep not in G:
        return

    # 找 keep -> drop 路径
    try:
        path = nx.shortest_path(G, keep, drop, weight='weight')
    except nx.NetworkXNoPath:
        return

    # drop 在原图的邻居 (非桥接路径上的)
    drop_neighbors = list(G.neighbors(drop))
    bridge_set = set(path)

    external_neighbors = [n for n in drop_neighbors if n not in bridge_set]

    # 把 external_neighbors 接到 keep (用一个直接边, 权重取 drop-nb 边的权重)
    for nb in external_neighbors:
        if not G.has_edge(keep, nb):
            w = G[drop][nb]['weight']
            G.add_edge(keep, nb, weight=w)

    # 删除桥接路径上 keep 和 drop 之间的所有中间节点 + drop
    for n in path[1:]:  # path[0] = keep
        if n in G:
            G.remove_node(n)


# ============================================================
# BFS 建树
# ============================================================

def _build_tree_from_graph(G, id_to_point, dist_map, origin, pitch):
    """从最终图 BFS 建有根树。根选距离变换最大的端点。"""
    remaining = list(G.nodes())
    endpoints = [n for n in remaining if G.degree(n) == 1]

    if endpoints:
        root = max(endpoints, key=lambda ep: dist_map[id_to_point[ep]])
    else:
        root = max(remaining, key=lambda n: dist_map[id_to_point[n]])

    visited = set()
    queue = deque([(root, -1)])
    visited.add(root)
    bfs_order = []
    old_to_new = {}
    counter = 0
    while queue:
        node, parent_old = queue.popleft()
        bfs_order.append((node, parent_old))
        old_to_new[node] = counter
        counter += 1
        for nb in G.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, node))

    children_map = {old_id: [] for old_id, _ in bfs_order}
    for old_id, parent_old in bfs_order:
        if parent_old != -1 and parent_old in children_map:
            children_map[parent_old].append(old_id)

    tree = []
    for old_id, parent_old in bfs_order:
        new_id = old_to_new[old_id]
        vi, vj, vk = id_to_point[old_id]
        phys = voxel_to_physical([vi, vj, vk], origin, pitch)
        px, py, pz = float(phys[0]), float(phys[1]), float(phys[2])
        parent_new = old_to_new[parent_old] if parent_old != -1 else -1
        children = children_map[old_id]
        lc = old_to_new[children[0]] if len(children) >= 1 else -1
        rc = old_to_new[children[1]] if len(children) >= 2 else -1
        if len(children) > 2:
            print(f"       警告: 节点 {new_id} 有 {len(children)} 个孩子, "
                  f"保留前 2 个")
        tree.append([new_id, px, py, pz, parent_new, lc, rc])

    return tree


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    extract_centerline(path)
