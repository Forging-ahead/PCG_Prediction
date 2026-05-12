"""
中心线平滑模块（STL版）
========================
坐标已是物理坐标(mm)，无需spacing转换。

流程：
  1. 读取中心线树 → 邻接表 → 分类节点
  2. 提取所有"段"(关键点之间的路径)
  3. 逐段样条平滑
  4. 邻接表重组 → BFS建树 → 输出 newCenterlist.txt
"""

import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from collections import defaultdict, deque

from utils import load_raw_tree, classify_nodes, save_tree


def smooth_centerline(stl_path, output_txt_path=None,
                      smooth_factor=500, n_mult=3,
                      w_key=1e3, w_mid=10.0):
    """
    对中心线进行样条平滑。

    参数:
        stl_path: str, STL文件路径（用于定位中心线文件目录）
        output_txt_path: str or None, 输出路径，默认 newCenterlist.txt
        smooth_factor: float, 越大越平滑
        n_mult: int, 采样密度倍数
        w_key: float, 关键点（端点/分支点）权重
        w_mid: float, 普通点权重

    返回:
        new_tree: list of list [ID, x, y, z, parentID, leftChildID, rightChildID]
    """
    # ========== 第1步：读取 ==========
    print("[1/4] 读取原始中心线(CenterlinePoints.txt)...")
    nodes, adj, parentdir = load_raw_tree(stl_path)
    print(f"  节点数: {len(nodes)}")

    endpoints, branch_points = classify_nodes(nodes, adj)
    key_points = endpoints | branch_points
    print(f"  端点: {len(endpoints)}, 分支点: {len(branch_points)}")

    # ========== 第2步：提取所有段 ==========
    print("[2/4] 提取分支段...")
    segments = _extract_segments(nodes, adj, key_points)

    if len(segments) == 0 and len(nodes) > 0:
        # 退化情况：无关键点，整条线作为一段
        start_id = min(nodes.keys())
        segment = [start_id]
        visited = {start_id}
        current = start_id
        while True:
            next_nodes = [n for n in adj[current] if n not in visited]
            if not next_nodes:
                break
            current = next_nodes[0]
            segment.append(current)
            visited.add(current)
        segments.append(segment)

    print(f"  段数: {len(segments)}")

    # ========== 第3步：逐段平滑 ==========
    print("[3/4] 样条平滑...")
    smoothed_segments = []
    for seg_idx, segment in enumerate(segments):
        coords = [[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
                   for nid in segment]

        if len(coords) < 2:
            smoothed_segments.append(coords)
        elif len(coords) == 2:
            p0, p1 = np.array(coords[0]), np.array(coords[1])
            n_interp = max(3, n_mult * 2)
            interp = [tuple(p0 + t * (p1 - p0)) for t in np.linspace(0, 1, n_interp)]
            smoothed_segments.append(interp)
        else:
            try:
                smoothed = _fit_spline_segment(
                    coords, smooth_factor, n_mult, w_key, w_mid)
                smoothed_segments.append(smoothed)
            except Exception as e:
                print(f"    段{seg_idx}平滑失败({e})，保留原始")
                smoothed_segments.append([tuple(c) for c in coords])

    # ========== 第4步：重建树 ==========
    print("[4/4] 重建树...")
    new_tree = _rebuild_tree(
        segments, smoothed_segments, nodes, key_points, endpoints)

    if output_txt_path is None:
        output_txt_path = os.path.join(parentdir, "newCenterlist.txt")

    save_tree(new_tree, output_txt_path)

    n_ep = sum(1 for r in new_tree if r[5] == -1 and r[6] == -1)
    n_br = sum(1 for r in new_tree if r[5] != -1 and r[6] != -1)
    print(f"  新节点: {len(new_tree)}, 端点: {n_ep}, 分支点: {n_br}")
    print(f"  已保存: {output_txt_path}")

    return new_tree


# ============================================================
# 内部函数
# ============================================================

def _extract_segments(nodes, adj, key_points):
    """提取关键点之间的所有段"""
    segments = []
    visited_edges = set()

    for start in key_points:
        for neighbor in adj[start]:
            edge = (min(start, neighbor), max(start, neighbor))
            if edge in visited_edges:
                continue

            segment = [start]
            prev = start
            current = neighbor

            while current not in key_points:
                segment.append(current)
                next_nodes = [n for n in adj[current] if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]

            segment.append(current)

            for i in range(len(segment) - 1):
                e = (min(segment[i], segment[i + 1]), max(segment[i], segment[i + 1]))
                visited_edges.add(e)

            segments.append(segment)

    return segments


def _rebuild_tree(segments, smoothed_segments, nodes, key_points, endpoints):
    """从平滑后的段重建树结构"""
    new_adj = defaultdict(set)
    new_coords = {}
    new_id_counter = [0]
    key_new_id = {}

    def alloc_id():
        nid = new_id_counter[0]
        new_id_counter[0] += 1
        return nid

    # 为关键点分配ID
    for kp in key_points:
        n = nodes[kp]
        nid = alloc_id()
        key_new_id[kp] = nid
        new_coords[nid] = (n['x'], n['y'], n['z'])

    # 处理每段：插入中间点
    for seg_idx, segment in enumerate(segments):
        smoothed = smoothed_segments[seg_idx]
        start_new = key_new_id[segment[0]]
        end_new = key_new_id[segment[-1]]
        mid_coords = smoothed[1:-1]

        if len(mid_coords) == 0:
            new_adj[start_new].add(end_new)
            new_adj[end_new].add(start_new)
        else:
            prev_nid = start_new
            for coord in mid_coords:
                cur_nid = alloc_id()
                new_coords[cur_nid] = tuple(coord)
                new_adj[prev_nid].add(cur_nid)
                new_adj[cur_nid].add(prev_nid)
                prev_nid = cur_nid
            new_adj[prev_nid].add(end_new)
            new_adj[end_new].add(prev_nid)

    # BFS建树
    root_new = None
    for kp in endpoints:
        if kp in key_new_id:
            root_new = key_new_id[kp]
            break
    if root_new is None:
        root_new = 0

    visited = set()
    queue = deque([(root_new, -1)])
    visited.add(root_new)
    bfs_order = []

    while queue:
        nid, pid = queue.popleft()
        bfs_order.append((nid, pid))
        for nb in sorted(new_adj[nid]):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, nid))

    children_map = defaultdict(list)
    for nid, pid in bfs_order:
        if pid >= 0:
            children_map[pid].append(nid)

    old_to_final = {}
    for final_id, (nid, pid) in enumerate(bfs_order):
        old_to_final[nid] = final_id

    new_tree = []
    for final_id, (nid, pid) in enumerate(bfs_order):
        phys = new_coords[nid]
        parent_final = old_to_final[pid] if pid >= 0 else -1
        children = children_map.get(nid, [])
        lc = old_to_final[children[0]] if len(children) >= 1 else -1
        rc = old_to_final[children[1]] if len(children) >= 2 else -1
        new_tree.append([final_id, phys[0], phys[1], phys[2], parent_final, lc, rc])

    return new_tree


def _fit_spline_segment(coords, smooth_factor=500, n_mult=3,
                        w_key=1e3, w_mid=10.0):
    """对一段坐标做样条平滑"""
    pts = np.asarray(coords, float)
    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    t_pts = np.concatenate(([0], np.cumsum(seglen)))
    L = t_pts[-1]

    if L <= 0:
        return [tuple(p) for p in pts]

    weights = np.ones(len(pts)) * w_mid
    weights[0] = w_key
    weights[-1] = w_key

    M = len(t_pts)
    k = min(3, M - 1)

    sx = UnivariateSpline(t_pts, pts[:, 0], w=weights, k=k, s=smooth_factor)
    sy = UnivariateSpline(t_pts, pts[:, 1], w=weights, k=k, s=smooth_factor)
    sz = UnivariateSpline(t_pts, pts[:, 2], w=weights, k=k, s=smooth_factor)

    # 密采样后等弧长重采样
    dense = max(2000, M * 10)
    us = np.linspace(0, L, dense)
    curve = np.vstack((sx(us), sy(us), sz(us))).T

    dseg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.insert(np.cumsum(dseg), 0, 0.0)
    tot = cum[-1]
    if tot <= 0:
        return [tuple(p) for p in pts]

    N = max(len(pts) * n_mult, 3)
    target = np.linspace(0, tot, N)
    sampled = []
    idx = 0

    for td in target:
        while idx < len(cum) - 1 and cum[idx + 1] < td:
            idx += 1
        if idx >= len(cum) - 1:
            sampled.append(curve[-1])
        else:
            t0, t1 = cum[idx], cum[idx + 1]
            a = (td - t0) / (t1 - t0) if t1 > t0 else 0
            sampled.append(curve[idx] + a * (curve[idx + 1] - curve[idx]))

    sampled = [tuple(p) for p in sampled]
    sampled[0] = tuple(pts[0])
    sampled[-1] = tuple(pts[-1])
    return sampled


if __name__ == '__main__':
    import sys, time
    path = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    t0 = time.time()
    smooth_centerline(path)
    print(f"\n耗时: {time.time() - t0:.2f}s")