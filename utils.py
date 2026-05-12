"""
公共工具模块
============
所有模块共享的树加载、节点分类、路径查找等功能。
坐标系统：所有坐标均为物理坐标(mm)，无需spacing转换。
"""

import os
import numpy as np
from collections import defaultdict, deque


# ============================================================
# STL 体素化
# ============================================================

def voxelize_stl(stl_path, pitch=0.5):
    """
    将STL网格体素化为三维二值数组。

    参数:
        stl_path: str, STL文件路径
        pitch: float, 体素尺寸(mm)，各向同性

    返回:
        binary: ndarray (ni, nj, nk), uint8 二值数组
        origin: ndarray (3,), 体素网格原点的物理坐标
        pitch: float, 体素尺寸
    """
    import trimesh

    mesh = trimesh.load(stl_path)
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_grid = voxel_grid.fill()  # 填充内部

    binary = voxel_grid.matrix.astype(np.uint8)
    origin = voxel_grid.transform[:3, 3].copy()

    print(f"  体素化: pitch={pitch}mm, 尺寸={binary.shape}, "
          f"前景体素={np.sum(binary)}")

    return binary, origin, pitch


def voxel_to_physical(indices, origin, pitch):
    """体素索引(i,j,k) → 物理坐标(mm)"""
    return origin + np.asarray(indices, dtype=float) * pitch


def physical_to_voxel(coords, origin, pitch):
    """物理坐标(mm) → 体素索引(i,j,k)"""
    return np.round((np.asarray(coords, dtype=float) - origin) / pitch).astype(int)


# ============================================================
# 中心线树 加载/保存
# ============================================================

def _load_centerline_file(filepath):
    """从指定路径加载中心线树，返回 (nodes, adj)"""
    nodes = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            while len(parts) < 7:
                parts.append('-1')
            nid = int(parts[0])
            nodes[nid] = {
                'id': nid,
                'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3]),
                'parent': int(parts[4]), 'left': int(parts[5]), 'right': int(parts[6])
            }

    adj = defaultdict(set)
    for nid, node in nodes.items():
        for nb in [node['parent'], node['left'], node['right']]:
            if nb >= 0 and nb in nodes:
                adj[nid].add(nb)
                adj[nb].add(nid)

    print(f"  加载中心线: {os.path.basename(filepath)} ({len(nodes)}节点)")
    return nodes, adj


def load_raw_tree(stl_path):
    """
    加载原始中心线（CenterlinePoints.txt）。
    用于 smooth_centerline，确保读取的是提取阶段的输出，
    而不是上一轮平滑的结果。

    返回: (nodes, adj, parentdir)
    """
    parentdir = os.path.dirname(stl_path)
    filepath = os.path.join(parentdir, "CenterlinePoints.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到: {filepath}")
    nodes, adj = _load_centerline_file(filepath)
    return nodes, adj, parentdir


def load_tree(stl_path):
    """
    加载中心线树，优先读平滑后的 newCenterlist.txt。
    用于特征提取、可视化等下游模块。

    返回: (nodes, adj, parentdir)
    """
    parentdir = os.path.dirname(stl_path)

    for name in ["newCenterlist.txt", "CenterlinePoints.txt"]:
        filepath = os.path.join(parentdir, name)
        if os.path.exists(filepath):
            nodes, adj = _load_centerline_file(filepath)
            return nodes, adj, parentdir

    raise FileNotFoundError(
        f"找不到中心线文件: {parentdir}/newCenterlist.txt 或 CenterlinePoints.txt")


def save_tree(tree, output_path):
    """保存中心线树到文件"""
    with open(output_path, 'w') as f:
        for row in tree:
            f.write(' '.join(str(v) for v in row) + '\n')


# ============================================================
# 节点分类与路径
# ============================================================

def classify_nodes(nodes, adj):
    """分类节点为端点和分支点"""
    endpoints = set()
    branch_points = set()
    for nid in nodes:
        deg = len(adj[nid])
        if deg == 1:
            endpoints.add(nid)
        elif deg >= 3:
            branch_points.add(nid)
    return endpoints, branch_points


def find_path(adj, start, end):
    """BFS寻找两点之间的路径"""
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for nb in adj[current]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def path_to_coords(path, nodes):
    """路径节点ID列表 → 物理坐标数组 (N, 3)"""
    coords = []
    for nid in path:
        n = nodes[nid]
        coords.append([n['x'], n['y'], n['z']])
    return np.array(coords)


def node_distance(n1, n2):
    """两个节点之间的欧几里得距离(mm)，坐标已是物理坐标"""
    return np.sqrt(
        (n1['x'] - n2['x']) ** 2 +
        (n1['y'] - n2['y']) ** 2 +
        (n1['z'] - n2['z']) ** 2
    )


def path_physical_length(path, nodes):
    """路径的物理弧长(mm)"""
    if len(path) < 2:
        return 0.0
    coords = path_to_coords(path, nodes)
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))