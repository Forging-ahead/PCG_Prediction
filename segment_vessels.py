"""
血管解剖分段模块（v4 - LPV/RPV 主导的 MPV 扩展）
================================================
基于中心线树拓扑（分支点/端点）+ 物理先验（长度/曲率/方向）
对门静脉系统进行解剖分段。

支持的解剖结构:
  MPV  - 门静脉主脉
  SV   - 脾静脉
  SMV  - 肠系膜上静脉
  LPV  - 肝门左静脉
  RPV  - 肝门右静脉
  TIPS - TIPS术后支架（仅术后）
  LGV  - 胃左静脉（术前代偿）
  PGV  - 胃后静脉（术前代偿）

判别准则:
  (1) MPV 初始: 两端均为分支点的段, 多条候选时按 L·exp(-2·τ) 选最长最直
  (2) SV 端 vs 肝侧端: 子树中 SV-score = L·(τ+0.01) 高者为 SV 端
  (3) SV / SMV: SV-score 最高 = SV (长且弯), 剩余 = SMV
  (4) TIPS: 肝侧子树全部端点段按 TIPS-score = L·exp(-2.5·τ) 评分,
            最高者 = TIPS (长且直, 长度主导)
  (5) LPV / RPV: 端点 X 坐标 (LPS 坐标系: X 大者 = LPV)
  (6) MPV 终点扩展: bp_mpv_end = LPV/RPV 起点中沿弧长距 SV 端最远的 bp
                  (TIPS 不参与, 因其为人工分流)
  (7) 段裁剪: 起点 == bp_mpv_end 时沿段找下一个 bp; 否则段不动
  (8) 术前 LGV vs PGV: 3 个 bp 排成链 bp1-bp2-bp3, 计算 bp1↔bp3 路径 τ
                       小 → LGV 代偿 (MPV 贯穿), 大 → PGV 代偿 (MPV = bp1→bp2)
  (9) PGV 代偿下 SV-distal vs PGV: 方向一致性 cos(SV入射, 候选出射)
                                   值大者 = SV-distal, 值小者 = PGV
      并做 PGV 合理性质控: 若 PGV 分叉点过近 SV-SMV 汇合且候选过短,
      降级为无代偿分段, 避免把汇合处短毛刺误标为 PGV.
"""

import os
import json
import numpy as np
from collections import deque

from utils import (
    load_tree, classify_nodes, find_path,
    path_to_coords, path_physical_length
)


# ============================================================
# 文件夹命名规则
# ============================================================

def is_invalid_folder(folder_name):
    """文件夹无效: 名称包含 @ 或 !"""
    return '@' in folder_name or '!' in folder_name


def is_post_tips(folder_name):
    """是否 TIPS 术后: 名称包含 #"""
    return '#' in folder_name


# ============================================================
# 几何工具：曲率 / 方向
# ============================================================

def _path_tortuosity(coords):
    """1 - 弦/弧长。0 = 笔直, 越大越弯曲。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return 0.0
    chord = np.linalg.norm(coords[-1] - coords[0])
    arclen = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    if arclen <= 1e-6:
        return 0.0
    return float(1.0 - chord / arclen)


def _path_mean_curvature(coords):
    """路径平均离散曲率 (1/mm)"""
    coords = np.asarray(coords)
    if len(coords) < 3:
        return 0.0
    ks = []
    for i in range(1, len(coords) - 1):
        v1, v2 = coords[i] - coords[i - 1], coords[i + 1] - coords[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        ks.append(np.arccos(cos_a) / (0.5 * (n1 + n2)))
    return float(np.mean(ks)) if ks else 0.0


def _direction_at_start(coords, sample_dist=8.0):
    """路径从首端出发的单位方向 (沿弧长走 sample_dist mm 取参考点)。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return None
    start = coords[0]
    cumlen = 0.0
    sample_pt = coords[-1]
    for i in range(1, len(coords)):
        cumlen += np.linalg.norm(coords[i] - coords[i - 1])
        if cumlen >= sample_dist:
            sample_pt = coords[i]
            break
    direction = sample_pt - start
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else None


def _direction_at_end(coords, sample_dist=8.0):
    """路径在末端的单位入射方向 (指向最后一点)。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return None
    end = coords[-1]
    cumlen = 0.0
    sample_pt = coords[0]
    for i in range(len(coords) - 1, 0, -1):
        cumlen += np.linalg.norm(coords[i] - coords[i - 1])
        if cumlen >= sample_dist:
            sample_pt = coords[i - 1]
            break
    direction = end - sample_pt
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else None


# ============================================================
# 段评分
# ============================================================

def _seg_score_sv(seg, nodes):
    """SV 评分: 长度 × (曲率 + ε)。SV 偏长且偏弯。"""
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    t = _path_tortuosity(coords)
    return L * (t + 0.01)


def _seg_score_tips(seg, nodes):
    """
    TIPS 评分: 长度主导, 曲率乘性衰减。
    score = L * exp(-2.5 * tortuosity)
    """
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    t = _path_tortuosity(coords)
    return L * np.exp(-2.5 * t)


def _mpv_init_score(seg, nodes):
    """MPV 初始候选评分: 长度主导, 曲率轻微衰减。"""
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    t = _path_tortuosity(coords)
    return L * np.exp(-2.0 * t)


# ============================================================
# 段提取
# ============================================================

def _extract_all_segments(nodes, adj, endpoints, branch_points):
    """提取所有"段"(关键点之间的路径)。"""
    key_points = endpoints | branch_points
    segments = []
    visited_edges = set()

    for start in key_points:
        for neighbor in adj[start]:
            edge = (min(start, neighbor), max(start, neighbor))
            if edge in visited_edges:
                continue

            seg = [start]
            prev = start
            current = neighbor
            while current not in key_points:
                seg.append(current)
                next_nodes = [n for n in adj[current] if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]
            seg.append(current)

            for i in range(len(seg) - 1):
                e = (min(seg[i], seg[i + 1]), max(seg[i], seg[i + 1]))
                visited_edges.add(e)
            segments.append(seg)

    return segments


def _find_endpoint_branches_at(segments_raw, bp, endpoints):
    """从分支点 bp 出发, 返回所有另一端是端点的段(统一: bp 在头, 端点在尾)。"""
    result = []
    for seg in segments_raw:
        if seg[0] == bp and seg[-1] in endpoints:
            result.append(list(seg))
        elif seg[-1] == bp and seg[0] in endpoints:
            result.append(list(seg[::-1]))
    return result


def _find_bp_to_bp_segments(segments_raw, branch_points):
    """返回所有两端均为分支点的段。"""
    return [seg for seg in segments_raw
            if seg[0] in branch_points and seg[-1] in branch_points]


# ============================================================
# 子树收集
# ============================================================

def _collect_subtree(adj, root_bp, exclude_neighbor, endpoints, branch_points):
    """
    从 root_bp 出发收集子树, 不回溯到 exclude_neighbor 方向。

    返回 dict:
        'root_branches':   直接从 root_bp 出去的端点分支 (root_bp 在头)
        'deeper_branches': 嵌套在子树深处的端点分支 (deeper_bp 在头)
        'all_branches':    上面两者合并
        'visited_bps':     子树中遇到的所有 bp
    """
    key_points = endpoints | branch_points
    root_branches = []
    deeper_branches = []
    visited_bps = {root_bp}
    visited_edges = set()

    queue = deque()
    for nb in adj[root_bp]:
        if nb == exclude_neighbor:
            continue
        queue.append((root_bp, nb))

    while queue:
        bp_start, first_nb = queue.popleft()
        edge = (min(bp_start, first_nb), max(bp_start, first_nb))
        if edge in visited_edges:
            continue
        visited_edges.add(edge)

        seg = [bp_start]
        prev = bp_start
        current = first_nb
        while current not in key_points:
            seg.append(current)
            next_nodes = [n for n in adj[current] if n != prev]
            if not next_nodes:
                break
            prev = current
            current = next_nodes[0]
        seg.append(current)

        for i in range(len(seg) - 1):
            e = (min(seg[i], seg[i + 1]), max(seg[i], seg[i + 1]))
            visited_edges.add(e)

        if seg[-1] in endpoints:
            if bp_start == root_bp:
                root_branches.append(seg)
            else:
                deeper_branches.append(seg)
        elif seg[-1] in branch_points:
            next_bp = seg[-1]
            if next_bp not in visited_bps:
                visited_bps.add(next_bp)
                for nb in adj[next_bp]:
                    if nb == seg[-2]:
                        continue
                    e2 = (min(next_bp, nb), max(next_bp, nb))
                    if e2 not in visited_edges:
                        queue.append((next_bp, nb))

    return {
        'root_branches': root_branches,
        'deeper_branches': deeper_branches,
        'all_branches': root_branches + deeper_branches,
        'visited_bps': visited_bps,
    }


# ============================================================
# SV / SMV、LPV / RPV 选择
# ============================================================

def _select_sv_smv(branches, nodes):
    """SV/SMV 端: SV-score 最高 = SV (长且弯), 剩余 = SMV。"""
    if not branches:
        return None, None
    if len(branches) == 1:
        return branches[0], None

    if len(branches) > 2:
        branches = sorted(branches,
                          key=lambda s: path_physical_length(s, nodes),
                          reverse=True)[:2]

    s0, s1 = branches[0], branches[1]
    if _seg_score_sv(s0, nodes) >= _seg_score_sv(s1, nodes):
        return s0, s1
    return s1, s0


def _assign_lpv_rpv(branches, nodes):
    """
    LPS 坐标系约定 (DICOM 默认): X 越大 → patient's left → LPV。
    若数据是 RAS 坐标系, 把 if x0 > x1 反一下即可。
    """
    if not branches:
        return None, None
    if len(branches) == 1:
        return branches[0], None

    if len(branches) > 2:
        branches = sorted(branches,
                          key=lambda s: path_physical_length(s, nodes),
                          reverse=True)[:2]

    s0, s1 = branches[0], branches[1]
    x0, x1 = nodes[s0[-1]]['x'], nodes[s1[-1]]['x']
    if x0 > x1:
        return s0, s1
    return s1, s0


def _select_sv_distal_pgv(branches, sv_main_path, nodes, sample_dist=8.0):
    """
    在 bp_svsub 处区分 SV-distal 和 PGV (方向一致性)。
    SV 是连续血管, SV-distal 出射方向延续 SV-proximal 入射方向;
    PGV 从 SV 上分支出去, 方向有偏转。
    """
    if not branches:
        return None, None
    if len(branches) == 1:
        return branches[0], None

    sv_main_coords = path_to_coords(sv_main_path, nodes)
    incoming_dir = _direction_at_end(sv_main_coords, sample_dist)
    if incoming_dir is None:
        return _select_sv_smv(branches, nodes)

    scored = []
    for br in branches:
        out_dir = _direction_at_start(path_to_coords(br, nodes), sample_dist)
        score = float(np.dot(incoming_dir, out_dir)) if out_dir is not None else -2.0
        scored.append((score, br))

    scored.sort(key=lambda x: x[0], reverse=True)
    sv_distal = scored[0][1]
    pgv = scored[1][1]
    print(f"    SV/PGV 方向一致性: SV={scored[0][0]:.3f}, PGV={scored[1][0]:.3f}")
    return sv_distal, pgv


def _pgv_candidate_quality(pgv_seg, sv_distal_seg, sv_proximal,
                           smv_seg, nodes):
    """
    判断 PGV 候选是否像真实代偿支。

    这个门控只用于防止把 SV-SMV 汇合点附近的短小骨架支误标为 PGV。
    真实 PGV 通常应从 SV 远端分出, 与汇合点有一定距离, 且长度不能只像
    一个局部表面/端点伪分支。
    """
    if pgv_seg is None or len(pgv_seg) < 2:
        return False, "无 PGV 候选"

    pgv_len = path_physical_length(pgv_seg, nodes)
    sv_distal_len = path_physical_length(sv_distal_seg, nodes) if sv_distal_seg else 0.0
    sv_prox_len = path_physical_length(sv_proximal, nodes) if sv_proximal else 0.0
    smv_len = path_physical_length(smv_seg, nodes) if smv_seg else 0.0

    # PGV 分叉点离 SV-SMV 汇合太近时, 很容易是中心线拓扑毛刺或短端点。
    near_confluence = sv_prox_len < 8.0

    # 长度门限用相对值兜底, 避免固定阈值误杀整体较小的样本。
    reference_len = max(sv_distal_len, smv_len, 1.0)
    short_vs_system = pgv_len < max(10.0, 0.20 * reference_len)

    # 若 PGV 比它竞争的 SV 远端短太多, 且起点就在汇合附近, 大概率不是
    # 真实代偿血管, 而是本例图中这种被误分出的短支。
    tiny_vs_sv = sv_distal_len > 1e-6 and pgv_len < 0.35 * sv_distal_len

    if near_confluence and (short_vs_system or tiny_vs_sv):
        return False, (
            f"PGV 起点离 SV-SMV 汇合仅 {sv_prox_len:.1f}mm, "
            f"且候选较短 {pgv_len:.1f}mm "
            f"(SV远端 {sv_distal_len:.1f}mm, SMV {smv_len:.1f}mm)"
        )

    # 极短支即使不在汇合点附近, 也更像骨架毛刺。
    if pgv_len < 6.0:
        return False, f"PGV 候选过短 {pgv_len:.1f}mm"

    return True, (
        f"PGV 候选通过: L={pgv_len:.1f}mm, "
        f"距汇合={sv_prox_len:.1f}mm"
    )


# ============================================================
# MPV 终点扩展 + 段裁剪
# ============================================================

def _find_mpv_end_by_liver_branches(adj, nodes, bp_sv_init, lpv_seg, rpv_seg,
                                     bp_liver_init):
    """
    确定 MPV 真正终点。
    定义: LPV/RPV 起点中, 沿中心线距 SV 端 (bp_sv_init) 弧长更远的那个 bp。

    解剖学语义:
        肝门是 MPV 主干自然分叉为左右肝静脉的位置。若 LPV 早早从主干分出
        (常见解剖变异), RPV 在更深处分出, 则 MPV 应延伸到 RPV 起点,
        中间过渡段并入 MPV。TIPS 不参与判定 (人工分流不属于自然血管树)。
    """
    candidates = []
    if lpv_seg is not None and len(lpv_seg) >= 1:
        candidates.append(lpv_seg[0])
    if rpv_seg is not None and len(rpv_seg) >= 1:
        candidates.append(rpv_seg[0])

    if not candidates:
        return bp_liver_init
    if len(candidates) == 1:
        return candidates[0]

    def _arc_dist(bp):
        path = find_path(adj, bp_sv_init, bp)
        if path is None:
            return -1.0
        return path_physical_length(path, nodes)

    return max(candidates, key=_arc_dist)


def _trim_branch_to_subbp(branch_seg, mpv_end_bp, branch_points):
    """段起点 == MPV 终点时, 沿段找下一个 bp 作为新起点。"""
    if branch_seg is None or len(branch_seg) < 2:
        return branch_seg
    if branch_seg[0] != mpv_end_bp:
        return branch_seg
    for i in range(1, len(branch_seg)):
        nid = branch_seg[i]
        if nid in branch_points and nid != mpv_end_bp:
            return branch_seg[i:]
    return branch_seg


def _trim_branches_to_mpv_end(branch_segs_dict, bp_mpv_end,
                               adj, branch_points, nodes):
    """
    把字典中所有段裁剪到"MPV 终点之后"开始。
      - 起点 == bp_mpv_end: 沿段找下一个 bp, 从该 bp 开始
      - 起点 != bp_mpv_end: 段不动 (从 MPV 干道中部分出, 几何路径正确)
    """
    trimmed = {}
    for name, seg in branch_segs_dict.items():
        if seg is None or len(seg) < 2:
            trimmed[name] = seg
            continue
        if seg[0] == bp_mpv_end:
            trimmed[name] = _trim_branch_to_subbp(seg, bp_mpv_end, branch_points)
        else:
            trimmed[name] = seg
    return trimmed


# ============================================================
# 三 bp 工具 (术前用)
# ============================================================

def _order_bp_chain(bps, adj):
    """将 3 个分支点排序: bp_end - bp_mid - bp_end。"""
    if len(bps) != 3:
        raise ValueError(f"期望 3 个分支点, 得到 {len(bps)}")
    a, b, c = bps
    p_ab = find_path(adj, a, b)
    p_ac = find_path(adj, a, c)
    if c in p_ab:
        return a, c, b
    elif b in p_ac:
        return a, b, c
    else:
        return b, a, c


# ============================================================
# 主入口
# ============================================================

def segment_vessels(stl_path, post_tips=None, output_json_path=None,
                    lgv_pgv_tortuosity_threshold=0.05):
    """对中心线进行解剖分段并输出 JSON。"""
    nodes, adj, parentdir = load_tree(stl_path)
    folder_name = os.path.basename(parentdir)
    if post_tips is None:
        post_tips = is_post_tips(folder_name)

    endpoints, branch_points = classify_nodes(nodes, adj)
    print(f"  节点统计: 端点 {len(endpoints)}, 分支点 {len(branch_points)}")
    print(f"  类型: {'TIPS术后' if post_tips else 'TIPS术前'}")

    segments_raw = _extract_all_segments(nodes, adj, endpoints, branch_points)

    if post_tips:
        result = _segment_post_tips(
            nodes, adj, endpoints, branch_points, segments_raw)
    else:
        result = _segment_pre_tips(
            nodes, adj, endpoints, branch_points, segments_raw,
            lgv_pgv_tortuosity_threshold)

    output = _build_output_json(folder_name, post_tips, result,
                                nodes, branch_points, endpoints)

    if output_json_path is None:
        output_json_path = os.path.join(parentdir, "centerline_profiles.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    seg_names = [n for n, v in output['segments'].items() if v is not None]
    print(f"  识别血管: {seg_names}")
    if output['has_compensation']:
        print(f"  代偿类型: {output['compensation_type']}")
    print(f"  分段结果已保存: {output_json_path}")
    return output


# ============================================================
# 术后 (post-TIPS)
# ============================================================

def _segment_post_tips(nodes, adj, endpoints, branch_points, segments_raw):
    """TIPS 术后分段。"""
    if len(branch_points) < 2:
        raise ValueError(f"TIPS术后期望 ≥2 分支点, 实际 {len(branch_points)}")

    # ----- 1. MPV 初始候选 -----
    bp_bp_segs = _find_bp_to_bp_segments(segments_raw, branch_points)
    if not bp_bp_segs:
        raise ValueError("找不到 MPV (两端均为分支点的段)")

    mpv_init_seg = max(bp_bp_segs, key=lambda s: _mpv_init_score(s, nodes))
    bp_init_a, bp_init_b = mpv_init_seg[0], mpv_init_seg[-1]

    # ----- 2. 双侧子树 -----
    sub_a = _collect_subtree(adj, bp_init_a, mpv_init_seg[1],
                             endpoints, branch_points)
    sub_b = _collect_subtree(adj, bp_init_b, mpv_init_seg[-2],
                             endpoints, branch_points)

    # ----- 3. SV 端 vs 肝侧端 -----
    score_a = max((_seg_score_sv(s, nodes) for s in sub_a['all_branches']),
                  default=0)
    score_b = max((_seg_score_sv(s, nodes) for s in sub_b['all_branches']),
                  default=0)

    if score_a >= score_b:
        sv_subtree, liver_subtree = sub_a, sub_b
        bp_sv_init, bp_liver_init = bp_init_a, bp_init_b
    else:
        sv_subtree, liver_subtree = sub_b, sub_a
        bp_sv_init, bp_liver_init = bp_init_b, bp_init_a

    print(f"    SV-score: a={score_a:.1f}, b={score_b:.1f}")
    print(f"    肝侧子树: root_brs={len(liver_subtree['root_branches'])}, "
          f"deeper_brs={len(liver_subtree['deeper_branches'])}, "
          f"all={len(liver_subtree['all_branches'])}")

    # ----- 4. SV / SMV -----
    sv_brs = sv_subtree['root_branches'] or sv_subtree['all_branches']
    sv_seg, smv_seg = _select_sv_smv(sv_brs, nodes)

    # ----- 5. TIPS / LPV / RPV (全集最长最直规则) -----
    all_liver_brs = liver_subtree['all_branches']
    tips_seg = lpv_seg = rpv_seg = None

    if len(all_liver_brs) == 1:
        tips_seg = all_liver_brs[0]
    elif len(all_liver_brs) >= 2:
        tips_seg = max(all_liver_brs, key=lambda s: _seg_score_tips(s, nodes))

        print(f"    TIPS 候选评分 (按 L·exp(-2.5·τ)):")
        for br in sorted(all_liver_brs,
                         key=lambda s: _seg_score_tips(s, nodes),
                         reverse=True):
            L = path_physical_length(br, nodes)
            t = _path_tortuosity(path_to_coords(br, nodes))
            sc = _seg_score_tips(br, nodes)
            tag = "  ← TIPS" if br is tips_seg else ""
            print(f"      根bp={br[0]}, 端点={br[-1]}, "
                  f"L={L:6.1f}mm, τ={t:.3f}, score={sc:6.1f}{tag}")

        leftover = [s for s in all_liver_brs if s is not tips_seg]
        lpv_seg, rpv_seg = _assign_lpv_rpv(leftover, nodes)

    # ----- 6. MPV 终点 = LPV/RPV 中更靠肝侧者 (TIPS 不参与) -----
    bp_mpv_end = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_sv_init, lpv_seg, rpv_seg, bp_liver_init)

    # ----- 7. 子分支起点裁剪 -----
    trimmed = _trim_branches_to_mpv_end(
        {'tips': tips_seg, 'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end, adj, branch_points, nodes)
    tips_seg = trimmed['tips']
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    # ----- 8. 最终 MPV -----
    mpv_seg = find_path(adj, bp_sv_init, bp_mpv_end)

    L_init = path_physical_length(mpv_init_seg, nodes)
    L_final = path_physical_length(mpv_seg, nodes)
    print(f"    MPV: 起点={bp_sv_init}, 终点={bp_mpv_end}")
    print(f"         长度 {L_init:.1f}mm → {L_final:.1f}mm "
          f"(扩展 +{L_final - L_init:.1f}mm)")
    print(f"    分支起点: TIPS={tips_seg[0] if tips_seg else None}, "
          f"LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")

    return {
        'segments': {
            'mpv': mpv_seg,
            'sv': sv_seg, 'smv': smv_seg,
            'tips': tips_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg,
        },
        'has_compensation': False,
        'compensation_type': None,
    }


# ============================================================
# 术前 (pre-TIPS)
# ============================================================

def _segment_pre_tips(nodes, adj, endpoints, branch_points,
                      segments_raw, lgv_pgv_threshold):
    """术前分段路由。"""
    n_bp = len(branch_points)

    if n_bp == 2:
        return _segment_pre_tips_no_comp(
            nodes, adj, endpoints, branch_points, segments_raw)
    elif n_bp == 3:
        return _segment_pre_tips_with_comp(
            nodes, adj, endpoints, branch_points, segments_raw,
            lgv_pgv_threshold)
    elif n_bp > 3:
        print(f"  警告: 分支点数={n_bp} > 3, 退化处理")
        return _segment_pre_tips_fallback(
            nodes, adj, endpoints, branch_points, segments_raw,
            lgv_pgv_threshold)
    else:
        raise ValueError(f"TIPS术前期望 ≥2 分支点, 实际 {n_bp}")


def _segment_pre_tips_no_comp(nodes, adj, endpoints, branch_points,
                              segments_raw):
    """术前无代偿: 仅 MPV/SV/SMV/LPV/RPV. 含 MPV 终点扩展。"""
    bp_bp_segs = _find_bp_to_bp_segments(segments_raw, branch_points)
    if not bp_bp_segs:
        raise ValueError("找不到 MPV")

    mpv_init_seg = max(bp_bp_segs, key=lambda s: _mpv_init_score(s, nodes))
    bp_init_a, bp_init_b = mpv_init_seg[0], mpv_init_seg[-1]

    sub_a = _collect_subtree(adj, bp_init_a, mpv_init_seg[1],
                             endpoints, branch_points)
    sub_b = _collect_subtree(adj, bp_init_b, mpv_init_seg[-2],
                             endpoints, branch_points)

    score_a = max((_seg_score_sv(s, nodes) for s in sub_a['all_branches']),
                  default=0)
    score_b = max((_seg_score_sv(s, nodes) for s in sub_b['all_branches']),
                  default=0)

    if score_a >= score_b:
        sv_subtree, liver_subtree = sub_a, sub_b
        bp_sv_init, bp_liver_init = bp_init_a, bp_init_b
    else:
        sv_subtree, liver_subtree = sub_b, sub_a
        bp_sv_init, bp_liver_init = bp_init_b, bp_init_a

    sv_brs = sv_subtree['root_branches'] or sv_subtree['all_branches']
    sv_seg, smv_seg = _select_sv_smv(sv_brs, nodes)

    liver_brs = liver_subtree['all_branches']
    lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs, nodes)

    bp_mpv_end = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_sv_init, lpv_seg, rpv_seg, bp_liver_init)
    trimmed = _trim_branches_to_mpv_end(
        {'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end, adj, branch_points, nodes)
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    mpv_seg = find_path(adj, bp_sv_init, bp_mpv_end)

    L_init = path_physical_length(mpv_init_seg, nodes)
    L_final = path_physical_length(mpv_seg, nodes)
    print(f"    MPV: 起点={bp_sv_init}, 终点={bp_mpv_end}, "
          f"长度 {L_init:.1f}→{L_final:.1f}mm")
    print(f"    分支起点: LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")

    return {
        'segments': {
            'mpv': mpv_seg, 'sv': sv_seg, 'smv': smv_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg,
        },
        'has_compensation': False,
        'compensation_type': None,
    }


def _segment_pre_tips_with_comp(nodes, adj, endpoints, branch_points,
                                segments_raw, lgv_pgv_threshold):
    """术前 3 个分支点: 区分 LGV / PGV。"""
    bps = list(branch_points)
    bp1, bp2, bp3 = _order_bp_chain(bps, adj)

    path_13 = find_path(adj, bp1, bp3)
    coords_13 = path_to_coords(path_13, nodes)
    tort_13 = _path_tortuosity(coords_13)

    print(f"  3 分支点链: {bp1}-{bp2}-{bp3}")
    print(f"  bp1↔bp3 tortuosity = {tort_13:.4f} (阈值 {lgv_pgv_threshold})")

    if tort_13 < lgv_pgv_threshold:
        print(f"  → 判定: LGV 代偿 (路径较直, MPV 贯穿 bp1→bp3)")
        return _build_lgv_segments(
            nodes, adj, endpoints, segments_raw, bp1, bp2, bp3, branch_points)
    else:
        print(f"  → 判定: PGV 代偿 (路径有转折, MPV = bp1→bp2)")
        return _build_pgv_segments(
            nodes, adj, endpoints, segments_raw, bp1, bp2, bp3, branch_points)


def _build_lgv_segments(nodes, adj, endpoints, segments_raw,
                        bp1, bp2, bp3, branch_points):
    """LGV 代偿: MPV 贯穿 bp1-bp2-bp3, LGV 从 bp2 分出。"""
    branches_1 = _find_endpoint_branches_at(segments_raw, bp1, endpoints)
    branches_3 = _find_endpoint_branches_at(segments_raw, bp3, endpoints)

    score_1 = max((_seg_score_sv(s, nodes) for s in branches_1), default=0)
    score_3 = max((_seg_score_sv(s, nodes) for s in branches_3), default=0)

    if score_3 >= score_1:
        sv_branches, liver_branches_direct = branches_3, branches_1
        bp_svjct, bp_liver = bp3, bp1
    else:
        sv_branches, liver_branches_direct = branches_1, branches_3
        bp_svjct, bp_liver = bp1, bp3

    L_to_svjct = path_physical_length(find_path(adj, bp2, bp_svjct), nodes)
    L_to_liver = path_physical_length(find_path(adj, bp2, bp_liver), nodes)
    print(f"    LGV分叉点位置: 到SV端={L_to_svjct:.1f}mm, "
          f"到肝侧={L_to_liver:.1f}mm")

    sv_seg, smv_seg = _select_sv_smv(sv_branches, nodes)

    # 收集肝侧子树以处理嵌套 LPV/RPV
    path_liver_to_svjct = find_path(adj, bp_liver, bp_svjct)
    excl_nb = path_liver_to_svjct[1] if len(path_liver_to_svjct) >= 2 else None
    liver_subtree = _collect_subtree(
        adj, bp_liver, excl_nb, endpoints, branch_points)

    liver_brs_all = liver_subtree['all_branches'] or liver_branches_direct
    lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs_all, nodes)

    # MPV 终点扩展 (肝侧)
    bp_mpv_end_liver = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_svjct, lpv_seg, rpv_seg, bp_liver)
    trimmed = _trim_branches_to_mpv_end(
        {'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end_liver, adj, branch_points, nodes)
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    # MPV = bp_mpv_end_liver → bp_svjct, 注意 bp2 仍在路径中, LGV 仍从 bp2 分出
    mpv_seg = find_path(adj, bp_mpv_end_liver, bp_svjct)

    lgv_branches = _find_endpoint_branches_at(segments_raw, bp2, endpoints)
    lgv_seg = lgv_branches[0] if lgv_branches else None

    print(f"    MPV: 肝侧={bp_mpv_end_liver} → SV交汇={bp_svjct}")
    print(f"    分支起点: LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")

    return {
        'segments': {
            'mpv': mpv_seg, 'sv': sv_seg, 'smv': smv_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg, 'lgv': lgv_seg,
        },
        'has_compensation': True,
        'compensation_type': 'LGV',
    }


def _build_pgv_segments(nodes, adj, endpoints, segments_raw,
                        bp1, bp2, bp3, branch_points):
    """PGV 代偿: bp_liver - MPV - bp_svjct - SV-prox - bp_svsub - SV-distal/PGV."""
    branches_1 = _find_endpoint_branches_at(segments_raw, bp1, endpoints)
    branches_3 = _find_endpoint_branches_at(segments_raw, bp3, endpoints)

    score_1 = max((_seg_score_sv(s, nodes) for s in branches_1), default=0)
    score_3 = max((_seg_score_sv(s, nodes) for s in branches_3), default=0)

    if score_3 >= score_1:
        bp_liver, bp_svsub = bp1, bp3
        liver_branches_direct, svsub_branches = branches_1, branches_3
    else:
        bp_liver, bp_svsub = bp3, bp1
        liver_branches_direct, svsub_branches = branches_3, branches_1

    bp_svjct = bp2
    print(f"    PGV拓扑: 肝侧={bp_liver}, MPV/SV交汇={bp_svjct}, "
          f"SV远端bp={bp_svsub}")

    # 1) LPV / RPV: 子树扫描以处理嵌套
    path_liver_to_svjct = find_path(adj, bp_liver, bp_svjct)
    excl_nb = path_liver_to_svjct[1] if len(path_liver_to_svjct) >= 2 else None
    liver_subtree = _collect_subtree(
        adj, bp_liver, excl_nb, endpoints, branch_points)

    liver_brs_all = liver_subtree['all_branches'] or liver_branches_direct
    lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs_all, nodes)

    # 2) SMV 在 bp_svjct
    smv_branches = _find_endpoint_branches_at(segments_raw, bp_svjct, endpoints)
    smv_seg = smv_branches[0] if smv_branches else None

    # 3) SV-distal vs PGV: 方向一致性
    sv_main_path = find_path(adj, bp_svjct, bp_svsub)
    sv_distal_seg, pgv_seg = _select_sv_distal_pgv(
        svsub_branches, sv_main_path, nodes)

    # 4) MPV 终点扩展 (肝侧) + LPV/RPV 裁剪
    bp_mpv_end_liver = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_svjct, lpv_seg, rpv_seg, bp_liver)
    trimmed = _trim_branches_to_mpv_end(
        {'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end_liver, adj, branch_points, nodes)
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    # 5) MPV = bp_mpv_end_liver → bp_svjct
    mpv_seg = find_path(adj, bp_mpv_end_liver, bp_svjct)

    # 6) SV = bp_svjct → bp_svsub + SV-distal
    sv_proximal = find_path(adj, bp_svjct, bp_svsub)
    if sv_distal_seg is not None:
        sv_seg = sv_proximal + sv_distal_seg[1:]
    else:
        sv_seg = sv_proximal

    pgv_ok, pgv_reason = _pgv_candidate_quality(
        pgv_seg, sv_distal_seg, sv_proximal, smv_seg, nodes)
    if pgv_ok:
        print(f"    PGV质控: {pgv_reason}")
    else:
        print(f"    PGV质控: {pgv_reason} → 降级为无代偿分段")
        pgv_seg = None

    print(f"    MPV: 肝侧={bp_mpv_end_liver} → SV交汇={bp_svjct}")
    print(f"    分支起点: LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")

    return {
        'segments': {
            'mpv': mpv_seg, 'sv': sv_seg, 'smv': smv_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg, 'pgv': pgv_seg,
        },
        'has_compensation': bool(pgv_ok),
        'compensation_type': 'PGV' if pgv_ok else None,
    }


def _segment_pre_tips_fallback(nodes, adj, endpoints, branch_points,
                               segments_raw, lgv_pgv_threshold):
    """>3 分支点的退化处理。"""
    sorted_bps = sorted(branch_points,
                        key=lambda b: len(adj[b]), reverse=True)
    chosen = sorted_bps[:3] if len(sorted_bps) >= 3 else list(branch_points)
    if len(chosen) < 3:
        return _segment_pre_tips_no_comp(
            nodes, adj, endpoints, set(chosen), segments_raw)
    return _segment_pre_tips_with_comp(
        nodes, adj, endpoints, set(chosen), segments_raw, lgv_pgv_threshold)


# ============================================================
# JSON 构造
# ============================================================

def _build_output_json(folder_name, post_tips, result, nodes,
                       branch_points, endpoints):
    out = {
        "patient_id": folder_name,
        "is_post_tips": post_tips,
        "has_compensation": result.get('has_compensation', False),
        "compensation_type": result.get('compensation_type', None),
        "n_branch_points": len(branch_points),
        "n_endpoints": len(endpoints),
        "branch_points": [
            {"id": int(bp),
             "coord": [float(nodes[bp]['x']),
                       float(nodes[bp]['y']),
                       float(nodes[bp]['z'])]}
            for bp in sorted(branch_points)
        ],
        "segments": {}
    }
    for name, path in result['segments'].items():
        if path is None or len(path) < 2:
            out['segments'][name] = None
            continue
        coords = path_to_coords(path, nodes)
        out['segments'][name] = {
            "path": [int(n) for n in path],
            "endpoints_id": [int(path[0]), int(path[-1])],
            "endpoints_coord": [
                [float(nodes[path[0]]['x']),
                 float(nodes[path[0]]['y']),
                 float(nodes[path[0]]['z'])],
                [float(nodes[path[-1]]['x']),
                 float(nodes[path[-1]]['y']),
                 float(nodes[path[-1]]['z'])],
            ],
            "n_points": len(path),
            "length_mm": float(path_physical_length(path, nodes)),
            "tortuosity": float(_path_tortuosity(coords)),
            "mean_curvature": float(_path_mean_curvature(coords)),
        }
    return out


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    segment_vessels(p)
