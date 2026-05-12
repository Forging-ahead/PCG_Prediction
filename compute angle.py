"""
SV-SMV 夹角独立调用接口（v3 - 读 JSON 驱动）
"""

import os
import json
from utils import load_tree
from extract_features import _compute_sv_smv_angle_from_segments


def compute_sv_smv_angle(stl_path, n_fit_points=10, output_dir=None):
    """独立计算 SV-SMV 夹角。"""
    print(f"\n===== SV-SMV 夹角计算 =====")
    nodes, adj, parentdir = load_tree(stl_path)
    if output_dir is None:
        output_dir = parentdir

    seg_json = os.path.join(parentdir, "centerline_profiles.json")
    if not os.path.exists(seg_json):
        raise FileNotFoundError(f"分段 JSON 不存在: {seg_json}")
    with open(seg_json, 'r', encoding='utf-8') as f:
        seg_dict = json.load(f).get('segments', {})

    result, err = _compute_sv_smv_angle_from_segments(
        seg_dict, nodes, n_fit_points=n_fit_points)

    if result is None:
        raise ValueError(f"无法计算夹角: {err}")

    json_path = os.path.join(output_dir, "sv_smv_angle.json")
    save_data = {k: v for k, v in result.items() if not k.startswith('_')}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {json_path}")
    print(f"  夹角: {result['angle_degrees']:.1f}°")
    return result


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    compute_sv_smv_angle(p)