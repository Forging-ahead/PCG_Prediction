"""
门静脉血管分析完整流程
=======================
流水线 (按患者):
  Step 1:    中心线提取        → CenterlinePoints.txt
  Step 2:    中心线平滑        → newCenterlist.txt
  Step 3:    解剖分段          → centerline_profiles.json
  Step 4:    统计特征          → portal_vein_features.json
  Step 5:    剖面特征          → centerline_pointwise_profiles.json
  Step 5.5:  导出可视化        → vis_interactive.html + vis_overview.png
  Step 6:    VTK 弹窗 (可选)

流水线 (跨患者, 需要 patient/label/<TARGET>.txt 存在):
  Step A: 统计特征 vs 目标量相关性分析
  Step B: 剖面特征 vs 目标量逐点相关性分析

文件夹命名规则:
  20210909WuJinHeng       → TIPS术前
  20210921WuJinHeng#      → TIPS术后 (含 '#')
  *@*  或  *!*            → 无效, 跳过
"""

import os
import time
import traceback

from extract_centerline import extract_centerline
from smooth_centerline import smooth_centerline
from segment_vessels import (segment_vessels, is_post_tips,
                              is_invalid_folder)
from extract_features import extract_all_features
from extract_profiles import extract_profiles
from export_visualization import export_patient_visualization
from visualize_segments import visualize_segments


# ============================================================
# 步骤选择 (开关式控制每步是否执行)
# ============================================================

class PipelineSteps:
    """各步骤启用开关。某些步骤失败不影响后续步骤继续尝试。"""
    extract_centerline = True
    smooth_centerline = True
    segment_vessels = True
    extract_features = True
    extract_profiles = True
    export_visualization = True
    visualize = True


# ============================================================
# 单患者处理
# ============================================================

def _clean_old_outputs(folder_path):
    """清理上一轮产生的中间文件 (避免与新结果混淆)。"""
    for old in ["CenterlinePoints.txt",
                "newCenterlist.txt",
                "centerline_profiles.json",
                "portal_vein_features.json",
                "unified_features.json",
                "centerline_pointwise_profiles.json",
                "sv_smv_angle.json",
                "vis_interactive.html",
                "vis_overview.png",
                "centerline_screenshot.png",
                "segment_screenshot.png"]:
        p = os.path.join(folder_path, old)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def _process_one_patient(stl_path, post_tips, params, steps):
    """
    按流水线处理单个患者。任何一步异常都被捕获并打印, 不向上抛出。

    返回: dict, 各步骤的成功 / 失败状态。
    """
    folder_path = os.path.dirname(stl_path)
    status = {
        'centerline': False, 'smooth': False, 'segment': False,
        'features': False, 'profiles': False,
        'export_vis': False, 'visualize': False,
    }

    # ---- Step 1: 中心线提取 ----
    if steps.extract_centerline:
        try:
            t0 = time.time()
            extract_centerline(
                stl_path,
                pitch=params['pitch'],
                min_branch_length_mm=params['min_branch_length_mm'],
                min_relative_length=params['min_relative_length'],
                min_radius_ratio=params['min_radius_ratio'],
                keep_radius_ratio=params['keep_radius_ratio'],
                absolute_min_branch_length_mm=params[
                    'absolute_min_branch_length_mm'],
                absolute_min_radius_mm=params['absolute_min_radius_mm'],
                merge_bp_distance_mm=params['merge_bp_distance_mm'])
            print(f"  [Step 1] 中心线提取: {time.time()-t0:.2f}s")
            status['centerline'] = True
        except Exception as e:
            print(f"  [Step 1] 中心线提取失败: {e}")
            traceback.print_exc()
            return status  # 中心线失败, 后续无意义

    # ---- Step 2: 中心线平滑 ----
    if steps.smooth_centerline:
        try:
            t0 = time.time()
            smooth_centerline(stl_path)
            print(f"  [Step 2] 中心线平滑: {time.time()-t0:.2f}s")
            status['smooth'] = True
        except Exception as e:
            print(f"  [Step 2] 平滑失败: {e}")
            traceback.print_exc()

    # ---- Step 3: 解剖分段 ----
    if steps.segment_vessels:
        try:
            t0 = time.time()
            segment_vessels(stl_path, post_tips=post_tips)
            print(f"  [Step 3] 解剖分段: {time.time()-t0:.2f}s")
            status['segment'] = True
        except Exception as e:
            print(f"  [Step 3] 分段失败: {e}")
            traceback.print_exc()
            # 分段失败, 后续特征/剖面无意义, 但仍可降级导出可视化
            if steps.export_visualization:
                try:
                    export_patient_visualization(
                        stl_path, export_html=True, export_png=True)
                    status['export_vis'] = True
                except Exception as ee:
                    print(f"  [Step 5.5] 降级可视化失败: {ee}")
            return status

    # ---- Step 4: 统计特征 ----
    if steps.extract_features:
        try:
            t0 = time.time()
            extract_all_features(
                stl_path,
                n_fit_points=params['n_fit_points'],
                curvature_window=params['curvature_window'],
                sample_step=params['sample_step'],
                pitch=params['pitch'])
            print(f"  [Step 4] 统计特征: {time.time()-t0:.2f}s")
            status['features'] = True
        except Exception as e:
            print(f"  [Step 4] 统计特征失败: {e}")
            traceback.print_exc()

    # ---- Step 5: 剖面特征 ----
    if steps.extract_profiles:
        try:
            t0 = time.time()
            extract_profiles(
                stl_path,
                n_points=params['n_profile_points'],
                pitch=params['pitch'],
                curvature_window=params['curvature_window'],
                section_step=params['sample_step'],
                ownership_factor=params['ownership_factor'],
                max_diameter_rate_per_mm=params[
                    'max_diameter_rate_per_mm'])
            print(f"  [Step 5] 剖面特征: {time.time()-t0:.2f}s")
            status['profiles'] = True
        except Exception as e:
            print(f"  [Step 5] 剖面特征失败: {e}")
            traceback.print_exc()

    # ---- Step 5.5: 导出可视化 (HTML + PNG) ----
    if steps.export_visualization:
        try:
            t0 = time.time()
            export_patient_visualization(
                stl_path,
                export_html=True,
                export_png=True,
                verbose=True)
            print(f"  [Step 5.5] 可视化导出: {time.time()-t0:.2f}s")
            status['export_vis'] = True
        except Exception as e:
            print(f"  [Step 5.5] 可视化导出失败: {e}")
            traceback.print_exc()

    # ---- Step 6: VTK 弹窗可视化 ----
    if steps.visualize:
        try:
            visualize_segments(stl_path, block=True)
            status['visualize'] = True
        except Exception as e:
            print(f"  [Step 6] VTK 弹窗失败: {e}")

    return status


# ============================================================
# 批量处理
# ============================================================

def process_stl_files(root_folder, params, steps,
                       stl_name="vessel.stl",
                       clean_old=True):
    """
    批量处理 root_folder 下所有合法子文件夹的 vessel.stl。

    参数:
        root_folder: 根目录
        params:      参数字典 (见底部 DEFAULT_PARAMS)
        steps:       PipelineSteps 实例
        stl_name:    STL 文件名
        clean_old:   是否在每轮处理前清理旧文件
    """
    if not os.path.exists(root_folder):
        print(f"根目录不存在: {root_folder}")
        return

    subfolders = sorted([
        f for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f))
    ])
    if not subfolders:
        print("未找到子文件夹")
        return

    valid = [f for f in subfolders if not is_invalid_folder(f)]
    invalid = [f for f in subfolders if is_invalid_folder(f)]

    print(f"\n{'='*60}")
    print(f"批量处理: {root_folder}")
    print(f"{'='*60}")
    print(f"找到 {len(subfolders)} 个子文件夹  "
          f"(有效 {len(valid)}, 无效 {len(invalid)})")
    if invalid:
        print(f"  跳过 (含 @ 或 !): {invalid}")

    summary = {
        'total': len(valid),
        'no_stl': 0,
        'centerline_ok': 0,
        'segment_ok': 0,
        'features_ok': 0,
        'profiles_ok': 0,
        'export_vis_ok': 0,
    }

    for folder in valid:
        folder_path = os.path.join(root_folder, folder)
        stl_path = os.path.join(folder_path, stl_name)

        if not os.path.exists(stl_path):
            print(f"\n[Skip] {folder}: 缺少 {stl_name}")
            summary['no_stl'] += 1
            continue

        post_tips = is_post_tips(folder)
        tag = "TIPS术后" if post_tips else "TIPS术前"

        print(f"\n{'='*60}")
        print(f"处理: {folder}   [{tag}]")
        print(f"{'='*60}")

        if clean_old:
            _clean_old_outputs(folder_path)

        status = _process_one_patient(stl_path, post_tips, params, steps)

        if status['centerline']:    summary['centerline_ok']   += 1
        if status['segment']:       summary['segment_ok']      += 1
        if status['features']:      summary['features_ok']     += 1
        if status['profiles']:      summary['profiles_ok']     += 1
        if status['export_vis']:    summary['export_vis_ok']   += 1

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"  总样本: {summary['total']}")
    print(f"  缺少 STL: {summary['no_stl']}")
    print(f"  中心线提取成功: {summary['centerline_ok']} / {summary['total']}")
    print(f"  解剖分段成功:   {summary['segment_ok']} / {summary['total']}")
    print(f"  统计特征成功:   {summary['features_ok']} / {summary['total']}")
    print(f"  剖面特征成功:   {summary['profiles_ok']} / {summary['total']}")
    print(f"  可视化导出成功: {summary['export_vis_ok']} / {summary['total']}")

    return summary


# ============================================================
# 跨患者相关性分析 (Step A + Step B)
# ============================================================

def run_correlation_analysis(root_folder, target="PVP",
                              output_root=None,
                              run_statistical=True,
                              run_profile=True,
                              drop_features_above_missing=0.5,
                              min_branch_coverage=0.3):
    """
    跨患者相关性分析。需要每个 patient 文件夹下:
      label/<TARGET>.txt                  (PVP 或 PCG 数值)
      portal_vein_features.json           (统计特征, Step 4 输出)
      centerline_pointwise_profiles.json  (剖面特征, Step 5 输出)
    """
    target = target.upper()
    print(f"\n{'='*60}")
    print(f"跨患者相关性分析: target={target}")
    print(f"{'='*60}")

    # ---- Step A: 统计特征 vs target ----
    if run_statistical:
        print(f"\n--- Step A: 统计特征 vs {target} ---")
        try:
            from correlation_analysis import collect_and_analyze
            stat_dir = (output_root if output_root
                        else os.path.join(root_folder,
                                          f"correlation_{target.lower()}"))
            collect_and_analyze(
                root_folder,
                output_dir=stat_dir,
                target=target,
                drop_features_above_missing=drop_features_above_missing)
        except Exception as e:
            print(f"  Step A 失败: {e}")
            traceback.print_exc()

    # ---- Step B: 剖面特征逐点相关 ----
    if run_profile:
        print(f"\n--- Step B: 剖面特征逐点 vs {target} ---")
        try:
            from profile_correlation import run_profile_analysis
            run_profile_analysis(
                root_folder,
                output_dir=None,  # None = 自动放到 root/profile_correlation_<target>/
                target=target,
                min_branch_coverage=min_branch_coverage)
        except Exception as e:
            print(f"  Step B 失败: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"相关性分析完成")
    print(f"{'='*60}")


# ============================================================
# 默认参数
# ============================================================

DEFAULT_PARAMS = {
    # 中心线提取
    'pitch': 0.5,
    'min_branch_length_mm': 8.0,
    'min_relative_length': 0.05,
    'min_radius_ratio': 0.4,
    'keep_radius_ratio': 0.55,       # 保护门: r_branch/r_junction ≥ 0.55 时
                                      # 视为真分支, 跳过所有长度判据
    'absolute_min_branch_length_mm': 3.0,  # 硬阈值: 弧长 < 3mm 必为骨架毛刺,
                                            # 跳过保护门强剪
    'absolute_min_radius_mm': 0.75,        # 硬阈值: max_radius < 0.75mm
                                            # (≤1.5 体素) 必为噪声, 强剪
    'merge_bp_distance_mm': 5.0,

    # 特征 / 剖面
    'n_fit_points': 10,
    'n_profile_points': 100,
    'curvature_window': 7,
    'sample_step': 3,
    'ownership_factor': 1.8,        # 中心线锚定最大内切半径裁剪倍数:
                                     # clean_area = raw_section ∩ circle(c, k*r_anchor)
    'max_diameter_rate_per_mm': 0.5,  # 沿管轴等效直径相对变化率上限 (1/mm)
                                       # 0.5 = 每 mm 最多 50% 变化, 超阈孤立
                                       # 点视为单点突变伪影
}


# ============================================================
# 主入口 (用户配置)
# ============================================================

if __name__ == '__main__':

    # ============================================
    # 用户配置
    # ============================================

    ROOT_FOLDER = r"E:\zhengzhou_vkan3"

    # 跨患者相关性分析的目标量 ("PVP" 或 "PCG")
    TARGET = "PVP"

    # 模式选择:
    #   "all"       - 处理 + 跨患者分析 (完整流程)
    #   "process"   - 仅处理 (Step 1 - 5.5, 6)
    #   "correlate" - 仅跨患者分析 (Step A - B), 假设之前已处理过
    MODE = "all"

    # 各步骤开关 (仅在 MODE="all" 或 "process" 时生效)
    steps = PipelineSteps()
    steps.extract_centerline = True
    steps.smooth_centerline = True
    steps.segment_vessels = True
    steps.extract_features = True
    steps.extract_profiles = True
    steps.export_visualization = True   # 导出 HTML + PNG
    steps.visualize = True              # VTK 弹窗 (批量时建议关掉)

    # 参数
    params = dict(DEFAULT_PARAMS)
    # 例: params['min_branch_length_mm'] = 10.0

    # 跨患者分析的开关
    run_statistical_correlation = True
    run_profile_correlation = True

    # ============================================
    # 执行
    # ============================================

    t_total = time.time()

    if MODE in ("all", "process"):
        process_stl_files(
            ROOT_FOLDER,
            params=params,
            steps=steps,
            stl_name="vessel.stl",
            clean_old=True)

    if MODE in ("all", "correlate"):
        run_correlation_analysis(
            ROOT_FOLDER,
            target=TARGET,
            run_statistical=run_statistical_correlation,
            run_profile=run_profile_correlation,
            drop_features_above_missing=0.5,
            min_branch_coverage=0.3)

    print(f"\n{'='*60}")
    print(f"全部完成! 总耗时: {time.time() - t_total:.1f}s")
    print(f"{'='*60}")
