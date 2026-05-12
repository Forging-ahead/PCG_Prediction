"""
血管分段可视化模块 (VTK 窗口版)
================================
读取 centerline_profiles.json + 中心线文件 + 逐点剖面 JSON,
在 VTK 中按解剖段着色显示, 并支持最大截面圈交互显示。

快捷键:
  R: 重置视角         1-8: 切换各段可见性
  M: 切换血管模型      C: 切换原始中心线
  L: 切换标签         B: 切换分支点 (默认隐藏)
  X: 切换最大截面圈   (默认隐藏)
  +/-: 透明度         W: 线框/实体
  S: 截图             Q: 退出
"""

import os
import json
import numpy as np
import vtk

from utils import load_tree


# ============================================================
# 配色 / 映射
# ============================================================

SEGMENT_COLORS = {
    'mpv':  (1.00, 0.20, 0.20),
    'sv':   (0.20, 0.50, 1.00),
    'smv':  (1.00, 0.60, 0.20),
    'lpv':  (0.70, 0.30, 1.00),
    'rpv':  (0.20, 0.90, 0.40),
    'tips': (0.00, 0.90, 0.90),
    'lgv':  (1.00, 0.90, 0.20),
    'pgv':  (1.00, 0.30, 0.90),
}

SEGMENT_LABELS = {
    'mpv': 'MPV', 'sv': 'SV', 'smv': 'SMV',
    'lpv': 'LPV', 'rpv': 'RPV', 'tips': 'TIPS',
    'lgv': 'LGV', 'pgv': 'PGV',
}

KEY_TO_SEG = {
    '1': 'mpv', '2': 'sv', '3': 'smv',
    '4': 'lpv', '5': 'rpv', '6': 'tips',
    '7': 'lgv', '8': 'pgv',
}


# ============================================================
# 交互器
# ============================================================

class SegmentInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, renderer, render_window, actors_dict, stl_path):
        super().__init__()
        self.renderer = renderer
        self.render_window = render_window
        self.actors = actors_dict
        self.stl_path = stl_path
        self.stl_opacity = 0.25

        self.info_actor = vtk.vtkTextActor()
        self.info_actor.GetTextProperty().SetFontSize(13)
        self.info_actor.GetTextProperty().SetColor(0.05, 0.05, 0.05)
        self.info_actor.SetPosition(10, 145)
        self.info_actor.SetInput("")
        self.renderer.AddActor2D(self.info_actor)

        self.AddObserver("KeyPressEvent", self._on_key_press)

    def _toggle(self, key, label):
        a = self.actors.get(key)
        if a:
            vis = a.GetVisibility()
            a.SetVisibility(not vis)
            lab = self.actors.get(key + '_label')
            if lab:
                lab.SetVisibility(not vis)
            self.info_actor.SetInput(f"{label}: {'Hide' if vis else 'Show'}")
            self.render_window.Render()

    def _toggle_group(self, prefix, label):
        any_vis = any(a.GetVisibility() for k, a in self.actors.items()
                      if k.startswith(prefix))
        for k, a in self.actors.items():
            if k.startswith(prefix):
                a.SetVisibility(not any_vis)
        self.info_actor.SetInput(f"{label}: {'Show' if not any_vis else 'Hide'}")
        self.render_window.Render()

    def _on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()

        if key in ('r', 'R'):
            self.renderer.ResetCamera()
            self.render_window.Render()
        elif key in KEY_TO_SEG:
            seg = KEY_TO_SEG[key]
            self._toggle('seg_' + seg, SEGMENT_LABELS[seg])
        elif key in ('m', 'M'):
            self._toggle('stl', 'Vessel mesh')
        elif key in ('c', 'C'):
            self._toggle('centerline', 'Centerline')
        elif key in ('l', 'L'):
            self._toggle_group('label_', 'Labels')
        elif key in ('b', 'B'):
            self._toggle_group('bp_', 'Branch points')
        elif key in ('x', 'X'):
            self._toggle_group('maxsec_', 'Max sections')
        elif key in ('plus', 'equal', 'minus'):
            stl = self.actors.get('stl')
            if stl:
                d = 0.1 if key != 'minus' else -0.1
                self.stl_opacity = max(0.0, min(1.0, self.stl_opacity + d))
                stl.GetProperty().SetOpacity(self.stl_opacity)
                self.info_actor.SetInput(f"Opacity: {self.stl_opacity:.1f}")
                self.render_window.Render()
        elif key in ('w', 'W'):
            stl = self.actors.get('stl')
            if stl:
                rep = stl.GetProperty().GetRepresentation()
                if rep == vtk.VTK_SURFACE:
                    stl.GetProperty().SetRepresentationToWireframe()
                else:
                    stl.GetProperty().SetRepresentationToSurface()
                self.render_window.Render()
        elif key in ('s', 'S'):
            self._screenshot()
        elif key in ('q', 'Q'):
            self.render_window.Finalize()
            self.GetInteractor().TerminateApp()

    def _screenshot(self):
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.render_window)
        w2i.Update()
        pdir = os.path.dirname(self.stl_path)
        path = os.path.join(pdir, "segment_screenshot.png")
        wr = vtk.vtkPNGWriter()
        wr.SetFileName(path)
        wr.SetInputConnection(w2i.GetOutputPort())
        wr.Write()
        print(f"  截图已保存: {path}")
        self.info_actor.SetInput("Screenshot saved")
        self.render_window.Render()


# ============================================================
# Actor 构造 - 基础
# ============================================================

def _build_polyline_actor(coords, color, line_width=5, point_size=0):
    """从坐标数组构建一个折线 actor。"""
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for c in coords:
        pts.InsertNextPoint(c)
    for i in range(len(coords) - 1):
        ln = vtk.vtkLine()
        ln.GetPointIds().SetId(0, i)
        ln.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(ln)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(line_width)
    if point_size > 0:
        actor.GetProperty().SetPointSize(point_size)
    return actor


def _build_vtk_polyline_loop(coords, color, line_width=4):
    """构造闭合 VTK 折线 (尾点连回首点)。用于截面圈。"""
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    n = len(coords)
    for c in coords:
        pts.InsertNextPoint(c)
    for i in range(n):
        ln = vtk.vtkLine()
        ln.GetPointIds().SetId(0, i)
        ln.GetPointIds().SetId(1, (i + 1) % n)
        lines.InsertNextCell(ln)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(line_width)
    return actor


def _build_label_actor(text, position, color, renderer):
    """3D 跟随相机的文字标签。"""
    src = vtk.vtkVectorText()
    src.SetText(text)
    src.Update()

    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(src.GetOutputPort())

    follower = vtk.vtkFollower()
    follower.SetMapper(m)
    follower.SetScale(3.0, 3.0, 3.0)
    follower.SetPosition(position[0] + 1.5, position[1] + 1.5, position[2] + 1.5)
    follower.GetProperty().SetColor(color)
    follower.GetProperty().SetOpacity(1.0)
    follower.SetCamera(renderer.GetActiveCamera())
    return follower


def _build_sphere_actor(center, radius, color):
    """球体 actor (用于分支点 / 中心标记)。"""
    s = vtk.vtkSphereSource()
    s.SetCenter(center)
    s.SetRadius(radius)
    s.SetPhiResolution(16)
    s.SetThetaResolution(16)
    s.Update()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(color)
    return a


def _build_centerline_actor(nodes, color=(0.5, 0.5, 0.5), line_width=1.5):
    """整条中心线 (灰色参考)。"""
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    id_to_idx = {}
    for nid, n in nodes.items():
        idx = pts.InsertNextPoint(n['x'], n['y'], n['z'])
        id_to_idx[nid] = idx
    for nid, n in nodes.items():
        for child in [n['left'], n['right']]:
            if child >= 0 and child in id_to_idx:
                ln = vtk.vtkLine()
                ln.GetPointIds().SetId(0, id_to_idx[nid])
                ln.GetPointIds().SetId(1, id_to_idx[child])
                lines.InsertNextCell(ln)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(pd)
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(color)
    a.GetProperty().SetLineWidth(line_width)
    return a


# ============================================================
# 最大截面圈 (复用 export_visualization 里的工具)
# ============================================================

def _add_max_section_actors(renderer, stl_path, seg_data, nodes, actors_dict):
    """
    在 VTK renderer 中为每段添加最大截面圈、等效圆、中心标记。
    默认全部隐藏, 通过 X 键切换显示。

    每段添加的 actor 键名 (统一前缀 'maxsec_' 便于按 X 切换):
        maxsec_ring_<seg_name>     - 真实截面闭合环
        maxsec_eq_<seg_name>       - 等效圆 (用 4 段虚线模拟"虚线"风格)
        maxsec_marker_<seg_name>   - 中心钻石球
    """
    parentdir = os.path.dirname(stl_path)

    # 加载 pointwise 数据
    pw_path = os.path.join(parentdir, "centerline_pointwise_profiles.json")
    if not os.path.exists(pw_path):
        print("  [Max sections] 缺少 centerline_pointwise_profiles.json")
        return

    with open(pw_path, 'r', encoding='utf-8') as f:
        pointwise = json.load(f)

    # 加载 STL mesh
    try:
        import trimesh
        mesh = trimesh.load(stl_path)
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'geometry'):
                mesh = list(mesh.geometry.values())[0]
            else:
                print("  [Max sections] STL 加载失败")
                return
    except Exception as e:
        print(f"  [Max sections] STL 加载异常: {e}")
        return

    # 复用 export_visualization 的工具函数
    try:
        from export_visualization import (
            _find_max_section, _interp_centerline_at_pos,
            _compute_real_cross_section_ring, _build_equivalent_circle_3d,
            SEGMENT_COLORS as EXP_HEX_COLORS)
    except ImportError as e:
        print(f"  [Max sections] 依赖 export_visualization 模块: {e}")
        return

    n_added = 0
    for seg_name, seg_info in seg_data.get('segments', {}).items():
        if seg_info is None:
            continue
        profile = pointwise.get(seg_name)
        if profile is None:
            continue

        max_info = _find_max_section(profile)
        if max_info is None:
            continue

        point, tangent = _interp_centerline_at_pos(
            seg_info['path'], nodes, max_info['pos_index'], n_total=100)
        if point is None:
            continue

        # 此模块内的 SEGMENT_COLORS 是 0-1 RGB
        rgb = SEGMENT_COLORS.get(seg_name, (0.5, 0.5, 0.5))

        # ---- 真实截面圈 ----
        ring_3d = _compute_real_cross_section_ring(mesh, point, tangent)
        if ring_3d is not None and len(ring_3d) >= 3:
            ring_actor = _build_vtk_polyline_loop(ring_3d, rgb, line_width=4)
            ring_actor.SetVisibility(False)
            renderer.AddActor(ring_actor)
            actors_dict['maxsec_ring_' + seg_name] = ring_actor
            n_added += 1
            print(f"    [{seg_name.upper()}] @ {max_info['position_pct']}%, "
                  f"A={max_info['area']:.2f}mm², 真实环 {len(ring_3d)} 点 ✓")
        else:
            print(f"    [{seg_name.upper()}] @ {max_info['position_pct']}%, "
                  f"A={max_info['area']:.2f}mm², 真实环计算失败")

        # ---- 等效圆 ----
        eq_radius = max_info['eq_diameter'] / 2.0
        if eq_radius > 1e-6:
            eq_circle = _build_equivalent_circle_3d(
                point, tangent, eq_radius, n_pts=64)
            eq_actor = _build_vtk_polyline_loop(eq_circle, rgb, line_width=2)
            eq_actor.SetVisibility(False)
            renderer.AddActor(eq_actor)
            actors_dict['maxsec_eq_' + seg_name] = eq_actor

        # ---- 中心钻石球 ----
        marker = _build_sphere_actor(point, 1.2, rgb)
        marker.SetVisibility(False)
        renderer.AddActor(marker)
        actors_dict['maxsec_marker_' + seg_name] = marker

    if n_added > 0:
        print(f"  [Max sections] 已加载 {n_added} 个段 (按 X 切换显示)")


# ============================================================
# 主入口
# ============================================================

def visualize_segments(stl_path, block=True):
    """打开 VTK 窗口可视化分段结果。"""
    parentdir = os.path.dirname(stl_path)
    json_path = os.path.join(parentdir, "centerline_profiles.json")

    if not os.path.exists(json_path):
        print(f"  [Warn] 未找到分段文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)

    nodes, _, _ = load_tree(stl_path)

    def _run():
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.97)

        rw = vtk.vtkRenderWindow()
        rw.SetSize(1280, 900)
        rw.SetWindowName(f"Segments - {seg_data['patient_id']}")
        rw.AddRenderer(renderer)
        rw.SetMultiSamples(4)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)

        actors_dict = {}

        # ----- STL 血管模型 -----
        if os.path.exists(stl_path):
            try:
                rd = vtk.vtkSTLReader()
                rd.SetFileName(stl_path)
                rd.Update()
                m = vtk.vtkPolyDataMapper()
                m.SetInputConnection(rd.GetOutputPort())
                m.ScalarVisibilityOff()
                a = vtk.vtkActor()
                a.SetMapper(m)
                a.GetProperty().SetOpacity(0.25)
                a.GetProperty().SetColor(0.78, 0.78, 0.85)
                renderer.AddActor(a)
                actors_dict['stl'] = a
            except Exception as e:
                print(f"  STL 加载失败: {e}")

        # ----- 原始中心线 (灰色, 默认隐藏) -----
        try:
            cl_actor = _build_centerline_actor(nodes, color=(0.5, 0.5, 0.5),
                                                line_width=1.5)
            cl_actor.SetVisibility(False)
            renderer.AddActor(cl_actor)
            actors_dict['centerline'] = cl_actor
        except Exception:
            pass

        # ----- 各解剖段 -----
        loaded_segs = []
        for seg_name, seg_info in seg_data['segments'].items():
            if seg_info is None:
                continue
            color = SEGMENT_COLORS.get(seg_name, (0.5, 0.5, 0.5))
            label = SEGMENT_LABELS.get(seg_name, seg_name.upper())

            coords = [[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
                       for nid in seg_info['path']]
            line_a = _build_polyline_actor(coords, color, line_width=6)
            renderer.AddActor(line_a)
            actors_dict['seg_' + seg_name] = line_a

            mid_idx = len(coords) // 2
            lab_a = _build_label_actor(label, coords[mid_idx], color, renderer)
            renderer.AddActor(lab_a)
            actors_dict['seg_' + seg_name + '_label'] = lab_a
            actors_dict['label_' + seg_name] = lab_a

            loaded_segs.append((seg_name, label, seg_info['length_mm'],
                                seg_info['tortuosity']))

        # ----- 最大截面圈 (默认隐藏, X 键切换) -----
        try:
            _add_max_section_actors(renderer, stl_path, seg_data, nodes,
                                     actors_dict)
        except Exception as e:
            print(f"  最大截面跳过: {e}")
            import traceback
            traceback.print_exc()

        # ----- 分支点 (默认隐藏, B 键切换) -----
        for bp_info in seg_data.get('branch_points', []):
            sp = _build_sphere_actor(bp_info['coord'], 1.5,
                                      color=(0.05, 0.05, 0.05))
            sp.SetVisibility(False)
            renderer.AddActor(sp)
            actors_dict['bp_' + str(bp_info['id'])] = sp

        # ----- 坐标轴 -----
        ax_widget = vtk.vtkOrientationMarkerWidget()
        ax_widget.SetOrientationMarker(vtk.vtkAxesActor())
        ax_widget.SetInteractor(iren)
        ax_widget.SetViewport(0.0, 0.0, 0.15, 0.15)
        ax_widget.EnabledOn()
        ax_widget.InteractiveOff()

        # ----- 标题 -----
        title = vtk.vtkTextActor()
        title_text = (f" Patient: {seg_data['patient_id']}\n"
                      f" Type: {'POST-TIPS' if seg_data['is_post_tips'] else 'PRE-TIPS'}")
        if seg_data.get('has_compensation'):
            title_text += f"  |  Compensation: {seg_data['compensation_type']}"
        title.SetInput(title_text)
        title.GetTextProperty().SetFontSize(15)
        title.GetTextProperty().SetColor(0.10, 0.10, 0.10)
        title.GetTextProperty().SetBold(True)
        title.GetTextProperty().SetFontFamilyToCourier()
        title.SetPosition(10, rw.GetSize()[1] - 50)
        renderer.AddActor2D(title)

        # ----- 段信息列表 -----
        legend_txt = " Loaded segments:\n"
        for nm, lb, L, t in loaded_segs:
            legend_txt += f"  [{lb:4s}] L={L:6.1f}mm  τ={t:.3f}\n"
        legend = vtk.vtkTextActor()
        legend.SetInput(legend_txt)
        legend.GetTextProperty().SetFontSize(12)
        legend.GetTextProperty().SetColor(0.15, 0.15, 0.15)
        legend.GetTextProperty().SetFontFamilyToCourier()
        legend.SetPosition(10, rw.GetSize()[1] - 70 - 18 * len(loaded_segs))
        renderer.AddActor2D(legend)

        # ----- 帮助文字 -----
        help_txt = (" 1-8: toggle segments  M: mesh  C: centerline\n"
                    " L: labels  B: branch points  X: max sections\n"
                    " +/-: opacity  W: wireframe  R: reset  S: shot  Q: quit")
        help_a = vtk.vtkTextActor()
        help_a.SetInput(help_txt)
        help_a.GetTextProperty().SetFontSize(11)
        help_a.GetTextProperty().SetColor(0.30, 0.30, 0.30)
        help_a.GetTextProperty().SetFontFamilyToCourier()
        help_a.SetPosition(10, 40)
        renderer.AddActor2D(help_a)

        # ----- 颜色图例 -----
        for i, (nm, lb, _, _) in enumerate(loaded_segs):
            color = SEGMENT_COLORS[nm]
            t = vtk.vtkTextActor()
            t.SetInput(f"  ■ {lb}")
            t.GetTextProperty().SetFontSize(15)
            t.GetTextProperty().SetColor(color)
            t.GetTextProperty().SetBold(True)
            t.GetTextProperty().SetFontFamilyToCourier()
            t.SetPosition(rw.GetSize()[0] - 130, rw.GetSize()[1] - 80 - 22*i)
            renderer.AddActor2D(t)

        # ----- 交互器 -----
        style = SegmentInteractorStyle(renderer, rw, actors_dict, stl_path)
        iren.SetInteractorStyle(style)

        renderer.ResetCamera()
        cam = renderer.GetActiveCamera()
        cam.Elevation(20)
        cam.Azimuth(30)
        renderer.ResetCameraClippingRange()

        rw.Render()
        print(f"\n  窗口已打开 - 快捷键: 1-8 切换段, M 模型, C 中心线, "
              f"B 分支点, X 截面圈, Q 退出")
        iren.Start()

    if block:
        _run()
    else:
        import threading
        threading.Thread(target=_run, daemon=False).start()


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    visualize_segments(p, block=True)