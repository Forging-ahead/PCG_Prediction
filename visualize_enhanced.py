"""
VTK可视化模块（增强版）
======================
快捷键：
  R: 重置视角        1-6: 预设视图
  O: 优化前(红色)     N: 优化后(绿色)
  M: 血管模型         A: SV-SMV夹角标注
  B: 截平面可视化     D: 解剖分段+名称
  +/-: 透明度         W: 线框/实体
  S: 截图             Q: 退出
"""

import os
import numpy as np
import vtk


# ============================================================
# 交互器
# ============================================================

class EnhancedInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, renderer, render_window, actors_dict, filename=""):
        super().__init__()
        self.renderer = renderer
        self.render_window = render_window
        self.actors = actors_dict
        self.filename = filename
        self.stl_opacity = 0.3

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)

        self.info_actor = vtk.vtkTextActor()
        self.info_actor.GetTextProperty().SetFontSize(14)
        self.info_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.info_actor.SetPosition(10, 120)
        self.info_actor.SetInput("")
        self.renderer.AddActor2D(self.info_actor)

        self.AddObserver("KeyPressEvent", self._on_key_press)
        self.AddObserver("LeftButtonPressEvent", self._on_left_click)

    def _toggle(self, key, label):
        actor = self.actors.get(key)
        if actor:
            vis = actor.GetVisibility()
            actor.SetVisibility(not vis)
            self.info_actor.SetInput(f"{label}: {'Hidden' if vis else 'Visible'}")
            self.render_window.Render()

    def _toggle_group(self, keys, label):
        any_visible = any(
            self.actors.get(k) and self.actors[k].GetVisibility() for k in keys)
        for k in keys:
            a = self.actors.get(k)
            if a:
                a.SetVisibility(not any_visible)
        self.info_actor.SetInput(f"{label}: {'Visible' if not any_visible else 'Hidden'}")
        self.render_window.Render()

    def _on_left_click(self, obj, event):
        pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, self.renderer)
        p = self.picker.GetPickPosition()
        if self.picker.GetCellId() >= 0:
            self.info_actor.SetInput(f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
            self.render_window.Render()
        self.OnLeftButtonDown()

    def _on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        camera = self.renderer.GetActiveCamera()
        focal = camera.GetFocalPoint()
        dist = camera.GetDistance()

        view_map = {
            '1': ((focal[0], focal[1] - dist, focal[2]), (0, 0, 1)),
            '2': ((focal[0], focal[1] + dist, focal[2]), (0, 0, 1)),
            '3': ((focal[0] - dist, focal[1], focal[2]), (0, 0, 1)),
            '4': ((focal[0] + dist, focal[1], focal[2]), (0, 0, 1)),
            '5': ((focal[0], focal[1], focal[2] + dist), (0, 1, 0)),
            '6': ((focal[0], focal[1], focal[2] - dist), (0, 1, 0)),
        }

        if key in ('r', 'R'):
            self.renderer.ResetCamera()
            self.render_window.Render()
        elif key in view_map:
            pos, up = view_map[key]
            camera.SetPosition(pos)
            camera.SetViewUp(up)
            self.renderer.ResetCameraClippingRange()
            self.render_window.Render()
        elif key in ('o', 'O'):
            self._toggle('old', 'Before optimization (Red)')
        elif key in ('n', 'N'):
            self._toggle('new', 'After optimization (Green)')
        elif key in ('m', 'M'):
            self._toggle('stl', 'Vessel model')
        elif key in ('a', 'A'):
            self._toggle('angle_group', 'SV-SMV Angle')
        elif key in ('b', 'B'):
            self._toggle_group(
                [k for k in self.actors if k.startswith('xsec_')],
                'Cross-sections')
        elif key in ('d', 'D'):
            self._toggle_group(
                [k for k in self.actors
                 if k.startswith('seg_')],
                'Anatomy segments')
        elif key in ('plus', 'equal', 'minus'):
            stl = self.actors.get('stl')
            if stl:
                delta = 0.1 if key != 'minus' else -0.1
                self.stl_opacity = max(0.0, min(1.0, self.stl_opacity + delta))
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
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            w2i.Update()
            pdir = os.path.dirname(self.filename) if self.filename else "."
            path = os.path.join(pdir, "centerline_screenshot.png")
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(path)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            print(f"截图: {path}")
            self.render_window.Render()
        elif key in ('q', 'Q'):
            self.render_window.Finalize()
            self.GetInteractor().TerminateApp()


# ============================================================
# 工具函数
# ============================================================

def _build_centerline_actor(filepath, color, line_width=2, point_size=4):
    """从中心线txt构建VTK actor"""
    if not os.path.exists(filepath):
        return None, None, 0

    node_list = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                while len(parts) < 7:
                    parts.append('-1')
                node_list.append([
                    int(parts[0]),
                    float(parts[1]), float(parts[2]), float(parts[3]),
                    int(parts[4]), int(parts[5]), int(parts[6])
                ])

    if not node_list:
        return None, None, 0

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    verts = vtk.vtkCellArray()
    id_to_idx = {}

    for idx, p in enumerate(node_list):
        pid = points.InsertNextPoint(p[1], p[2], p[3])
        id_to_idx[int(p[0])] = idx
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)

    for p in node_list:
        node_idx = id_to_idx[int(p[0])]
        for child_id in [int(p[5]), int(p[6])]:
            if child_id not in (-1, -2) and child_id in id_to_idx:
                ln = vtk.vtkLine()
                ln.GetPointIds().SetId(0, node_idx)
                ln.GetPointIds().SetId(1, id_to_idx[child_id])
                lines.InsertNextCell(ln)

    line_pd = vtk.vtkPolyData()
    line_pd.SetPoints(points)
    line_pd.SetLines(lines)
    line_mapper = vtk.vtkPolyDataMapper()
    line_mapper.SetInputData(line_pd)
    line_actor = vtk.vtkActor()
    line_actor.SetMapper(line_mapper)
    line_actor.GetProperty().SetColor(color)
    line_actor.GetProperty().SetLineWidth(line_width)

    pt_pd = vtk.vtkPolyData()
    pt_pd.SetPoints(points)
    pt_pd.SetVerts(verts)
    pt_mapper = vtk.vtkPolyDataMapper()
    pt_mapper.SetInputData(pt_pd)
    pt_actor = vtk.vtkActor()
    pt_actor.SetMapper(pt_mapper)
    pt_actor.GetProperty().SetColor(color)
    pt_actor.GetProperty().SetPointSize(point_size)

    return line_actor, pt_actor, len(node_list)


def _add_polyline(renderer, pts_list, color, line_width=3, point_size=0):
    """向renderer添加折线"""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for coord in pts_list:
        points.InsertNextPoint(coord)
    for i in range(len(pts_list) - 1):
        ln = vtk.vtkLine()
        ln.GetPointIds().SetId(0, i)
        ln.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(ln)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(line_width)
    renderer.AddActor(actor)
    return actor


# ============================================================
# 截平面可视化构建
# ============================================================

def _build_cross_section_actors(renderer, stl_path, anatomy, nodes,
                                n_samples=8, actors_dict=None):
    """
    沿每条分支等间距取截面轮廓，用 mesh_plane 获取3D交线，
    找到包含中心点的那个环作为VTK线框显示。
    """
    import trimesh as tm
    import trimesh.intersections
    from extract_profiles import _compute_tangents, _make_orthonormal_basis
    from utils import path_to_coords

    try:
        mesh = tm.load(stl_path)
        if not isinstance(mesh, tm.Trimesh):
            if hasattr(mesh, 'geometry'):
                mesh = list(mesh.geometry.values())[0]
            else:
                return
    except Exception as e:
        print(f"  截面可视化: STL加载失败 ({e})")
        return

    branch_defs = {
        'mpv': (anatomy['mpv_path'], (1.0, 0.3, 0.3)),
        'sv':  (anatomy['sv_branch'], (0.3, 0.8, 1.0)),
        'smv': (anatomy['smv_branch'], (1.0, 0.6, 0.2)),
    }

    total_sections = 0

    for bname, (path_ids, color) in branch_defs.items():
        if len(path_ids) < 3:
            continue

        coords = path_to_coords(path_ids, nodes)
        tangents = _compute_tangents(coords)
        M = len(coords)

        indices = np.linspace(1, M - 2, min(n_samples, M - 2), dtype=int)

        for si, idx in enumerate(indices):
            point = coords[idx]
            normal = tangents[idx]

            try:
                lines_3d = trimesh.intersections.mesh_plane(
                    mesh, plane_normal=normal, plane_origin=point)

                if lines_3d is None or len(lines_3d) == 0:
                    continue

                # 找包含中心点的环:
                # 1) 投影到2D找环
                u, v = _make_orthonormal_basis(normal)
                segs_2d = []
                for seg in lines_3d:
                    r0, r1 = seg[0] - point, seg[1] - point
                    segs_2d.append((
                        (float(np.dot(r0, u)), float(np.dot(r0, v))),
                        (float(np.dot(r1, u)), float(np.dot(r1, v)))
                    ))

                from shapely.geometry import LineString, Point as SPoint
                from shapely.ops import polygonize, unary_union

                ls = [LineString([s[0], s[1]]) for s in segs_2d]
                merged = unary_union(ls)
                polys = list(polygonize(merged))

                if not polys:
                    continue

                # 选包含中心的多边形
                center = SPoint(0, 0)
                target_poly = None
                for p in polys:
                    if p.is_valid and p.contains(center):
                        if target_poly is None or p.area < target_poly.area:
                            target_poly = p

                if target_poly is None:
                    valid = [p for p in polys if p.is_valid and p.area > 0]
                    if valid:
                        target_poly = min(valid, key=lambda p: p.distance(center))

                if target_poly is None:
                    continue

                # 把2D轮廓坐标转回3D
                ring_2d = list(target_poly.exterior.coords)
                ring_3d = []
                for x2, y2 in ring_2d:
                    ring_3d.append(point + x2 * u + y2 * v)

                if len(ring_3d) < 3:
                    continue

                # 创建VTK actor
                vtk_pts = vtk.vtkPoints()
                vtk_lines = vtk.vtkCellArray()

                for pt3d in ring_3d:
                    vtk_pts.InsertNextPoint(pt3d)

                for j in range(len(ring_3d)):
                    ln = vtk.vtkLine()
                    ln.GetPointIds().SetId(0, j)
                    ln.GetPointIds().SetId(1, (j + 1) % len(ring_3d))
                    vtk_lines.InsertNextCell(ln)

                pd = vtk.vtkPolyData()
                pd.SetPoints(vtk_pts)
                pd.SetLines(vtk_lines)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(pd)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetLineWidth(2.5)
                actor.GetProperty().SetOpacity(0.8)
                actor.SetVisibility(False)

                renderer.AddActor(actor)
                key = f"xsec_{bname}_{si}_{total_sections}"
                if actors_dict is not None:
                    actors_dict[key] = actor
                total_sections += 1

            except Exception:
                continue

    print(f"  截面可视化: {total_sections} 个截面已加载（按 B 显示）")


# ============================================================
# 主可视化函数
# ============================================================

def visualize_centerline(stl_path, block=True, angle_fit_pts=20, n_sections=25):
    parentdir = os.path.dirname(stl_path)

    def _run():
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.95)

        rw = vtk.vtkRenderWindow()
        rw.SetSize(1200, 900)
        rw.SetWindowName(f"Centerline - {os.path.basename(stl_path)}")
        rw.AddRenderer(renderer)
        rw.SetMultiSamples(4)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)

        actors_dict = {}

        # ===== STL血管模型 =====
        if os.path.exists(stl_path):
            try:
                rd = vtk.vtkSTLReader()
                rd.SetFileName(stl_path)
                rd.Update()
                mp = vtk.vtkPolyDataMapper()
                mp.SetInputConnection(rd.GetOutputPort())
                mp.ScalarVisibilityOff()
                stl_actor = vtk.vtkActor()
                stl_actor.SetMapper(mp)
                stl_actor.GetProperty().SetOpacity(0.3)
                stl_actor.GetProperty().SetColor(0.75, 0.75, 0.85)
                renderer.AddActor(stl_actor)
                actors_dict['stl'] = stl_actor
            except Exception as e:
                print(f"  STL: {e}")

        # ===== 优化前/后中心线 =====
        for fpath, color, lw, ps, key_name, label in [
            (os.path.join(parentdir, "CenterlinePoints.txt"),
             (1.0, 0.2, 0.2), 2, 3, 'old', '优化前(红色 O)'),
            (os.path.join(parentdir, "newCenterlist.txt"),
             (0.1, 0.9, 0.1), 3, 5, 'new', '优化后(绿色 N)'),
        ]:
            la, pa, cnt = _build_centerline_actor(fpath, color, lw, ps)
            if la:
                grp = vtk.vtkAssembly()
                grp.AddPart(la)
                grp.AddPart(pa)
                renderer.AddActor(grp)
                actors_dict[key_name] = grp
                print(f"  {label}: {cnt} 点")

        # ===== 解剖结构（预加载，A/B/D 都需要） =====
        anatomy = None
        e_nodes = None
        try:
            from utils import load_tree, path_to_coords
            from extract_features import identify_anatomy

            e_nodes, e_adj, _ = load_tree(stl_path)
            anatomy = identify_anatomy(e_nodes, e_adj)
        except Exception as e:
            print(f"  解剖识别跳过: {e}")

        # ===== A键: SV-SMV夹角 =====
        if anatomy is not None:
            try:
                from extract_features import _compute_sv_smv_angle
                angle_result, angle_err = _compute_sv_smv_angle(
                    e_nodes, anatomy, angle_fit_pts)

                if angle_result is not None:
                    angle_group = vtk.vtkAssembly()

                    conf = np.array(angle_result['confluence_point_physical'])
                    d1 = np.array(angle_result['branch1_direction'])
                    d2 = np.array(angle_result['branch2_direction'])

                    # 拟合点线
                    for pts, color in [
                        (angle_result.get('_branch1_coords', []), (0, 0.8, 1)),
                        (angle_result.get('_branch2_coords', []), (1, 0.5, 0))
                    ]:
                        if pts and len(pts) >= 2:
                            a = _add_polyline(renderer, pts, color, 5)
                            angle_group.AddPart(a)

                    # 方向边线
                    for d, color in [(d1, (0, 0.8, 1)), (d2, (1, 0.5, 0))]:
                        a = _add_polyline(renderer,
                            [conf.tolist(), (conf + d * 20).tolist()], color, 4)
                        angle_group.AddPart(a)

                    # 弧线
                    omega = np.arccos(np.clip(np.dot(d1, d2), -1, 1))
                    arc_pts = []
                    for i in range(41):
                        t = i / 40
                        if abs(omega) < 1e-8:
                            dd = d1
                        else:
                            dd = (np.sin((1 - t) * omega) * d1 +
                                  np.sin(t * omega) * d2) / np.sin(omega)
                        arc_pts.append((conf + dd * 12.0).tolist())
                    a = _add_polyline(renderer, arc_pts, (1, 1, 0), 4)
                    angle_group.AddPart(a)

                    renderer.AddActor(angle_group)
                    actors_dict['angle_group'] = angle_group

                    # 角度文字
                    txt = vtk.vtkTextActor()
                    txt.SetInput(f"SV-SMV: {angle_result['angle_degrees']:.1f}°")
                    txt.GetTextProperty().SetFontSize(22)
                    txt.GetTextProperty().SetColor(1.0, 0.8, 0.0)
                    txt.GetTextProperty().SetBold(True)
                    txt.GetTextProperty().SetFontFamilyToCourier()
                    c = txt.GetPositionCoordinate()
                    c.SetCoordinateSystemToNormalizedDisplay()
                    c.SetValue(0.50, 0.95)
                    renderer.AddActor2D(txt)

                    print(f"  夹角: {angle_result['angle_degrees']:.1f}° (A键)")
                else:
                    print(f"  夹角跳过: {angle_err}")
            except Exception as e:
                print(f"  夹角跳过: {e}")

        # ===== B键: 截平面可视化 =====
        if anatomy is not None and e_nodes is not None:
            try:
                _build_cross_section_actors(
                    renderer, stl_path, anatomy, e_nodes,
                    n_samples=n_sections, actors_dict=actors_dict)
            except Exception as e:
                print(f"  截面可视化跳过: {e}")

        # ===== D键: 解剖分段 + 名称标签 =====
        if anatomy is not None and e_nodes is not None:
            try:
                from utils import path_to_coords as ptc

                seg_defs = [
                    ('seg_mpv',  anatomy['mpv_path'],   (1, 0, 0),       'MPV'),
                    ('seg_sv',   anatomy['sv_branch'],  (0, 0.8, 1),     'SV'),
                    ('seg_smv',  anatomy['smv_branch'], (1, 0.5, 0),     'SMV'),
                    ('seg_lpv',  anatomy['lpv_branch'], (0.6, 0.3, 1),   'LPV'),
                    ('seg_rpv',  anatomy['rpv_branch'], (0.1, 0.8, 0.3), 'RPV'),
                ]

                for key, path_ids, color, label in seg_defs:
                    if len(path_ids) < 2:
                        continue
                    coords = ptc(path_ids, e_nodes)

                    pts = vtk.vtkPoints()
                    lns = vtk.vtkCellArray()
                    for c_pt in coords:
                        pts.InsertNextPoint(c_pt)
                    for i in range(len(coords) - 1):
                        ln = vtk.vtkLine()
                        ln.GetPointIds().SetId(0, i)
                        ln.GetPointIds().SetId(1, i + 1)
                        lns.InsertNextCell(ln)

                    pd = vtk.vtkPolyData()
                    pd.SetPoints(pts)
                    pd.SetLines(lns)
                    m = vtk.vtkPolyDataMapper()
                    m.SetInputData(pd)
                    act = vtk.vtkActor()
                    act.SetMapper(m)
                    act.GetProperty().SetColor(color)
                    act.GetProperty().SetLineWidth(6)
                    act.SetVisibility(False)
                    renderer.AddActor(act)
                    actors_dict[key] = act

                    # 3D文字标签
                    end_pt = coords[-1] if label != 'MPV' else coords[len(coords)//2]
                    ts = vtk.vtkVectorText()
                    ts.SetText(label)
                    ts.Update()
                    tm = vtk.vtkPolyDataMapper()
                    tm.SetInputConnection(ts.GetOutputPort())
                    ta = vtk.vtkFollower()
                    ta.SetMapper(tm)
                    ta.SetScale(3, 3, 3)
                    ta.SetPosition(end_pt[0]+2, end_pt[1]+2, end_pt[2]+2)
                    ta.GetProperty().SetColor(color)
                    ta.SetCamera(renderer.GetActiveCamera())
                    ta.SetVisibility(False)
                    renderer.AddActor(ta)
                    actors_dict[key + '_label'] = ta

                # 汇合/分叉球标记
                for key, nid, color in [
                    ('seg_conf', anatomy['confluence'], (1, 1, 0)),
                    ('seg_bif',  anatomy['bifurcation'], (1, 0, 1)),
                ]:
                    n = e_nodes[nid]
                    sp = vtk.vtkSphereSource()
                    sp.SetCenter(n['x'], n['y'], n['z'])
                    sp.SetRadius(2.0)
                    sp.SetPhiResolution(16)
                    sp.SetThetaResolution(16)
                    sp.Update()
                    sm = vtk.vtkPolyDataMapper()
                    sm.SetInputConnection(sp.GetOutputPort())
                    sa = vtk.vtkActor()
                    sa.SetMapper(sm)
                    sa.GetProperty().SetColor(color)
                    sa.SetVisibility(False)
                    renderer.AddActor(sa)
                    actors_dict[key] = sa

                print(f"  解剖标注已加载（D键）")
            except Exception as e:
                print(f"  解剖标注跳过: {e}")

        # ===== 坐标轴 =====
        aw = vtk.vtkOrientationMarkerWidget()
        aa = vtk.vtkAxesActor()
        aw.SetOrientationMarker(aa)
        aw.SetInteractor(iren)
        aw.SetViewport(0.0, 0.0, 0.15, 0.15)
        aw.EnabledOn()
        aw.InteractiveOff()

        # ===== 图例 =====
        leg = vtk.vtkTextActor()
        leg.SetInput(
            " O=Before(Red) N=After(Green)\n"
            " M=Vessel  A=Angle  B=Sections\n"
            " D=Anatomy labels\n"
            " +/-=Opacity W=Wire S=Shot Q=Quit")
        leg.GetTextProperty().SetFontSize(13)
        leg.GetTextProperty().SetColor(0.2, 0.2, 0.2)
        leg.GetTextProperty().SetBold(True)
        leg.GetTextProperty().SetFontFamilyToCourier()
        leg.SetPosition(10, 40)
        renderer.AddActor2D(leg)

        # ===== 交互 =====
        style = EnhancedInteractorStyle(
            renderer=renderer, render_window=rw,
            actors_dict=actors_dict, filename=stl_path)
        iren.SetInteractorStyle(style)

        renderer.ResetCamera()
        cam = renderer.GetActiveCamera()
        cam.Elevation(20)
        cam.Azimuth(30)
        renderer.ResetCameraClippingRange()

        rw.Render()
        print(f"\n窗口已打开，快捷键: O/N/M/A/B/D  Q=退出")
        iren.Start()

    if block:
        _run()
    else:
        import threading
        threading.Thread(target=_run, daemon=False).start()


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    visualize_centerline(p, block=True)