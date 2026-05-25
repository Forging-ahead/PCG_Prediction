"""
Local browser workbench for the portal-vein STL pipeline.

Run:
    python web_frontend.py --host 127.0.0.1 --port 8765

The server intentionally uses the standard library so the UI can start before
the scientific pipeline dependencies are installed. Pipeline steps still use
the existing repository modules and will report missing dependencies in the job
log when they are unavailable.
"""

from __future__ import annotations

import argparse
import contextlib
import cgi
import io
import json
import math
import mimetypes
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "web"
RUNS_ROOT = APP_ROOT / "web_runs"
DEFAULT_CONFIG_PATH = APP_ROOT / "web_frontend_config.json"

RUNS_ROOT.mkdir(exist_ok=True)

SEGMENT_COLORS = {
    "mpv": "#ff3333",
    "sv": "#3380ff",
    "smv": "#ff9933",
    "lpv": "#b34dff",
    "rpv": "#33e666",
    "tips": "#00c7c7",
    "lgv": "#d6a800",
    "pgv": "#ff4dee",
}

SEGMENT_LABELS = {
    "mpv": "MPV",
    "sv": "SV",
    "smv": "SMV",
    "lpv": "LPV",
    "rpv": "RPV",
    "tips": "TIPS",
    "lgv": "LGV",
    "pgv": "PGV",
}

PIPELINE_STEPS = [
    "centerline",
    "smooth",
    "segment",
    "profiles",
    "features",
    "export",
]

STEP_LABELS = {
    "centerline": "Centerline extraction",
    "smooth": "Centerline smoothing",
    "segment": "Anatomical segmentation",
    "profiles": "Pointwise cross-sections",
    "features": "Feature extraction",
    "export": "Visualization export",
}

DEFAULT_PARAMS = {
    "pitch": 0.5,
    "min_branch_length_mm": 10.0,
    "min_relative_length": 0.05,
    "min_radius_ratio": 0.4,
    "keep_radius_ratio": 0.55,
    "absolute_min_branch_length_mm": 3.0,
    "absolute_min_radius_mm": 0.5,
    "merge_bp_distance_mm": 5.0,
    "n_fit_points": 10,
    "n_profile_points": 100,
    "curvature_window": 7,
    "sample_step": 3,
    "ownership_factor": 1.8,
    "junction_policy": "min_valid",
    "max_diameter_rate_per_mm": 0.5,
}

OUTPUT_FILES = [
    "CenterlinePoints.txt",
    "newCenterlist.txt",
    "centerline_profiles.json",
    "centerline_pointwise_profiles.json",
    "portal_vein_features.json",
    "unified_features.json",
    "feature_description.json",
    "sv_smv_angle.json",
    "vis_interactive.html",
    "vis_overview.png",
    "centerline_screenshot.png",
    "segment_screenshot.png",
]

STEP_OUTPUTS = {
    "centerline": ["CenterlinePoints.txt"],
    "smooth": ["newCenterlist.txt"],
    "segment": ["centerline_profiles.json"],
    "profiles": ["centerline_pointwise_profiles.json"],
    "features": ["unified_features.json"],
    "export": ["vis_interactive.html"],
}

SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
STATE_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _load_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = APP_ROOT / config_path
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise ValueError(f"Failed to read config {config_path}: {exc}") from exc


def _runtime_info() -> dict:
    return {
        "python": sys.executable,
        "python_prefix": sys.prefix,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or "",
        "conda_prefix": os.environ.get("CONDA_PREFIX") or "",
    }


def _find_conda_exe(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_exe = os.environ.get("CONDA_EXE")
    if env_exe:
        candidates.append(Path(env_exe))
    candidates.append(Path(sys.prefix) / "Scripts" / "conda.exe")
    candidates.append(Path(sys.prefix) / "condabin" / "conda.bat")
    which_conda = shutil.which("conda")
    if which_conda:
        candidates.append(Path(which_conda))
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not locate conda. Pass --conda-exe or start from an Anaconda prompt."
    )


def _maybe_reexec_in_conda(args, argv: list[str]) -> None:
    requested = (args.conda_env or "").strip()
    if not requested or args.no_conda_reexec:
        return
    current = os.environ.get("CONDA_DEFAULT_ENV") or ""
    if current == requested:
        return

    conda_exe = _find_conda_exe(args.conda_exe)
    script = str(Path(__file__).resolve())
    forwarded = []
    skip_next = False
    consumed_with_value = {"--conda-env", "--conda-exe", "--config", "--host", "--port"}
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in consumed_with_value:
            skip_next = True
            continue
        if any(item.startswith(flag + "=") for flag in consumed_with_value):
            continue
        if item == "--no-conda-reexec":
            continue
        forwarded.append(item)

    command = [
        conda_exe,
        "run",
        "-n",
        requested,
        "python",
        script,
        "--no-conda-reexec",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.conda_exe:
        command.extend(["--conda-exe", str(args.conda_exe)])
    command.extend(forwarded)

    if getattr(sys, "stdout", None) is not None:
        print(f"Restarting web frontend in conda env: {requested}")
    raise SystemExit(subprocess.call(command, cwd=str(APP_ROOT)))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _sanitize_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json_file(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_invalid_folder(folder_name: str) -> bool:
    return "@" in folder_name or "!" in folder_name


def _is_post_tips(folder_name: str) -> bool:
    return "#" in folder_name


def _safe_float(value, default=None):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _merge_params(user_params: dict | None) -> dict:
    params = dict(DEFAULT_PARAMS)
    if not isinstance(user_params, dict):
        return params
    for key, default in DEFAULT_PARAMS.items():
        if key not in user_params:
            continue
        if isinstance(default, int) and not isinstance(default, bool):
            params[key] = _safe_int(user_params[key], default)
        elif isinstance(default, float):
            params[key] = _safe_float(user_params[key], default)
        else:
            params[key] = user_params[key]
    return params


def _new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _patient_record(stl_path: Path) -> dict:
    folder = stl_path.parent
    return {
        "id": folder.name or stl_path.stem,
        "folder": str(folder),
        "stl_path": str(stl_path),
        "stl_name": stl_path.name,
        "is_post_tips": _is_post_tips(folder.name),
    }


def _discover_batch(root_folder: Path, stl_name: str) -> list[dict]:
    patients: list[dict] = []
    if not root_folder.exists():
        return patients
    direct = root_folder / stl_name
    if direct.exists():
        patients.append(_patient_record(direct))
    for child in sorted(root_folder.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or _is_invalid_folder(child.name):
            continue
        stl = child / stl_name
        if stl.exists():
            patients.append(_patient_record(stl))
    return patients


def _resolve_patient(session: dict, patient_id: str | None) -> dict | None:
    patients = session.get("patients") or []
    if not patients:
        return None
    if not patient_id or patient_id == "first":
        return patients[0]
    for patient in patients:
        if patient.get("id") == patient_id:
            return patient
    return patients[0]


def _read_centerline_file(path: Path):
    if not path.exists():
        return None
    nodes = {}
    try:
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                while len(parts) < 7:
                    parts.append("-1")
                nid = int(float(parts[0]))
                nodes[nid] = {
                    "id": nid,
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                    "parent": int(float(parts[4])),
                    "left": int(float(parts[5])),
                    "right": int(float(parts[6])),
                }
    except Exception:
        return None
    return nodes


def _line_arrays_from_nodes(nodes: dict | None) -> dict | None:
    if not nodes:
        return None
    x, y, z = [], [], []
    seen = set()
    for nid, node in nodes.items():
        for nb in (node.get("parent"), node.get("left"), node.get("right")):
            if nb is None or nb < 0 or nb not in nodes:
                continue
            edge = tuple(sorted((nid, nb)))
            if edge in seen:
                continue
            seen.add(edge)
            other = nodes[nb]
            x.extend([node["x"], other["x"], None])
            y.extend([node["y"], other["y"], None])
            z.extend([node["z"], other["z"], None])
    return {"x": x, "y": y, "z": z, "n_nodes": len(nodes), "n_edges": len(seen)}


def _centerline_adjacency(nodes: dict) -> dict[int, set[int]]:
    adj = {int(nid): set() for nid in nodes}
    for nid, node in nodes.items():
        nid = int(nid)
        for nb in (node.get("parent"), node.get("left"), node.get("right")):
            if nb is None or nb < 0 or nb not in nodes:
                continue
            nb = int(nb)
            adj[nid].add(nb)
            adj[nb].add(nid)
    return adj


def _path_length_from_nodes(path: list[int], nodes: dict) -> float:
    if len(path) < 2:
        return 0.0
    coords = np.asarray([
        [nodes[nid]["x"], nodes[nid]["y"], nodes[nid]["z"]]
        for nid in path if nid in nodes
    ], dtype=float)
    if len(coords) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _editable_centerline_branches(nodes: dict | None) -> list[dict]:
    """Return terminal endpoint-to-branchpoint paths that may be removed."""
    if not nodes:
        return []
    adj = _centerline_adjacency(nodes)
    endpoints = [nid for nid, nbs in adj.items() if len(nbs) == 1]
    out = []
    seen = set()

    for endpoint in endpoints:
        path = [endpoint]
        prev = None
        cur = endpoint
        while True:
            neighbors = [n for n in adj[cur] if n != prev]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            path.append(nxt)
            degree = len(adj[nxt])
            if degree != 2:
                if degree >= 3:
                    endpoint_to_junction = path
                    junction_to_endpoint = list(reversed(endpoint_to_junction))
                    key = (endpoint, nxt)
                    if key in seen:
                        break
                    seen.add(key)
                    coords = _coords_for_path(junction_to_endpoint, nodes)
                    if coords is None:
                        break
                    branch_id = f"{endpoint}:{nxt}"
                    out.append({
                        "id": branch_id,
                        "endpoint_id": int(endpoint),
                        "junction_id": int(nxt),
                        "path": [int(n) for n in junction_to_endpoint],
                        "x": coords[:, 0].tolist(),
                        "y": coords[:, 1].tolist(),
                        "z": coords[:, 2].tolist(),
                        "length_mm": _path_length_from_nodes(endpoint_to_junction, nodes),
                        "n_points": len(endpoint_to_junction),
                    })
                break
            prev = cur
            cur = nxt

    out.sort(key=lambda item: item["length_mm"], reverse=True)
    return out


def _rebuild_centerline_tree(nodes: dict, adj: dict[int, set[int]]) -> list[list[float | int]]:
    remaining = sorted(nid for nid in nodes if nid in adj)
    if not remaining:
        raise ValueError("Centerline would be empty after deletion.")

    root_candidates = [nid for nid in remaining if nodes[nid].get("parent") == -1]
    if root_candidates:
        root = root_candidates[0]
    else:
        endpoints = [nid for nid in remaining if len(adj.get(nid, set())) <= 1]
        root = endpoints[0] if endpoints else remaining[0]

    visited = {root}
    queue = deque([(root, -1)])
    bfs_order = []
    while queue:
        node, parent = queue.popleft()
        bfs_order.append((node, parent))
        for nb in sorted(adj.get(node, set())):
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, node))

    if len(visited) != len(remaining):
        largest = _largest_component_nodes(adj)
        dropped = set(remaining) - largest
        if not largest:
            raise ValueError("Centerline has no connected component after deletion.")
        root = root if root in largest else sorted(largest)[0]
        visited = {root}
        queue = deque([(root, -1)])
        bfs_order = []
        while queue:
            node, parent = queue.popleft()
            bfs_order.append((node, parent))
            for nb in sorted(adj.get(node, set())):
                if nb in visited or nb not in largest:
                    continue
                visited.add(nb)
                queue.append((nb, node))
        if dropped:
            print(f"       [warn] centerline edit dropped disconnected nodes: {len(dropped)}")

    old_to_new = {old: idx for idx, (old, _) in enumerate(bfs_order)}
    children = {old: [] for old, _ in bfs_order}
    for old, parent in bfs_order:
        if parent != -1 and parent in children:
            children[parent].append(old)

    tree = []
    for old, parent in bfs_order:
        node = nodes[old]
        child_ids = children[old]
        tree.append([
            old_to_new[old],
            float(node["x"]),
            float(node["y"]),
            float(node["z"]),
            old_to_new[parent] if parent != -1 else -1,
            old_to_new[child_ids[0]] if len(child_ids) >= 1 else -1,
            old_to_new[child_ids[1]] if len(child_ids) >= 2 else -1,
        ])
    return tree


def _largest_component_nodes(adj: dict[int, set[int]]) -> set[int]:
    remaining = set(adj)
    best = set()
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        queue = deque([start])
        remaining.discard(start)
        while queue:
            cur = queue.popleft()
            for nb in adj.get(cur, set()):
                if nb not in remaining:
                    continue
                remaining.discard(nb)
                comp.add(nb)
                queue.append(nb)
        if len(comp) > len(best):
            best = comp
    return best


def _write_centerline_tree(path: Path, tree: list[list[float | int]]):
    with path.open("w", encoding="utf-8") as f:
        for row in tree:
            f.write(" ".join(str(v) for v in row) + "\n")


def delete_centerline_terminal_branches(stl_path: Path, branch_ids: list[str]) -> dict:
    parent = stl_path.parent
    centerline_path = parent / "CenterlinePoints.txt"
    nodes = _read_centerline_file(centerline_path)
    if not nodes:
        raise ValueError("CenterlinePoints.txt not found or empty.")

    requested = {str(item) for item in branch_ids if str(item)}
    editable = {item["id"]: item for item in _editable_centerline_branches(nodes)}
    invalid = sorted(requested - set(editable))
    if invalid:
        raise ValueError(f"Only endpoint-to-branchpoint branches can be deleted. Invalid: {', '.join(invalid)}")
    if not requested:
        return {"deleted": [], "remaining_branches": list(editable.values()), "removed_nodes": 0}

    remove_nodes = set()
    deleted = []
    for branch_id in sorted(requested):
        item = editable[branch_id]
        path = list(reversed(item["path"]))  # endpoint -> junction
        junction = item["junction_id"]
        remove_nodes.update(n for n in path if n != junction)
        deleted.append(item)

    kept_nodes = {nid: node for nid, node in nodes.items() if nid not in remove_nodes}
    if len(kept_nodes) < 2:
        raise ValueError("Cannot delete branches because the centerline would become too small.")

    adj = _centerline_adjacency(kept_nodes)
    tree = _rebuild_centerline_tree(kept_nodes, adj)
    _write_centerline_tree(centerline_path, tree)

    # Downstream files are derived from the old raw centerline. Remove them so
    # smoothing/segmentation/features recompute from the edited raw tree.
    stale_outputs = [
        "newCenterlist.txt",
        "centerline_profiles.json",
        "centerline_pointwise_profiles.json",
        "portal_vein_features.json",
        "unified_features.json",
        "feature_description.json",
        "sv_smv_angle.json",
        "vis_interactive.html",
        "vis_overview.png",
        "centerline_screenshot.png",
        "segment_screenshot.png",
    ]
    removed_outputs = []
    for name in stale_outputs:
        p = parent / name
        if p.exists():
            try:
                p.unlink()
                removed_outputs.append(name)
            except Exception:
                pass

    new_nodes = _read_centerline_file(centerline_path)
    return {
        "deleted": deleted,
        "removed_nodes": len(remove_nodes),
        "removed_outputs": removed_outputs,
        "remaining_branches": _editable_centerline_branches(new_nodes),
    }


def _coords_for_path(path: list[int], nodes: dict) -> np.ndarray | None:
    coords = []
    for nid in path:
        if nid not in nodes:
            return None
        n = nodes[nid]
        coords.append([n["x"], n["y"], n["z"]])
    if len(coords) < 2:
        return None
    return np.asarray(coords, dtype=float)


def _point_and_tangent_at_fraction(coords: np.ndarray, frac: float):
    frac = min(1.0, max(0.0, float(frac)))
    if len(coords) < 2:
        return None, None
    seg_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(arc[-1])
    if total <= 1e-9:
        return coords[0], np.array([0.0, 0.0, 1.0])
    target = total * frac
    idx = int(np.searchsorted(arc, target) - 1)
    idx = max(0, min(len(coords) - 2, idx))
    a0, a1 = arc[idx], arc[idx + 1]
    local = (target - a0) / (a1 - a0) if a1 > a0 else 0.0
    point = coords[idx] + local * (coords[idx + 1] - coords[idx])
    lo = max(0, idx - 1)
    hi = min(len(coords) - 1, idx + 2)
    tangent = coords[hi] - coords[lo]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        tangent = np.array([0.0, 0.0, 1.0])
    else:
        tangent = tangent / norm
    return point, tangent


def _basis_from_normal(normal: np.ndarray):
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-15)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-15)
    return u, v


def _circle_arrays(point: np.ndarray, normal: np.ndarray, radius: float, n_pts: int = 40):
    if radius is None or not math.isfinite(radius) or radius <= 0:
        return None
    u, v = _basis_from_normal(normal)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    pts = np.asarray([point + radius * (math.cos(t) * u + math.sin(t) * v) for t in theta])
    return pts


def _finite_array(values):
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr


def _profile_positions(profile: dict) -> np.ndarray:
    n = len(profile.get("position") or profile.get("area") or [])
    if n <= 1:
        return np.asarray([], dtype=float)
    pos = profile.get("position")
    if pos and len(pos) == n:
        return np.asarray(pos, dtype=float)
    return np.linspace(0.0, 1.0, n)


def _build_segments(seg_data: dict | None, nodes: dict | None) -> dict:
    out = {}
    if not seg_data or not nodes:
        return out
    for name, info in (seg_data.get("segments") or {}).items():
        if not info:
            continue
        coords = _coords_for_path(info.get("path", []), nodes)
        if coords is None:
            continue
        label = SEGMENT_LABELS.get(name, name.upper())
        out[name] = {
            "label": label,
            "color": SEGMENT_COLORS.get(name, "#888888"),
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "z": coords[:, 2].tolist(),
            "midpoint": coords[len(coords) // 2].tolist(),
            "length_mm": info.get("length_mm"),
            "tortuosity": info.get("tortuosity"),
            "mean_curvature": info.get("mean_curvature"),
            "n_points": info.get("n_points", len(coords)),
            "path": info.get("path", []),
        }
    return out


def _valid_numeric_at(profile: dict, key: str, idx: int):
    values = profile.get(key)
    if not values or idx >= len(values):
        return None
    value = _safe_float(values[idx])
    return value


def _build_pointwise_layers(seg_data: dict | None, nodes: dict | None, pointwise: dict | None, section_stride: int):
    features = {}
    sampled_sections = {}
    max_sections = {}
    mean_sections = {}
    if not seg_data or not nodes or not pointwise:
        return {
            "feature_points": features,
            "sampled_sections": sampled_sections,
            "max_sections": max_sections,
            "mean_sections": mean_sections,
        }
    section_stride = max(1, int(section_stride or 10))

    for seg_name, seg_info in (seg_data.get("segments") or {}).items():
        profile = pointwise.get(seg_name)
        if not seg_info or not profile:
            continue
        coords = _coords_for_path(seg_info.get("path", []), nodes)
        if coords is None:
            continue
        positions = _profile_positions(profile)
        if len(positions) == 0:
            continue

        fx, fy, fz = [], [], []
        curvature_values, sizes, hover = [], [], []
        ring_x, ring_y, ring_z = [], [], []
        area = _finite_array(profile.get("area"))
        diameter = _finite_array(profile.get("eq_diameter"))
        curvature = _finite_array(profile.get("curvature"))
        circularity = _finite_array(profile.get("circularity"))
        inscribed = _finite_array(profile.get("inscribed_radius"))

        for i, frac in enumerate(positions):
            point, tangent = _point_and_tangent_at_fraction(coords, frac)
            if point is None:
                continue
            curv = _valid_numeric_at(profile, "curvature", i)
            dia = _valid_numeric_at(profile, "eq_diameter", i)
            ar = _valid_numeric_at(profile, "area", i)
            circ = _valid_numeric_at(profile, "circularity", i)
            ins = _valid_numeric_at(profile, "inscribed_radius", i)
            if curv is not None or dia is not None or ar is not None:
                fx.append(float(point[0]))
                fy.append(float(point[1]))
                fz.append(float(point[2]))
                curvature_values.append(curv if curv is not None else 0.0)
                sizes.append(max(4.0, min(15.0, 3.0 + (dia or 0.0) * 0.45)))
                hover.append(
                    f"{SEGMENT_LABELS.get(seg_name, seg_name.upper())}<br>"
                    f"point: {i}<br>"
                    f"curvature: {_format_metric(curv, 5)} 1/mm<br>"
                    f"diameter: {_format_metric(dia, 3)} mm<br>"
                    f"area: {_format_metric(ar, 3)} mm^2<br>"
                    f"circularity: {_format_metric(circ, 3)}<br>"
                    f"inscribed radius: {_format_metric(ins, 3)} mm"
                )
            if i % section_stride == 0:
                dia = _valid_numeric_at(profile, "eq_diameter", i)
                if dia is None or dia <= 0:
                    continue
                circle = _circle_arrays(point, tangent, dia / 2.0, n_pts=36)
                if circle is None:
                    continue
                ring_x.extend(circle[:, 0].tolist() + [None])
                ring_y.extend(circle[:, 1].tolist() + [None])
                ring_z.extend(circle[:, 2].tolist() + [None])

        features[seg_name] = {
            "label": SEGMENT_LABELS.get(seg_name, seg_name.upper()),
            "color": SEGMENT_COLORS.get(seg_name, "#888888"),
            "x": fx,
            "y": fy,
            "z": fz,
            "curvature": curvature_values,
            "size": sizes,
            "hover": hover,
        }
        if ring_x:
            sampled_sections[seg_name] = {
                "label": SEGMENT_LABELS.get(seg_name, seg_name.upper()),
                "color": SEGMENT_COLORS.get(seg_name, "#888888"),
                "x": ring_x,
                "y": ring_y,
                "z": ring_z,
            }

        max_idx = _best_index(area, mode="max")
        mean_idx = _best_index(area, mode="mean")
        max_ring = _section_at_index(coords, profile, max_idx)
        mean_ring = _section_at_index(coords, profile, mean_idx)
        if max_ring:
            max_sections[seg_name] = max_ring
        if mean_ring:
            mean_sections[seg_name] = mean_ring

    return {
        "feature_points": features,
        "sampled_sections": sampled_sections,
        "max_sections": max_sections,
        "mean_sections": mean_sections,
    }


def _best_index(arr: np.ndarray, mode: str):
    if arr.size == 0:
        return None
    valid = np.isfinite(arr) & (arr > 0)
    if not np.any(valid):
        return None
    if mode == "max":
        masked = np.where(valid, arr, -np.inf)
        return int(np.argmax(masked))
    mean_val = float(np.nanmean(arr[valid]))
    dist = np.where(valid, np.abs(arr - mean_val), np.inf)
    return int(np.argmin(dist))


def _section_at_index(coords: np.ndarray, profile: dict, idx: int | None):
    if idx is None:
        return None
    positions = _profile_positions(profile)
    if idx < 0 or idx >= len(positions):
        return None
    dia = _valid_numeric_at(profile, "eq_diameter", idx)
    area = _valid_numeric_at(profile, "area", idx)
    if dia is None or dia <= 0:
        return None
    point, tangent = _point_and_tangent_at_fraction(coords, float(positions[idx]))
    if point is None:
        return None
    circle = _circle_arrays(point, tangent, dia / 2.0, n_pts=48)
    if circle is None:
        return None
    return {
        "x": circle[:, 0].tolist(),
        "y": circle[:, 1].tolist(),
        "z": circle[:, 2].tolist(),
        "index": int(idx),
        "diameter": dia,
        "area": area,
    }


def _format_metric(value, digits=3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _load_mesh(stl_path: Path, max_faces: int = 80000) -> dict | None:
    try:
        return _load_mesh_with_trimesh(stl_path, max_faces=max_faces)
    except Exception:
        return _load_mesh_fallback(stl_path, max_faces=max_faces)


def _load_mesh_with_trimesh(stl_path: Path, max_faces: int = 80000) -> dict | None:
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return None
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return _compact_sampled_mesh(vertices, faces, max_faces=max_faces, source="trimesh")


def _load_mesh_fallback(stl_path: Path, max_faces: int = 80000) -> dict | None:
    try:
        with stl_path.open("rb") as f:
            header = f.read(80)
            n_raw = f.read(4)
            if len(n_raw) == 4:
                n_faces = struct.unpack("<I", n_raw)[0]
                expected = 84 + n_faces * 50
                actual = stl_path.stat().st_size
                if n_faces > 0 and expected == actual:
                    return _read_binary_stl_sampled(f, n_faces, max_faces=max_faces)
        return _read_ascii_stl_sampled(stl_path, max_faces=max_faces)
    except Exception:
        return None


def _read_binary_stl_sampled(handle, n_faces: int, max_faces: int):
    stride = max(1, int(math.ceil(n_faces / max_faces)))
    vertices = []
    faces = []
    face_idx = 0
    for i in range(n_faces):
        chunk = handle.read(50)
        if len(chunk) < 50:
            break
        if i % stride != 0:
            continue
        vals = struct.unpack("<12fH", chunk)
        tri = vals[3:12]
        base = len(vertices)
        vertices.extend([
            [tri[0], tri[1], tri[2]],
            [tri[3], tri[4], tri[5]],
            [tri[6], tri[7], tri[8]],
        ])
        faces.append([base, base + 1, base + 2])
        face_idx += 1
    return {
        "vertices": vertices,
        "faces": faces,
        "n_faces": int(n_faces),
        "n_faces_rendered": int(face_idx),
        "source": "stdlib-binary-stl",
    }


def _read_ascii_stl_sampled(stl_path: Path, max_faces: int):
    tris = []
    current = []
    with stl_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("vertex"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(current) == 3:
                tris.append(current)
                current = []
    if not tris:
        return None
    stride = max(1, int(math.ceil(len(tris) / max_faces)))
    vertices = []
    faces = []
    for tri in tris[::stride]:
        base = len(vertices)
        vertices.extend(tri)
        faces.append([base, base + 1, base + 2])
    return {
        "vertices": vertices,
        "faces": faces,
        "n_faces": int(len(tris)),
        "n_faces_rendered": int(len(faces)),
        "source": "stdlib-ascii-stl",
    }


def _compact_sampled_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int, source: str):
    n_faces = int(len(faces))
    if n_faces > max_faces:
        stride = int(math.ceil(n_faces / max_faces))
        faces = faces[::stride]
    used = np.unique(faces.reshape(-1))
    remap = {int(old): i for i, old in enumerate(used)}
    compact_vertices = vertices[used]
    compact_faces = np.asarray([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces], dtype=np.int64)
    return {
        "vertices": np.round(compact_vertices, 5).tolist(),
        "faces": compact_faces.tolist(),
        "n_faces": n_faces,
        "n_faces_rendered": int(len(compact_faces)),
        "source": source,
    }


def _load_feature_blocks(parent: Path):
    unified = _read_json_file(parent / "unified_features.json")
    flat = _read_json_file(parent / "portal_vein_features.json")
    if unified:
        return {
            "source": "unified_features.json",
            "meta": unified.get("_meta", {}),
            "vessel_presence": unified.get("vessel_presence", {}),
            "statistical": unified.get("statistical", {}),
            "system": unified.get("system", {}),
            "global": unified.get("global", {}),
            "segments_meta": unified.get("segments_meta", {}),
            "pointwise_meta": unified.get("pointwise_meta", {}),
        }
    if flat:
        statistical = {}
        for key, value in flat.items():
            if "_" not in key or key.startswith("_"):
                continue
            seg, feature = key.split("_", 1)
            if seg in SEGMENT_LABELS:
                statistical.setdefault(seg, {})[feature] = value
        return {
            "source": "portal_vein_features.json",
            "meta": flat.get("_meta", {}),
            "vessel_presence": {},
            "statistical": statistical,
            "system": {},
            "global": {
                k: flat.get(k)
                for k in (
                    "total_centerline_length",
                    "sv_smv_diameter_ratio",
                    "sv_smv_angle",
                    "has_lgv",
                    "has_pgv",
                    "has_compensation_vessel",
                    "has_tips",
                )
                if k in flat
            },
            "segments_meta": {},
            "pointwise_meta": {},
        }
    return {
        "source": None,
        "meta": {},
        "vessel_presence": {},
        "statistical": {},
        "system": {},
        "global": {},
        "segments_meta": {},
        "pointwise_meta": {},
    }


def build_visualization_data(stl_path: Path, section_stride: int = 10, max_faces: int = 80000) -> dict:
    parent = stl_path.parent
    raw_nodes = _read_centerline_file(parent / "CenterlinePoints.txt")
    smooth_nodes = _read_centerline_file(parent / "newCenterlist.txt")
    nodes = smooth_nodes or raw_nodes
    seg_data = _read_json_file(parent / "centerline_profiles.json")
    pointwise = _read_json_file(parent / "centerline_pointwise_profiles.json")

    pointwise_layers = _build_pointwise_layers(seg_data, nodes, pointwise, section_stride)
    branch_points = []
    if seg_data:
        for bp in seg_data.get("branch_points", []):
            if isinstance(bp, dict) and "coord" in bp:
                branch_points.append(bp)

    return _sanitize_json({
        "patient": _patient_record(stl_path),
        "mesh": _load_mesh(stl_path, max_faces=max_faces),
        "centerlines": {
            "raw": _line_arrays_from_nodes(raw_nodes),
            "smooth": _line_arrays_from_nodes(smooth_nodes),
        },
        "centerline_edit": {
            "branches": _editable_centerline_branches(raw_nodes),
        },
        "segments": _build_segments(seg_data, nodes),
        "branch_points": branch_points,
        "pointwise": pointwise_layers,
        "features": _load_feature_blocks(parent),
        "files": _available_outputs(parent),
        "step_files": _step_file_status(parent),
    })


def _available_outputs(parent: Path) -> list[dict]:
    files = []
    for name in OUTPUT_FILES:
        p = parent / name
        if p.exists():
            files.append({
                "name": name,
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return files


def _step_file_status(parent: Path) -> dict:
    status = {}
    for step, names in STEP_OUTPUTS.items():
        files = []
        for name in names:
            p = parent / name
            files.append({
                "name": name,
                "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else 0,
                "modified": p.stat().st_mtime if p.exists() else None,
            })
        status[step] = {
            "ready": all(item["exists"] for item in files),
            "files": files,
        }
    return status


def _reuse_pipeline_step(step: str, stl_path: Path):
    required = STEP_OUTPUTS.get(step) or []
    missing = [name for name in required if not (stl_path.parent / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot import saved result for {STEP_LABELS.get(step, step)}; "
            f"missing: {', '.join(missing)}"
        )
    print(f"Reused saved result for {STEP_LABELS.get(step, step)}: {', '.join(required)}")


def _create_session_single(fields) -> dict:
    session_id = _new_session_id()
    file_item = fields["stl_file"] if "stl_file" in fields else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError("Missing STL file.")
    original_name = Path(file_item.filename).name or "vessel.stl"
    if not original_name.lower().endswith(".stl"):
        original_name += ".stl"
    output_dir = None
    if "output_dir" in fields:
        raw_output = str(fields.getvalue("output_dir") or "").strip()
        if raw_output:
            output_dir = Path(raw_output)
    if output_dir:
        patient_dir = output_dir
        patient_dir.mkdir(parents=True, exist_ok=True)
    else:
        patient_dir = RUNS_ROOT / session_id / Path(original_name).stem
        patient_dir.mkdir(parents=True, exist_ok=True)
    stl_path = patient_dir / original_name
    with stl_path.open("wb") as out:
        shutil.copyfileobj(file_item.file, out)
    session = {
            "id": session_id,
            "mode": "single",
            "created": _now(),
            "root": str(patient_dir),
            "patients": [_patient_record(stl_path)],
            "params": dict(DEFAULT_PARAMS),
            "runtime": _runtime_info(),
        }
    with STATE_LOCK:
        SESSIONS[session_id] = session
    return session


def _create_session_batch(payload: dict) -> dict:
    root = Path(str(payload.get("root_folder") or "").strip())
    stl_name = str(payload.get("stl_name") or "vessel.stl").strip() or "vessel.stl"
    if not root.exists():
        raise ValueError(f"Folder does not exist: {root}")
    patients = _discover_batch(root, stl_name)
    if not patients:
        raise ValueError(f"No {stl_name} files found under {root}")
    session_id = _new_session_id()
    session = {
        "id": session_id,
        "mode": "batch",
        "created": _now(),
        "root": str(root),
        "stl_name": stl_name,
        "patients": patients,
        "params": dict(DEFAULT_PARAMS),
        "runtime": _runtime_info(),
    }
    with STATE_LOCK:
        SESSIONS[session_id] = session
    return session


def _new_job(session_id: str, steps: list[str], patients: list[dict], step_modes: dict | None = None) -> dict:
    job_id = uuid.uuid4().hex[:12]
    total = max(1, len(steps) * len(patients))
    job = {
        "id": job_id,
        "session_id": session_id,
        "status": "running",
        "created": _now(),
        "updated": _now(),
        "steps": steps,
        "step_modes": step_modes or {},
        "total": total,
        "completed": 0,
        "current": "",
        "logs": [],
        "errors": [],
        "results": {},
    }
    with STATE_LOCK:
        JOBS[job_id] = job
    return job


def _append_job_log(job: dict, message: str):
    with STATE_LOCK:
        job["logs"].append(message)
        job["logs"] = job["logs"][-500:]
        job["updated"] = _now()


def _set_job_progress(job: dict, current: str | None = None, completed_delta: int = 0):
    with STATE_LOCK:
        if current is not None:
            job["current"] = current
        job["completed"] += completed_delta
        job["updated"] = _now()


def _run_job(job_id: str, params: dict, post_tips_mode: str, export_png: bool):
    with STATE_LOCK:
        job = JOBS[job_id]
        session = SESSIONS[job["session_id"]]
        patients = list(job.get("_patients_runtime", []))
        steps = list(job["steps"])
        step_modes = dict(job.get("step_modes") or {})
        job.pop("_patients_runtime", None)
    try:
        for patient in patients:
            stl_path = Path(patient["stl_path"])
            for step in steps:
                label = STEP_LABELS.get(step, step)
                _set_job_progress(job, current=f"{patient['id']} - {label}")
                buffer = io.StringIO()
                started = time.time()
                ok = True
                err_msg = None
                try:
                    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                        if step_modes.get(step) == "reuse":
                            _reuse_pipeline_step(step, stl_path)
                        else:
                            _run_pipeline_step(step, stl_path, params, post_tips_mode, export_png)
                except Exception as exc:
                    ok = False
                    err_msg = f"{type(exc).__name__}: {exc}"
                    buffer.write("\n")
                    buffer.write(traceback.format_exc())
                elapsed = time.time() - started
                text = buffer.getvalue().strip()
                status_line = f"[{'OK' if ok else 'FAIL'}] {patient['id']} / {label} ({elapsed:.1f}s)"
                if err_msg:
                    status_line += f" - {err_msg}"
                _append_job_log(job, status_line)
                if text:
                    _append_job_log(job, text)
                with STATE_LOCK:
                    job["results"].setdefault(patient["id"], {})[step] = ok
                    if not ok:
                        job["errors"].append(status_line)
                _set_job_progress(job, completed_delta=1)
        with STATE_LOCK:
            job["status"] = "failed" if job["errors"] else "done"
            job["current"] = ""
            job["updated"] = _now()
    except Exception as exc:
        with STATE_LOCK:
            job["status"] = "failed"
            job["errors"].append(f"{type(exc).__name__}: {exc}")
            job["logs"].append(traceback.format_exc())
            job["updated"] = _now()


def _run_pipeline_step(step: str, stl_path: Path, params: dict, post_tips_mode: str, export_png: bool):
    if step == "centerline":
        from extract_centerline import extract_centerline

        extract_centerline(
            str(stl_path),
            pitch=params["pitch"],
            min_branch_length_mm=params["min_branch_length_mm"],
            min_relative_length=params["min_relative_length"],
            min_radius_ratio=params["min_radius_ratio"],
            keep_radius_ratio=params["keep_radius_ratio"],
            absolute_min_branch_length_mm=params["absolute_min_branch_length_mm"],
            absolute_min_radius_mm=params["absolute_min_radius_mm"],
            merge_bp_distance_mm=params["merge_bp_distance_mm"],
        )
    elif step == "smooth":
        from smooth_centerline import smooth_centerline

        smooth_centerline(str(stl_path))
    elif step == "segment":
        from segment_vessels import segment_vessels

        segment_vessels(str(stl_path), post_tips=_post_tips_value(stl_path, post_tips_mode))
    elif step == "profiles":
        from extract_profiles import extract_profiles

        extract_profiles(
            str(stl_path),
            n_points=params["n_profile_points"],
            pitch=params["pitch"],
            curvature_window=params["curvature_window"],
            section_step=params["sample_step"],
            ownership_factor=params["ownership_factor"],
            junction_policy=params["junction_policy"],
            max_diameter_rate_per_mm=params["max_diameter_rate_per_mm"],
        )
    elif step == "features":
        from extract_features import extract_all_features

        extract_all_features(
            str(stl_path),
            n_fit_points=params["n_fit_points"],
            curvature_window=params["curvature_window"],
            sample_step=params["sample_step"],
            pitch=params["pitch"],
        )
    elif step == "export":
        from export_visualization import export_patient_visualization

        export_patient_visualization(str(stl_path), export_html=True, export_png=export_png, verbose=True)
    else:
        raise ValueError(f"Unknown step: {step}")


def _post_tips_value(stl_path: Path, mode: str):
    if mode == "pre":
        return False
    if mode == "post":
        return True
    return _is_post_tips(stl_path.parent.name)


def _zip_patient_outputs(patients: list[dict]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for patient in patients:
            stl_path = Path(patient["stl_path"])
            parent = stl_path.parent
            prefix = patient["id"]
            if stl_path.exists():
                zf.write(stl_path, f"{prefix}/{stl_path.name}")
            for name in OUTPUT_FILES:
                p = parent / name
                if p.exists():
                    zf.write(p, f"{prefix}/{name}")
    return bio.getvalue()


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "PPGWorkbench/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._send_json({"ok": True, "time": _now(), "runtime": _runtime_info()})
            elif path.startswith("/api/session/") and path.endswith("/data"):
                self._handle_session_data(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/download"):
                self._handle_download(path, parsed.query)
            elif path.startswith("/api/job/"):
                self._handle_job(path)
            elif path == "/assets/plotly.min.js":
                self._serve_plotly()
            else:
                self._serve_static(path)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/session":
                self._handle_create_session()
            elif parsed.path == "/api/run":
                self._handle_run()
            elif parsed.path == "/api/centerline/delete-branches":
                self._handle_delete_centerline_branches()
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def log_message(self, fmt, *args):
        stream = getattr(sys, "stderr", None)
        if stream is not None:
            stream.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        return self.rfile.read(length) if length else b""

    def _read_json_body(self) -> dict:
        body = self._read_body()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, data, status=200):
        payload = json.dumps(_sanitize_json(data), ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data: bytes, content_type: str, status=200, extra_headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, path: str):
        if path in ("", "/"):
            file_path = STATIC_ROOT / "index.html"
        else:
            rel = Path(path.lstrip("/"))
            file_path = (STATIC_ROOT / rel).resolve()
            if not str(file_path).startswith(str(STATIC_ROOT.resolve())):
                self._send_json({"error": "Forbidden"}, status=403)
                return
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self._send_bytes(file_path.read_bytes(), ctype)

    def _serve_plotly(self):
        try:
            import plotly

            p = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if p.exists():
                self._send_bytes(p.read_bytes(), "application/javascript; charset=utf-8")
                return
        except Exception:
            pass
        self._send_json({"error": "Local Plotly asset not found"}, status=404)

    def _handle_create_session(self):
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            fields = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            mode = fields.getvalue("mode") or "single"
            if mode != "single":
                raise ValueError("Multipart session creation only supports single-file mode.")
            session = _create_session_single(fields)
        else:
            payload = self._read_json_body()
            mode = payload.get("mode") or "batch"
            if mode == "batch":
                session = _create_session_batch(payload)
            else:
                raise ValueError("Single-file mode requires multipart upload.")
        self._send_json({"session": session})

    def _handle_run(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session.")
        steps = payload.get("steps") or []
        steps = [s for s in steps if s in PIPELINE_STEPS]
        if not steps:
            raise ValueError("No valid pipeline steps selected.")
        raw_step_modes = payload.get("step_modes") or {}
        step_modes = {
            s: "reuse" if raw_step_modes.get(s) == "reuse" else "recompute"
            for s in steps
        }
        params = _merge_params(payload.get("params"))
        post_tips_mode = payload.get("post_tips_mode") or "auto"
        export_png = bool(payload.get("export_png", False))
        patient_id = payload.get("patient_id")
        patients = session.get("patients") or []
        if patient_id and patient_id != "all":
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        if not patients:
            raise ValueError("No patients selected.")
        job = _new_job(session_id, steps, patients, step_modes=step_modes)
        with STATE_LOCK:
            job["_patients_runtime"] = patients
            session["params"] = params
        thread = threading.Thread(
            target=_run_job,
            args=(job["id"], params, post_tips_mode, export_png),
            daemon=True,
        )
        thread.start()
        self._send_json({"job": job})

    def _handle_delete_centerline_branches(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        patient_id = payload.get("patient_id")
        branch_ids = payload.get("branch_ids") or []
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session.")
        patient = _resolve_patient(session, patient_id)
        if not patient:
            raise ValueError("Patient not found.")
        result = delete_centerline_terminal_branches(
            Path(patient["stl_path"]), [str(item) for item in branch_ids])
        self._send_json({"ok": True, "result": result})

    def _handle_job(self, path: str):
        job_id = path.rstrip("/").split("/")[-1]
        with STATE_LOCK:
            job = JOBS.get(job_id)
        if not job:
            self._send_json({"error": "Job not found"}, status=404)
            return
        self._send_json({"job": job})

    def _handle_session_data(self, path: str, query: str):
        session_id = path.split("/")[3]
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or [None])[0]
        section_stride = _safe_int((qs.get("section_stride") or [10])[0], 10)
        max_faces = _safe_int((qs.get("max_faces") or [80000])[0], 80000)
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            self._send_json({"error": "Session not found"}, status=404)
            return
        patient = _resolve_patient(session, patient_id)
        if not patient:
            self._send_json({"error": "Patient not found"}, status=404)
            return
        data = build_visualization_data(Path(patient["stl_path"]), section_stride=section_stride, max_faces=max_faces)
        data["session"] = session
        self._send_json(data)

    def _handle_download(self, path: str, query: str):
        session_id = path.split("/")[3]
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or ["all"])[0]
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            self._send_json({"error": "Session not found"}, status=404)
            return
        if patient_id == "all":
            patients = session.get("patients") or []
            name = f"ppg_outputs_{session_id}.zip"
        else:
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
            name = f"ppg_outputs_{patient_id or session_id}.zip"
        payload = _zip_patient_outputs(patients)
        self._send_bytes(
            payload,
            "application/zip",
            extra_headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--conda-env", default=None,
                        help="Restart the server under this conda environment before serving.")
    parser.add_argument("--conda-exe", default=None,
                        help="Path to conda.exe or conda.bat if it is not discoverable.")
    parser.add_argument("--no-conda-reexec", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = _load_config(args.config)
    args.host = args.host or str(config.get("host") or "127.0.0.1")
    args.port = args.port or int(config.get("port") or 8765)
    args.conda_env = (
        args.conda_env
        if args.conda_env is not None
        else str(config.get("conda_env") or "").strip() or None
    )
    args.conda_exe = (
        args.conda_exe
        if args.conda_exe is not None
        else str(config.get("conda_exe") or "").strip() or None
    )
    _maybe_reexec_in_conda(args, sys.argv[1:])
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    if getattr(sys, "stdout", None) is not None:
        print(f"PPG workbench running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
