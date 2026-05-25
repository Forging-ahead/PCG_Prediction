const DEFAULT_PARAMS = {
  pitch: 0.5,
  min_branch_length_mm: 10.0,
  min_relative_length: 0.05,
  min_radius_ratio: 0.4,
  keep_radius_ratio: 0.55,
  absolute_min_branch_length_mm: 3.0,
  absolute_min_radius_mm: 0.5,
  merge_bp_distance_mm: 5.0,
  n_fit_points: 10,
  n_profile_points: 100,
  curvature_window: 7,
  sample_step: 3,
  ownership_factor: 1.8,
  junction_policy: "min_valid",
  max_diameter_rate_per_mm: 0.5,
};

const PARAM_LABELS = {
  pitch: "体素间距",
  min_branch_length_mm: "最小分支长度",
  min_relative_length: "相对长度阈值",
  min_radius_ratio: "最小半径比例",
  keep_radius_ratio: "保留半径比例",
  absolute_min_branch_length_mm: "硬剪枝长度",
  absolute_min_radius_mm: "硬剪枝半径",
  merge_bp_distance_mm: "分叉点合并距离",
  n_fit_points: "拟合点数",
  n_profile_points: "剖面点数",
  curvature_window: "曲率窗口",
  sample_step: "截面采样步长",
  ownership_factor: "截面归属半径倍数",
  junction_policy: "交叉区策略",
  max_diameter_rate_per_mm: "直径变化率上限",
};

const STEPS = [
  ["centerline", "中心线提取"],
  ["smooth", "中心线平滑"],
  ["segment", "解剖分段"],
  ["profiles", "截面特征"],
  ["features", "统计特征"],
  ["export", "导出可视化"],
];

const LAYERS = {
  mesh: true,
  rawCenterline: false,
  smoothCenterline: true,
  segments: true,
  branchPoints: false,
  featurePoints: true,
  sampledSections: true,
  maxSections: true,
  meanSections: true,
  labels: true,
};

const CENTERLINE_EDIT_COLORS = [
  "#d9822b",
  "#7c3aed",
  "#0f9f6e",
  "#e11d48",
  "#2563eb",
  "#ca8a04",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#9333ea",
];

const state = {
  mode: "single",
  session: null,
  data: null,
  params: { ...DEFAULT_PARAMS },
  stepModes: Object.fromEntries(STEPS.map(([key]) => [key, "recompute"])),
  layers: { ...LAYERS },
  job: null,
  pollTimer: null,
  centerlineEdit: {
    active: false,
    selected: new Set(),
  },
};

const $ = (id) => document.getElementById(id);

function init() {
  buildStepButtons();
  buildParamInputs();
  bindEvents();
  renderCenterlineEditControls();
  checkHealth();
}

function bindEvents() {
  $("singleModeBtn").addEventListener("click", () => setMode("single"));
  $("batchModeBtn").addEventListener("click", () => setMode("batch"));
  $("createSessionBtn").addEventListener("click", createSession);
  $("runAllBtn").addEventListener("click", () => runSteps(STEPS.map(([key]) => key), true));
  $("refreshBtn").addEventListener("click", refreshData);
  $("downloadBtn").addEventListener("click", downloadResults);
  $("paramsBtn").addEventListener("click", () => $("paramsPanel").classList.toggle("hidden"));
  $("patientSelect").addEventListener("change", refreshData);
  $("sectionStride").addEventListener("input", () => {
    $("sectionStrideValue").textContent = $("sectionStride").value;
  });
  $("sectionStride").addEventListener("change", refreshData);
  $("meshOpacity").addEventListener("input", () => renderScene());
  $("centerlineEditBtn").addEventListener("click", toggleCenterlineEdit);
  $("centerlinePickBtn").addEventListener("click", toggleSelectedCenterlineBranch);
  $("centerlineBranchSelect").addEventListener("change", renderCenterlineEditControls);
  $("centerlineUndoBtn").addEventListener("click", clearCenterlineSelection);
  $("centerlineSaveBtn").addEventListener("click", saveCenterlineDeletion);

  document.querySelectorAll(".layer-toggle").forEach((input) => {
    input.addEventListener("change", () => {
      state.layers[input.dataset.layer] = input.checked;
      renderScene();
    });
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const envName = data.runtime?.conda_env || "unknown";
    $("serverState").textContent = data.ok ? `本地服务已连接 · ${envName}` : "服务异常";
  } catch (err) {
    $("serverState").textContent = "服务未连接";
  }
}

function setMode(mode) {
  state.mode = mode;
  $("singleModeBtn").classList.toggle("active", mode === "single");
  $("batchModeBtn").classList.toggle("active", mode === "batch");
  $("singleForm").classList.toggle("hidden", mode !== "single");
  $("batchForm").classList.toggle("hidden", mode !== "batch");
}

function buildStepButtons() {
  const wrap = $("stepButtons");
  wrap.innerHTML = "";
  STEPS.forEach(([key, label], idx) => {
    const row = document.createElement("div");
    row.className = "step-item";
    row.dataset.stepItem = key;

    const btn = document.createElement("button");
    btn.className = "step-button";
    btn.type = "button";
    btn.dataset.step = key;
    btn.innerHTML = `
      <span class="step-index">${idx + 1}</span>
      <span>${label}</span>
      <span class="step-state" data-step-state="${key}">待运行</span>
    `;
    btn.addEventListener("click", () => runSteps([key], false));

    const mode = document.createElement("select");
    mode.className = "step-mode";
    mode.title = "选择该步骤导入已有中间结果或重新计算";
    mode.dataset.stepMode = key;
    mode.innerHTML = `
      <option value="recompute">重新计算</option>
      <option value="reuse">导入已有</option>
    `;
    mode.value = state.stepModes[key] || "recompute";
    mode.addEventListener("change", () => {
      state.stepModes[key] = mode.value;
      renderStepAvailability();
    });

    row.appendChild(btn);
    row.appendChild(mode);
    wrap.appendChild(row);
  });
}

function buildParamInputs() {
  const wrap = $("paramGrid");
  wrap.innerHTML = "";
  Object.entries(DEFAULT_PARAMS).forEach(([key, value]) => {
    const label = document.createElement("label");
    label.textContent = PARAM_LABELS[key] || key;
    label.title = key;
    let input;
    if (key === "junction_policy") {
      input = document.createElement("select");
      ["min_valid", "cap_min", "keep"].forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        input.appendChild(opt);
      });
      input.value = value;
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = Number.isInteger(value) ? "1" : "0.01";
      input.value = value;
    }
    input.dataset.param = key;
    input.addEventListener("change", readParams);
    wrap.appendChild(label);
    wrap.appendChild(input);
  });
}

function readParams() {
  const next = { ...state.params };
  document.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    if (key === "junction_policy") {
      next[key] = input.value;
    } else {
      const value = Number(input.value);
      next[key] = Number.isFinite(value) ? value : DEFAULT_PARAMS[key];
    }
  });
  state.params = next;
  return next;
}

async function createSession() {
  clearJob();
  setBusy(true);
  try {
    let res;
    if (state.mode === "single") {
      const file = $("stlFile").files[0];
      if (!file) throw new Error("请选择 STL 文件");
      const form = new FormData();
      form.append("mode", "single");
      form.append("stl_file", file);
      form.append("output_dir", $("singleOutputDir").value.trim());
      res = await fetch("/api/session", { method: "POST", body: form });
    } else {
      const payload = {
        mode: "batch",
        root_folder: $("batchRoot").value.trim(),
        stl_name: $("stlName").value.trim() || "vessel.stl",
      };
      res = await fetchJson("/api/session", payload);
    }
    const payload = await readResponse(res);
    state.session = payload.session;
    populatePatients();
    await refreshData();
    logLine(`Session ${state.session.id} loaded.`);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

function populatePatients() {
  const select = $("patientSelect");
  select.innerHTML = "";
  const patients = state.session?.patients || [];
  if (state.session?.mode === "batch" && patients.length > 1) {
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = `全部病例 (${patients.length})`;
    select.appendChild(all);
  }
  patients.forEach((patient) => {
    const opt = document.createElement("option");
    opt.value = patient.id;
    opt.textContent = patient.id;
    select.appendChild(opt);
  });
  if (patients[0]) select.value = patients[0].id;
}

async function runSteps(steps, allPatients) {
  if (!state.session) {
    showError(new Error("请先载入输入"));
    return;
  }
  readParams();
  clearJob();
  setBusy(true);
  try {
    const selected = $("patientSelect").value;
    const patientId = allPatients && state.session.mode === "batch" ? "all" : (selected === "all" ? null : selected);
    const res = await fetchJson("/api/run", {
      session_id: state.session.id,
      steps,
      params: state.params,
      step_modes: state.stepModes,
      patient_id: patientId,
      post_tips_mode: $("postTipsMode").value,
      export_png: $("exportPng").checked,
    });
    const payload = await readResponse(res);
    state.job = payload.job;
    startPolling();
  } catch (err) {
    setBusy(false);
    showError(err);
  }
}

function startPolling() {
  if (!state.job) return;
  pollJob();
  state.pollTimer = setInterval(pollJob, 1500);
}

async function pollJob() {
  if (!state.job) return;
  try {
    const res = await fetch(`/api/job/${state.job.id}`);
    const payload = await res.json();
    state.job = payload.job;
    renderJob();
    if (["done", "failed"].includes(state.job.status)) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      setBusy(false);
      await refreshData();
    }
  } catch (err) {
    logLine(`Job polling failed: ${err.message}`);
  }
}

function renderJob() {
  const job = state.job;
  if (!job) return;
  const pct = job.total ? Math.round((job.completed / job.total) * 100) : 0;
  $("jobStatus").textContent = `${job.status} ${pct}% ${job.current || ""}`;
  $("logs").textContent = (job.logs || []).join("\n\n");
  $("logs").scrollTop = $("logs").scrollHeight;

  const selected = $("patientSelect").value;
  const patientResults = selected && job.results ? job.results[selected] : null;
  STEPS.forEach(([key]) => {
    const el = document.querySelector(`[data-step-state="${key}"]`);
    if (!el) return;
    if (patientResults && key in patientResults) {
      el.textContent = patientResults[key] ? "完成" : "失败";
      el.style.color = patientResults[key] ? "var(--ok)" : "var(--danger)";
    }
  });
}

async function refreshData() {
  if (!state.session) return;
  const selected = $("patientSelect").value;
  const patient = selected === "all" ? state.session.patients[0]?.id : selected;
  if (!patient) return;
  $("emptyState").classList.remove("hidden");
  try {
    const stride = $("sectionStride").value;
    const res = await fetch(`/api/session/${state.session.id}/data?patient=${encodeURIComponent(patient)}&section_stride=${stride}`);
    const data = await readResponse(res);
    state.data = data;
    reconcileCenterlineSelection();
    renderStepAvailability();
    renderScene();
    renderInspector();
    renderCenterlineEditControls();
  } catch (err) {
    showError(err);
  }
}

function renderStepAvailability() {
  const status = state.data?.step_files || {};
  STEPS.forEach(([key]) => {
    const row = document.querySelector(`[data-step-item="${key}"]`);
    const select = document.querySelector(`[data-step-mode="${key}"]`);
    const step = status[key];
    if (!row || !select) return;
    const reuse = state.stepModes[key] === "reuse";
    const ready = Boolean(step?.ready);
    row.classList.toggle("reuse-mode", reuse);
    row.classList.toggle("missing-reuse", reuse && !ready);
    select.title = ready
      ? "已找到该步骤保存的中间结果"
      : "未找到该步骤需要的中间结果文件";
  });
}

function currentPatientId() {
  const selected = $("patientSelect").value;
  return selected === "all" ? state.session?.patients?.[0]?.id : selected;
}

function toggleCenterlineEdit() {
  state.centerlineEdit.active = !state.centerlineEdit.active;
  if (state.centerlineEdit.active) {
    state.layers.rawCenterline = true;
    const rawToggle = document.querySelector('.layer-toggle[data-layer="rawCenterline"]');
    if (rawToggle) rawToggle.checked = true;
  } else {
    state.centerlineEdit.selected.clear();
  }
  renderScene();
  renderCenterlineEditControls();
}

function clearCenterlineSelection() {
  state.centerlineEdit.selected.clear();
  renderScene();
  renderCenterlineEditControls();
}

function centerlineBranchColor(index) {
  return CENTERLINE_EDIT_COLORS[index % CENTERLINE_EDIT_COLORS.length];
}

function centerlineBranchLabel(branch, index) {
  return `#${index + 1} 端点 ${branch.endpoint_id} → 分叉 ${branch.junction_id} · ${fmt(branch.length_mm, 1)} mm`;
}

function reconcileCenterlineSelection() {
  const valid = new Set((state.data?.centerline_edit?.branches || []).map((item) => item.id));
  for (const id of Array.from(state.centerlineEdit.selected)) {
    if (!valid.has(id)) state.centerlineEdit.selected.delete(id);
  }
}

function renderCenterlineEditControls() {
  const branches = state.data?.centerline_edit?.branches || [];
  const selected = state.centerlineEdit.selected.size;
  const branchSelect = $("centerlineBranchSelect");
  const previousValue = branchSelect.value;
  branchSelect.innerHTML = "";
  if (branches.length) {
    branches.forEach((branch, index) => {
      const option = document.createElement("option");
      option.value = branch.id;
      option.textContent = centerlineBranchLabel(branch, index);
      option.style.color = centerlineBranchColor(index);
      branchSelect.appendChild(option);
    });
    const validPrevious = branches.some((branch) => branch.id === previousValue);
    branchSelect.value = validPrevious ? previousValue : branches[0].id;
  } else {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无可删分支";
    branchSelect.appendChild(option);
  }
  const chosenId = branchSelect.value;
  const chosenIndex = branches.findIndex((branch) => branch.id === chosenId);
  if (chosenIndex >= 0) {
    branchSelect.style.borderColor = centerlineBranchColor(chosenIndex);
  } else {
    branchSelect.style.borderColor = "";
  }
  $("centerlineEditBtn").classList.toggle("active", state.centerlineEdit.active);
  $("centerlineEditBtn").disabled = !state.session || !state.data?.centerlines?.raw;
  branchSelect.disabled = !state.centerlineEdit.active || branches.length === 0;
  $("centerlinePickBtn").disabled = !state.centerlineEdit.active || !chosenId;
  $("centerlinePickBtn").textContent = chosenId && state.centerlineEdit.selected.has(chosenId)
    ? "取消该段"
    : "选择删除";
  $("centerlineUndoBtn").disabled = !state.centerlineEdit.active || selected === 0;
  $("centerlineSaveBtn").disabled = !state.centerlineEdit.active || selected === 0;
  $("centerlineEditStatus").textContent = state.centerlineEdit.active
    ? `可删 ${branches.length} 段 · 已选 ${selected}`
    : `可删 ${branches.length} 段`;
}

function toggleSelectedCenterlineBranch() {
  if (!state.centerlineEdit.active) return;
  const branchId = $("centerlineBranchSelect").value;
  if (!branchId) return;
  toggleCenterlineBranchSelection(branchId);
}

function toggleCenterlineBranchSelection(branchId) {
  if (state.centerlineEdit.selected.has(branchId)) {
    state.centerlineEdit.selected.delete(branchId);
  } else {
    state.centerlineEdit.selected.add(branchId);
  }
  const branches = state.data?.centerline_edit?.branches || [];
  const branch = branches.find((item) => item.id === branchId);
  const index = branches.findIndex((item) => item.id === branchId);
  $("pickedInfo").innerHTML = branch
    ? `原始中心线分支<br>编号: #${index + 1}<br>端点: ${branch.endpoint_id}<br>分叉点: ${branch.junction_id}<br>长度: ${fmt(branch.length_mm, 2)} mm<br>状态: ${state.centerlineEdit.selected.has(branchId) ? "待删除" : "未选择"}`
    : "原始中心线分支";
  renderScene();
}

async function saveCenterlineDeletion() {
  if (!state.session || state.centerlineEdit.selected.size === 0) return;
  const patientId = currentPatientId();
  if (!patientId) return;
  setBusy(true);
  try {
    const branchIds = Array.from(state.centerlineEdit.selected);
    const res = await fetchJson("/api/centerline/delete-branches", {
      session_id: state.session.id,
      patient_id: patientId,
      branch_ids: branchIds,
    });
    const payload = await readResponse(res);
    const removed = payload.result?.removed_nodes ?? 0;
    const stale = payload.result?.removed_outputs || [];
    logLine(`Deleted ${branchIds.length} centerline branch(es), removed ${removed} node(s).`);
    if (stale.length) logLine(`Cleared derived outputs: ${stale.join(", ")}`);
    state.centerlineEdit.selected.clear();
    state.centerlineEdit.active = false;
    state.stepModes.centerline = "reuse";
    const centerlineMode = document.querySelector('[data-step-mode="centerline"]');
    if (centerlineMode) centerlineMode.value = "reuse";
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
    renderCenterlineEditControls();
  }
}

function renderScene() {
  if (!state.data || !window.Plotly) return;
  const traces = [];
  const data = state.data;
  const opacity = Number($("meshOpacity").value || 22) / 100;

  if (data.mesh && data.mesh.vertices && state.layers.mesh) {
    const vertices = data.mesh.vertices;
    const faces = data.mesh.faces || [];
    traces.push({
      type: "mesh3d",
      name: `STL mesh (${data.mesh.n_faces_rendered || faces.length} faces)`,
      x: vertices.map((v) => v[0]),
      y: vertices.map((v) => v[1]),
      z: vertices.map((v) => v[2]),
      i: faces.map((f) => f[0]),
      j: faces.map((f) => f[1]),
      k: faces.map((f) => f[2]),
      color: "#b8c3cc",
      opacity,
      hoverinfo: "skip",
      flatshading: false,
      lighting: { ambient: 0.55, diffuse: 0.8, specular: 0.1 },
    });
  }

  addCenterlineTrace(traces, data.centerlines?.raw, "原始中心线", "#6b7280", state.layers.rawCenterline || state.centerlineEdit.active);
  addCenterlineTrace(traces, data.centerlines?.smooth, "平滑中心线", "#111827", state.layers.smoothCenterline);
  addEditableCenterlineTraces(traces, data.centerline_edit?.branches || []);
  addSegmentTraces(traces, data.segments || {});
  addBranchPointTrace(traces, data.branch_points || []);
  addFeaturePointTraces(traces, data.pointwise?.feature_points || {});
  addSectionTraces(traces, data.pointwise?.sampled_sections || {}, "sampledSections", "间隔截面", 2, 0.38);
  addNamedSectionTraces(traces, data.pointwise?.max_sections || {}, "maxSections", "最大截面", 6);
  addNamedSectionTraces(traces, data.pointwise?.mean_sections || {}, "meanSections", "平均截面", 4);
  addLabelTrace(traces, data.segments || {});

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#f8fafc",
    scene: {
      aspectmode: "data",
      xaxis: axisLayout("X"),
      yaxis: axisLayout("Y"),
      zaxis: axisLayout("Z"),
      camera: { eye: { x: 1.55, y: 1.45, z: 1.05 }, up: { x: 0, y: 0, z: 1 } },
    },
    legend: {
      x: 0.01,
      y: 0.99,
      bgcolor: "rgba(255,255,255,0.82)",
      bordercolor: "#d6dee7",
      borderwidth: 1,
      font: { size: 11 },
    },
    uirevision: "ppg-vessel-workbench",
  };
  Plotly.react("viewer", traces, layout, { displaylogo: false, responsive: true, scrollZoom: true });
  $("emptyState").classList.add("hidden");
  const plot = $("viewer");
  if (typeof plot.removeAllListeners === "function") {
    plot.removeAllListeners("plotly_click");
  }
  plot.on("plotly_click", (event) => {
    const pt = event.points?.[0];
    if (handleCenterlineEditClick(pt)) return;
    if (pt?.customdata) {
      $("pickedInfo").innerHTML = escapeHtml(String(pt.customdata)).replaceAll("\n", "<br>").replaceAll("&lt;br&gt;", "<br>");
    }
  });
  renderCenterlineEditControls();
}

function addEditableCenterlineTraces(traces, branches) {
  if (!state.centerlineEdit.active || !branches.length) return;
  branches.forEach((branch, index) => {
    const selected = state.centerlineEdit.selected.has(branch.id);
    const color = centerlineBranchColor(index);
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: selected ? `待删除 #${index + 1}` : `可删分支 #${index + 1}`,
      x: branch.x,
      y: branch.y,
      z: branch.z,
      customdata: branch.x.map(() => `centerline-edit:${branch.id}`),
      line: {
        color: selected ? "#b42318" : color,
        width: selected ? 12 : 9,
      },
      opacity: selected ? 0.98 : 0.88,
      hovertemplate: `#${index + 1}<br>端点 ${branch.endpoint_id} → 分叉点 ${branch.junction_id}<br>length: ${fmt(branch.length_mm, 2)} mm<extra></extra>`,
    });
  });
}

function handleCenterlineEditClick(point) {
  const marker = point?.customdata;
  if (!state.centerlineEdit.active || typeof marker !== "string" || !marker.startsWith("centerline-edit:")) {
    return false;
  }
  const branchId = marker.slice("centerline-edit:".length);
  $("centerlineBranchSelect").value = branchId;
  toggleCenterlineBranchSelection(branchId);
  return true;
}

function axisLayout(title) {
  return {
    title,
    backgroundcolor: "#f8fafc",
    gridcolor: "#d9e1e8",
    zerolinecolor: "#c9d4df",
    showspikes: false,
  };
}

function addCenterlineTrace(traces, line, name, color, visible) {
  if (!line || !visible) return;
  traces.push({
    type: "scatter3d",
    mode: "lines",
    name,
    x: line.x,
    y: line.y,
    z: line.z,
    line: { color, width: 3 },
    hoverinfo: "skip",
  });
}

function addSegmentTraces(traces, segments) {
  if (!state.layers.segments) return;
  Object.entries(segments).forEach(([key, seg]) => {
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: seg.label || key.toUpperCase(),
      x: seg.x,
      y: seg.y,
      z: seg.z,
      line: { color: seg.color, width: 8 },
      hovertemplate: `<b>${seg.label}</b><br>length: ${fmt(seg.length_mm, 2)} mm<br>tortuosity: ${fmt(seg.tortuosity, 4)}<br>mean curvature: ${fmt(seg.mean_curvature, 5)}<extra></extra>`,
    });
  });
}

function addBranchPointTrace(traces, branchPoints) {
  if (!state.layers.branchPoints || !branchPoints.length) return;
  traces.push({
    type: "scatter3d",
    mode: "markers",
    name: "分叉点",
    x: branchPoints.map((p) => p.coord?.[0]),
    y: branchPoints.map((p) => p.coord?.[1]),
    z: branchPoints.map((p) => p.coord?.[2]),
    marker: { size: 5, color: "#111820", line: { color: "#fff", width: 1 } },
    text: branchPoints.map((p) => `BP ${p.id}`),
    hovertemplate: "%{text}<extra></extra>",
  });
}

function addFeaturePointTraces(traces, featurePoints) {
  if (!state.layers.featurePoints) return;
  let colorbarShown = false;
  Object.entries(featurePoints).forEach(([key, fp]) => {
    if (!fp.x?.length) return;
    traces.push({
      type: "scatter3d",
      mode: "markers",
      name: `${fp.label} 曲率点`,
      x: fp.x,
      y: fp.y,
      z: fp.z,
      customdata: fp.hover,
      marker: {
        size: fp.size,
        color: fp.curvature,
        colorscale: "Viridis",
        opacity: 0.86,
        colorbar: colorbarShown ? undefined : { title: "curvature", thickness: 12 },
        showscale: !colorbarShown,
        line: { width: 0 },
      },
      hovertemplate: "%{customdata}<extra></extra>",
    });
    colorbarShown = true;
  });
}

function addSectionTraces(traces, sections, layerKey, label, width, opacity) {
  if (!state.layers[layerKey]) return;
  Object.entries(sections).forEach(([key, sec]) => {
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `${sec.label} ${label}`,
      x: sec.x,
      y: sec.y,
      z: sec.z,
      line: { color: sec.color, width },
      opacity,
      hoverinfo: "skip",
    });
  });
}

function addNamedSectionTraces(traces, sections, layerKey, label, width) {
  if (!state.layers[layerKey]) return;
  Object.entries(sections).forEach(([key, sec]) => {
    const color = state.data.segments?.[key]?.color || "#177e89";
    const segLabel = state.data.segments?.[key]?.label || key.toUpperCase();
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `${segLabel} ${label}`,
      x: sec.x,
      y: sec.y,
      z: sec.z,
      line: { color, width, dash: layerKey === "meanSections" ? "dash" : "solid" },
      hovertemplate: `<b>${segLabel} ${label}</b><br>point: ${sec.index}<br>diameter: ${fmt(sec.diameter, 3)} mm<br>area: ${fmt(sec.area, 3)} mm²<extra></extra>`,
    });
  });
}

function addLabelTrace(traces, segments) {
  if (!state.layers.labels) return;
  const x = [];
  const y = [];
  const z = [];
  const text = [];
  Object.values(segments).forEach((seg) => {
    if (!seg.midpoint) return;
    x.push(seg.midpoint[0]);
    y.push(seg.midpoint[1]);
    z.push(seg.midpoint[2]);
    text.push(`<b>${seg.label}</b>`);
  });
  if (!x.length) return;
  traces.push({
    type: "scatter3d",
    mode: "text",
    name: "标签",
    x,
    y,
    z,
    text,
    textfont: { size: 13, color: "#17212b" },
    showlegend: false,
    hoverinfo: "skip",
  });
}

function renderInspector() {
  const data = state.data;
  renderSegmentCards(data.features?.statistical || {}, data.segments || {});
  renderSystemFeatures(data.features || {});
}

function renderSegmentCards(stats, segments) {
  const wrap = $("segmentCards");
  wrap.innerHTML = "";
  const names = Object.keys(segments);
  if (!names.length) {
    wrap.textContent = "暂无分段结果";
    return;
  }
  names.forEach((name) => {
    const seg = segments[name];
    const block = stats[name] || {};
    const card = document.createElement("article");
    card.className = "segment-card";
    card.style.borderLeftColor = seg.color || "#177e89";
    card.innerHTML = `
      <h3>${seg.label}</h3>
      ${metricRow("长度", block.length ?? seg.length_mm, "mm")}
      ${metricRow("平均直径", block.mean_diameter, "mm")}
      ${metricRow("最大直径", block.max_diameter, "mm")}
      ${metricRow("平均面积", block.mean_area, "mm²")}
      ${metricRow("最大曲率", block.max_curvature, "1/mm")}
    `;
    wrap.appendChild(card);
  });
}

function metricRow(label, value, unit) {
  return `<div class="metric"><span>${label}</span><strong>${fmt(value, 3)} ${unit}</strong></div>`;
}

function renderSystemFeatures(features) {
  const wrap = $("systemFeatures");
  wrap.innerHTML = "";
  const flat = flattenSystem(features.system || {});
  const global = features.global || {};
  const rows = [
    ...Object.entries(global).map(([key, value]) => [key, value]),
    ...Object.entries(flat).map(([key, value]) => [key, value]),
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!rows.length) {
    wrap.textContent = "暂无系统特征";
    return;
  }
  rows.slice(0, 80).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    row.innerHTML = `<span title="${escapeHtml(key)}">${escapeHtml(key)}</span><strong>${escapeHtml(formatValue(value))}</strong>`;
    wrap.appendChild(row);
  });
}

function flattenSystem(system) {
  if (system.all_values && typeof system.all_values === "object") return system.all_values;
  if (system.available && typeof system.available === "object") {
    return { ...system.available, ...(system.unavailable || {}) };
  }
  return system;
}

function formatValue(value) {
  if (typeof value === "number") return fmt(value, 4);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return `[${value.slice(0, 3).map((v) => fmt(v, 3)).join(", ")}${value.length > 3 ? ", ..." : ""}]`;
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  const n = Number(value);
  if (!Number.isFinite(n)) return "NA";
  return n.toFixed(digits);
}

async function downloadResults() {
  if (!state.session) {
    showError(new Error("请先载入输入"));
    return;
  }
  const patient = $("patientSelect").value || "all";
  window.location.href = `/api/session/${state.session.id}/download?patient=${encodeURIComponent(patient)}`;
}

function clearJob() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.job = null;
  $("jobStatus").textContent = "空闲";
  $("logs").textContent = "";
}

function logLine(text) {
  $("logs").textContent += `${text}\n`;
  $("logs").scrollTop = $("logs").scrollHeight;
}

function setBusy(isBusy) {
  document.querySelectorAll("button").forEach((btn) => {
    if (btn.id === "paramsBtn") return;
    btn.disabled = isBusy;
  });
}

function showError(err) {
  const message = err?.message || String(err);
  $("jobStatus").textContent = message;
  $("jobStatus").style.color = "var(--danger)";
  logLine(`ERROR: ${message}`);
}

function fetchJson(url, payload) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function readResponse(res) {
  const payload = await res.json();
  if (!res.ok || payload.error) {
    throw new Error(payload.error || `HTTP ${res.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.addEventListener("DOMContentLoaded", init);
