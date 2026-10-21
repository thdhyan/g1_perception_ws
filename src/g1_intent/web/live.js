// G1 live console: runs after intent-sim's sim.js (same global scope) and adds the
// Live / Labeler / Robot tabs. The "live" run is the current /mocap/agents scene;
// while it is selected its poses are refreshed in place so the SNGNN / Nav / VO
// tabs work on the live scene. Server side: g1_intent/web_console.py.

const $ = id => document.getElementById(id);
const LIVE_POLL_MS = 200, LIVE_GEOM_MS = 1000;
let live = { headPreview: null, st: null, opts: null, planId: null, preview: null, previewKey: null, walked: [], walkedId: null,
             humanKey: null, lastGeom: 0, lastSn: 0, labRows: "" };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", body: JSON.stringify(body) });
  return r.json();
}
const origin = () => (live.st && live.st.canvas_origin) || [-5, 3.5];
const toCanvas = (x, y) => [x - origin()[0], origin()[1] - y];   // PhaseSpace m -> canvas m (y down)
const fmt = (v, d = 2) => v == null ? "-" : Number(v).toFixed(d);
const esc = s => escapeHtml(String(s ?? ""));
function msg(id, text, cls) { const el = $(id); el.textContent = text; el.className = cls || "dim"; }

// ── options ──────────────────────────────────────────────────────────
async function loadOptions() {
  const o = live.opts = await api("/api/options");
  const fill = (sel, vals, v) => { $(sel).innerHTML = vals.map(x => `<option>${esc(x)}</option>`).join(""); $(sel).value = v; };
  fill("liveProvider", Object.keys(o.providers), o.defaults.provider);
  fill("liveVariant", o.variants, o.defaults.prompt_variant);
  fill("liveStyle", o.styles, o.defaults.prompt_style);
  $("liveFormat").value = o.defaults.scene_format;
  $("liveTemp").value = o.defaults.temperature;
  $("liveModel").value = o.defaults.model;
  fillModels();
  $("robotNote").textContent = o.dry_run
    ? "DRY RUN: buttons are refused (no robot_bridge)."
    : "Sends to robot_bridge over the Unix socket (robot on ethernet). Refused while a plan runs, except Stop.";
}
function fillModels() {
  const p = live.opts.providers[$("liveProvider").value] || { models: [] };
  $("liveModels").innerHTML = p.models.map(m => `<option value="${esc(m)}">`).join("");
}
$("liveProvider").addEventListener("change", () => {
  fillModels();
  const p = live.opts.providers[$("liveProvider").value];
  if (p && p.models.length) $("liveModel").value = p.models[0];
});

function taskBody() {
  return { task: $("liveTask").value.trim(), provider: $("liveProvider").value, model: $("liveModel").value.trim(),
           prompt_variant: $("liveVariant").value, prompt_style: $("liveStyle").value,
           scene_format: $("liveFormat").value, temperature: Number($("liveTemp").value),
           anchor: $("liveAnchor").value || null };
}
$("planBtn").addEventListener("click", async () => {
  const b = taskBody();
  if (b.prompt_variant === "ra" && !b.anchor) return msg("liveMsg", "variant ra: pick an anchor", "err");
  const r = await api("/api/task", b);
  msg("liveMsg", r.ok ? "planning... (see log)" : r.error, r.ok ? "dim" : "err");
});
$("previewBtn").addEventListener("click", () => {
  document.querySelector('.tab[data-tab="promptTab"]').click();
  livePrompt();
});
async function livePrompt() {
  const r = await api("/api/prompt_preview", taskBody());
  $("promptTab").innerHTML = r.system_prompt
    ? `<p class="dim">Live scene, ${esc(taskBody().prompt_variant)} / ${esc(taskBody().prompt_style)}${r.warnings.length ? " · " + esc(r.warnings.join("; ")) : ""}</p><pre>${esc(r.system_prompt)}</pre>`
    : `<pre class="err">${esc(r.error)}</pre>`;
}
const simRenderPrompt = renderPrompt;
renderPrompt = function () { return selectedRunId === "live" ? livePrompt() : simRenderPrompt(); };

$("estopBtn").addEventListener("click", async () => { await api("/api/estop", {}); msg("liveMsg", "E-STOP sent", "err"); });

// ── live scene ───────────────────────────────────────────────────────
function agentsCanvas() {
  const ag = (live.st && live.st.agents && live.st.agents.agents) || [];
  return ag.map(a => { const [x, y] = toCanvas(a.x, a.y); return { ...a, cx: x, cy: y }; });
}
function updateLiveScene() {
  if (selectedRunId !== "live" || !scene || stepIdx !== -1 || playing) return;
  const ag = agentsCanvas(), humans = ag.filter(a => a.role === "human"), robot = ag.find(a => a.role === "robot");
  const key = humans.map(h => h.name).sort().join(",") + "|" + !!robot;
  if (live.humanKey !== null && key !== live.humanKey) { live.humanKey = key; selectRun("live"); return; }
  live.humanKey = key;
  for (const sc of [scene, trace.scene]) {
    if (robot && robot.yaw_deg != null) Object.assign(sc.robot, { x: robot.cx, y: robot.cy, facing_deg: ((robot.yaw_deg % 360) + 360) % 360 });
    for (const h of sc.humans) {
      const a = humans.find(q => q.name === h.name);
      if (a) Object.assign(h, { x: a.cx, y: a.cy, facing_deg: (((a.yaw_deg || 0) % 360) + 360) % 360 });
    }
    if (live.st && live.st.obstacles) sc.obstacles = live.st.obstacles.map(o => ({ ...o }));  // LiDAR walls / tables
    // next plan goal as the scene's "goal" object: the Nav / VO / SNGNN tabs plan toward it
    const g = nextGoal();
    sc.objects = sc.objects.filter(o => o.kind !== "goal");
    if (g) { const [x, y] = toCanvas(g.x, g.y); sc.objects.push({ name: "goal", kind: "goal", x, y }); }
  }
  trail = [[scene.robot.x, scene.robot.y]];
  const now = performance.now();
  if (now - live.lastGeom > LIVE_GEOM_MS) {   // cost maps are slow: re-plan at 1 Hz
    live.lastGeom = now; nav.key = null; navGmm.key = null; navStatic.key = null;
    const snOn = ["sngnnTab", "navTab"].some(t => $(t).classList.contains("active"));
    if (snOn && !sngnn.inflight) sngnnQuery();
  }
  draw();
}
function nextGoal() {
  const p = live.st && live.st.plan, ex = live.st && live.st.exec;
  if (!p || !p.goals.length) return null;
  const i = ex && ex.plan_id && ex.plan_id.split("~")[0] === p.id.split("~")[0] && ex.state === "running" ? ex.goal : 0;
  return p.goals[Math.min(i || 0, p.goals.length - 1)];
}

// ── overlay on the main canvas ───────────────────────────────────────
const simDraw = draw;
draw = function () {
  simDraw();
  if (!ctx || !live.st) return;
  const S = SCALE, isLive = selectedRunId === "live";
  ctx.save();
  // every PhaseSpace group the labeler sees; unlabelled ones grey
  const hats = (live.st.labeler && live.st.labeler.hats) || {};
  for (const [hat, h] of Object.entries(hats)) {
    if (h.x == null || !isLive) continue;
    const [x, y] = toCanvas(h.x, h.y);
    if (!h.name) {
      ctx.fillStyle = "rgba(120,120,120,0.5)";
      ctx.beginPath(); ctx.arc(x * S, y * S, 12, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = h.name ? "#333" : "#777"; ctx.font = "11px monospace";
    ctx.fillText(h.name ? `${h.name} [${hat}]` : hat, x * S + 14, y * S - 14);
  }
  // labeler heading slider being dragged: arrow from that hat
  const hp = live.headPreview, hh = hp && hats[hp.hat];
  if (hh && hh.x != null) {
    const [x, y] = toCanvas(hh.x, hh.y), a = -hp.deg * Math.PI / 180;
    ctx.strokeStyle = "rgb(230,120,0)"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x * S, y * S); ctx.lineTo(x * S + 60 * Math.cos(a), y * S + 60 * Math.sin(a)); ctx.stroke();
    ctx.fillStyle = "rgb(230,120,0)"; ctx.font = "bold 12px sans-serif"; ctx.fillText(`${hp.deg}°`, x * S + 64 * Math.cos(a), y * S + 64 * Math.sin(a));
  }
  // anchor ring
  const anc = $("liveAnchor").value;
  const ah = isLive && scene && scene.humans.find(h => h.name === anc);
  if (ah) {
    ctx.strokeStyle = "rgb(230,180,0)"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.arc(ah.x * S, ah.y * S, 32, 0, 2 * Math.PI); ctx.stroke();
  }
  const p = live.st.plan;
  if (isLive && p) {
    // low-level planner preview
    if (live.preview && live.previewKey && live.previewKey.startsWith(p.id.split("~")[0])) {
      ctx.strokeStyle = NAV_COLORS[live.preview.planner] || "#0a8"; ctx.lineWidth = 3; ctx.setLineDash([8, 5]);
      for (const pts of live.preview.paths) {
        if (pts.length < 2) continue;
        ctx.beginPath(); ctx.moveTo(pts[0][0] * S, pts[0][1] * S);
        for (const [x, y] of pts.slice(1)) ctx.lineTo(x * S, y * S);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // plan goals (PhaseSpace -> canvas), numbered, heading tick; waypoint goals small
    let n = 0;
    for (const g of p.goals) {
      const [x, y] = toCanvas(g.x, g.y), a = -g.yaw_deg * Math.PI / 180;
      if (g.waypoint) {
        ctx.fillStyle = "rgba(0,150,130,0.8)";
        ctx.beginPath(); ctx.arc(x * S, y * S, 4, 0, 2 * Math.PI); ctx.fill();
        continue;
      }
      n++;
      ctx.strokeStyle = "rgb(200,30,30)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x * S, y * S, 10, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x * S, y * S); ctx.lineTo(x * S + 22 * Math.cos(a), y * S + 22 * Math.sin(a)); ctx.stroke();
      ctx.fillStyle = "rgb(200,30,30)"; ctx.font = "bold 12px sans-serif"; ctx.fillText(String(n), x * S - 4, y * S + 4);
    }
  }
  // walked path + executor pose
  if (isLive && live.walked.length > 1) {
    ctx.strokeStyle = "rgba(200,30,30,0.6)"; ctx.lineWidth = 2;
    ctx.beginPath();
    live.walked.forEach(([wx, wy], k) => { const [x, y] = toCanvas(wx, wy); k ? ctx.lineTo(x * S, y * S) : ctx.moveTo(x * S, y * S); });
    ctx.stroke();
  }
  ctx.restore();
};

// click a human on the canvas: anchor for variant ra / rs
canvas.addEventListener("click", e => {
  if (selectedRunId !== "live" || !scene) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * canvas.width / r.width / SCALE, y = (e.clientY - r.top) * canvas.height / r.height / SCALE;
  let best = null, bd = 0.6;
  for (const h of scene.humans) { const d = Math.hypot(h.x - x, h.y - y); if (d < bd) { bd = d; best = h; } }
  if (!best) return;
  $("liveAnchor").value = best.name;
  if ($("liveVariant").value !== "ra") $("liveVariant").value = "ra";
  msg("liveMsg", `anchor: ${best.name} (variant ra)`);
  draw();
});

// ── low-level planner: robot -> each goal on the live cost map ───────
async function computeWaypoints() {
  const p = live.st && live.st.plan, planner = $("llPlanner").value;
  if (!p) throw new Error("no plan");
  if (planner === "direct") return { planner, paths: [], waypoints: [] };
  if (selectedRunId !== "live") { await selectRun("live"); updateLiveScene(); }
  const snKey = `live|-1|${isOn("showMovingHumans")}`;
  const S = navNum("navWS") > 0 && sngnn.cost && sngnn.key === snKey ? sngnn.cost : null;
  navGmm.key = null;
  const maps = navCostMaps(ROBOT_RADIUS / SCALE, HUMAN_R_M, S);
  const lam = navNum("navLam"), lr = navNum("navLr"), wd = navNum("navWd");
  let start = [scene.robot.x, scene.robot.y];
  const paths = [], notes = [];
  for (const g of p.goals.filter(g => !g.waypoint)) {
    const goal = toCanvas(g.x, g.y);
    const res = planner === "astar" ? navAstar(maps.C, maps.lethal, start, goal, lam) : navWalk(planner, maps.C, start, goal, lam, lr, wd);
    const pts = res.pts.length ? res.pts.slice() : [start];
    if (res.status !== "reached") notes.push(`${g.call}: ${res.status}`);
    pts.push(goal);   // the goal itself ends every leg
    paths.push(pts);
    start = goal;
  }
  return { planner, paths, waypoints: paths, notes, sngnn: !!S };
}
$("llPreviewBtn").addEventListener("click", async () => {
  try {
    const r = await computeWaypoints();
    live.preview = r; live.previewKey = `${live.st.plan.id.split("~")[0]}|${r.planner}`;
    msg("llInfo", r.planner === "direct" ? "direct: goto controller straight to each goal"
      : `${NAV_NAMES[r.planner]}: ${r.paths.map(p => p.length).join(" + ")} pts${r.sngnn ? ", with SNGNN" : ", no SNGNN map (GMM + obstacles)"}${r.notes.length ? " · " + r.notes.join("; ") : ""}`,
      r.notes.length ? "err" : "dim");
    draw();
  } catch (e) { msg("llInfo", String(e), "err"); }
});
$("approveBtn").addEventListener("click", async () => {
  const p = live.st && live.st.plan;
  if (!p) return;
  try {
    const w = await computeWaypoints();
    live.preview = w; live.previewKey = `${p.id.split("~")[0]}|${w.planner}`;
    const r = await api("/api/approve", { plan_id: p.id, planner: w.planner, waypoints: w.waypoints });
    msg("llInfo", r.ok ? `approved ${r.plan_id}: ${r.n_goals} goals (${r.n_waypoints} waypoints)` : r.error, r.ok ? "ok" : "err");
  } catch (e) { msg("llInfo", String(e), "err"); }
});

// ── plan / exec / log panels ─────────────────────────────────────────
function renderPlan(p) {
  if (!p) return;
  const calls = (p.calls || []).map(c => `<tr><td class="${c.ok ? "ok" : "err"}">${c.ok ? "✓" : "✗"}</td>
    <td>${esc(c.function)}(${esc(JSON.stringify(c.ps_args || c.args))})</td><td class="err">${esc(c.error || "")}</td></tr>`).join("");
  const goals = p.goals.filter(g => !g.waypoint).map((g, k) => `${k + 1}. ${esc(g.call)} → (${fmt(g.x)}, ${fmt(g.y)}) ${fmt(g.yaw_deg, 0)}°`).join("<br>");
  const term = p.terminal ? `${esc(p.terminal.function)}(${esc(JSON.stringify(p.terminal.args))})` : "none";
  $("planCard").innerHTML = `<b>${esc(p.id)}</b><br>
    <span class="dim">${esc(p.provider)}/${esc(p.model)} · variant ${esc(p.prompt_variant)}${p.anchor ? " · anchor " + esc(p.anchor) : ""} · path ${esc(p.path_used)}</span>
    <p>${esc(p.task)}</p>
    <table class="plan-calls">${calls}</table>
    <p><b>Goals</b><br>${goals || '<span class="err">none</span>'}<br><b>Terminal</b> ${term}</p>
    <div class="live-row"><button id="openTraceBtn">Open trace (Steps tab)</button></div>
    <details><summary class="dim">raw LLM content</summary><pre>${esc(p.raw_content || "")}</pre></details>`;
  $("openTraceBtn").addEventListener("click", () => {
    selectRun(p.id.split("~")[0]);
    document.querySelector('.tab[data-tab="stepsTab"]').click();
  });
}
function renderExec(ex) {
  if (!ex) { $("execCard").innerHTML = '<span class="dim">executor: no state yet</span>'; return; }
  $("execCard").innerHTML = `<b>Executor</b> <span class="${ex.state === "running" ? "ok" : ex.state === "aborted" ? "err" : ""}">${esc(ex.state)}</span>
    ${ex.dry_run ? '<span class="dim">(dry run)</span>' : ""}<br>
    <span class="dim">plan ${esc(ex.plan_id)} · goal ${ex.goal == null ? "-" : ex.goal + 1}/${ex.n_goals} · ${esc(ex.phase || "")}</span><br>
    ${esc(ex.msg)}<br><span class="dim">pose ${ex.pose ? ex.pose.map(v => fmt(v)).join(", ") : "-"} · cmd ${ex.cmd ? ex.cmd.join(", ") : "-"}</span>`;
}
async function reloadRunList() {
  allRuns = await api("/api/runs");
  populateFilters(); renderRunList();
}

// ── labeler tab ──────────────────────────────────────────────────────
function renderLabeler(lab) {
  if (!lab) return;
  $("labStatus").textContent = lab.status;
  renderStatics(lab);
  const hats = Object.keys(lab.hats).sort(), key = hats.join(",");
  const body = document.querySelector("#labTable tbody");
  if (key !== live.labRows) {   // rebuild only when the group set changes: typed text survives polling
    live.labRows = key;
    body.innerHTML = hats.map(h => `<tr data-hat="${esc(h)}"><td>${esc(h)}</td><td class="leds"></td>
      <td><button class="role"></button></td><td><input class="name"></td><td><input class="heading"><input class="head-slider" type="range" min="-180" max="180" step="1" title="drag: arrow on the canvas, release: apply"></td>
      <td class="yaw"></td><td><button class="apply">Apply</button> <button class="capture">Capture</button></td></tr>`).join("");
    body.querySelectorAll("tr").forEach(tr => {
      const hat = tr.dataset.hat, q = c => tr.querySelector(c);
      const send = async action => {
        const r = await api("/api/labeler", { action, hat, name: q(".name").value, heading: q(".heading").value });
        if (!r.ok) $("labStatus").textContent = r.error;
        q(".name").dataset.dirty = ""; q(".heading").dataset.dirty = "";
      };
      q(".role").addEventListener("click", () => send("toggle_role"));
      q(".apply").addEventListener("click", () => send("apply"));
      q(".capture").addEventListener("click", () => send("capture"));
      for (const c of [".name", ".heading"]) q(c).addEventListener("input", () => { q(c).dataset.dirty = "1"; });
      // heading slider: drag shows the arrow on the canvas, release applies it
      q(".head-slider").addEventListener("input", () => {
        q(".heading").value = q(".head-slider").value; q(".heading").dataset.dirty = "1";
        live.headPreview = { hat, deg: Number(q(".head-slider").value) }; draw();
      });
      q(".head-slider").addEventListener("change", async () => { await send("apply"); live.headPreview = null; });
    });
  }
  body.querySelectorAll("tr").forEach(tr => {
    const h = lab.hats[tr.dataset.hat], q = c => tr.querySelector(c);
    q(".leds").textContent = h.leds;
    q(".role").textContent = h.role;
    for (const [c, v] of [[".name", h.name], [".heading", h.heading == null ? "" : Math.round(h.heading)]])
      if (document.activeElement !== q(c) && !q(c).dataset.dirty) q(c).value = v;
    const sl = q(".head-slider");
    if (document.activeElement !== sl && !(live.headPreview && live.headPreview.hat === tr.dataset.hat))
      sl.value = h.heading == null ? (h.yaw == null ? 0 : Math.round(h.yaw * 180 / Math.PI)) : Math.round(h.heading);
    q(".yaw").textContent = `${h.yaw == null ? "-" : (h.yaw * 180 / Math.PI).toFixed(0) + "°"} ${h.src}`;
  });
}
// static humans: one tracker, several people standing still
function renderStatics(lab) {
  const hats = Object.keys(lab.hats).sort(), sel = $("stHat");
  if (sel.dataset.key !== hats.join(",")) {
    const cur = sel.value;
    sel.dataset.key = hats.join(",");
    sel.innerHTML = hats.map(h => `<option>${esc(h)}</option>`).join("");
    if (hats.includes(cur)) sel.value = cur;
  }
  $("stAdd").disabled = !!lab.placing;
  const st = lab.statics || {}, names = Object.keys(st).sort();
  const key = JSON.stringify(st), body = document.querySelector("#stTable tbody");
  if (body.dataset.key === key) return;
  body.dataset.key = key;
  body.innerHTML = names.map(n => { const s = st[n];
    return `<tr><td>${esc(n)}</td><td>${s.x.toFixed(2)}</td><td>${s.y.toFixed(2)}</td><td>${Math.round(s.yaw_deg)}</td>
      <td>${esc(s.hat)}</td><td><button class="st-del" data-name="${esc(n)}">Delete</button></td></tr>`; }).join("");
  body.querySelectorAll(".st-del").forEach(b => b.addEventListener("click", async () => {
    const r = await api("/api/labeler", { action: "delete_static", name: b.dataset.name });
    $("labStatus").textContent = r.status || r.error;
  }));
}
$("stSlider").addEventListener("input", () => {
  $("stHeading").value = $("stSlider").value;
  live.headPreview = { hat: $("stHat").value, deg: Number($("stSlider").value) }; draw();
});
$("stSlider").addEventListener("change", () => { live.headPreview = null; draw(); });
$("stAdd").addEventListener("click", async () => {
  const r = await api("/api/labeler", { action: "add_static", hat: $("stHat").value, name: $("stName").value,
                                        heading: $("stHeading").value });
  $("labStatus").textContent = r.status || r.error;
  if (r.ok) { $("stName").value = ""; $("stHeading").value = ""; }
});
$("labSave").addEventListener("click", async () => { const r = await api("/api/labeler", { action: "save" }); $("labStatus").textContent = r.status || r.error; });

// ── robot tab ────────────────────────────────────────────────────────
document.querySelectorAll(".robot-btns button").forEach(b => b.addEventListener("click", async () => {
  const cmd = b.dataset.cmd;
  if (cmd === "damp" && !confirm("Damp: the robot goes limp and falls if it is not supported. Send?")) return;
  $("robotOut").textContent = `${cmd}...`;
  const r = await api("/api/robot", { cmd });
  $("robotOut").textContent = `${cmd}: ${JSON.stringify(r, null, 1)}`;
}));

// ── poll ─────────────────────────────────────────────────────────────
function updateAnchors() {
  const sel = $("liveAnchor"), cur = sel.value;
  const names = ((live.st.agents && live.st.agents.agents) || []).filter(a => a.role === "human").map(a => a.name).sort();
  const want = ["", ...names].join(",");
  if (sel.dataset.names !== want) {
    sel.dataset.names = want;
    sel.innerHTML = `<option value="">(none)</option>` + names.map(n => `<option>${esc(n)}</option>`).join("");
    sel.value = names.includes(cur) ? cur : "";
  }
}
async function poll() {
  try {
    const st = live.st = await api("/api/live");
    const b = $("liveBadge");
    b.textContent = st.dry_run ? "DRY RUN" : "REAL ROBOT";
    b.className = st.dry_run ? "dry" : "real";
    updateAnchors();
    renderLabeler(st.labeler);
    renderExec(st.exec);
    $("liveLog").textContent = st.log.slice().reverse().map(([t, s]) => `${new Date(t * 1000).toLocaleTimeString()}  ${s}`).join("\n");
    const p = st.plan;
    $("approveBtn").disabled = !p || !p.goals.length || (st.exec && ["running", "gesture"].includes(st.exec.state));
    if (p && p.id !== live.planId) {
      const fresh = live.planId !== null && !p.id.startsWith(live.planId.split("~")[0]);
      live.planId = p.id; renderPlan(p);
      if (fresh || !allRuns.some(r => r.run_id === p.id.split("~")[0])) reloadRunList();
    }
    const ex = st.exec;
    if (ex && ex.state === "running" && ex.pose) {
      if (live.walkedId !== ex.plan_id) { live.walkedId = ex.plan_id; live.walked = []; }
      const l = live.walked[live.walked.length - 1];
      if (!l || Math.hypot(l[0] - ex.pose[0], l[1] - ex.pose[1]) > 0.02) live.walked.push([ex.pose[0], ex.pose[1]]);
    }
    updateLiveScene();
  } catch (e) { console.warn("poll", e); }
  setTimeout(poll, LIVE_POLL_MS);
}

loadOptions().then(() => { selectRun("live"); poll(); });
