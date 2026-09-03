import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const BG = 0x0D1117;
const COL = { OK: 0x43D9AD, I: 0xF9C74F, M: 0xFF7B72 };
const DIM_OPACITY = 0.13;
const PPS = 10;                    // point cloud rate (frames / second)

// ─── scene setup ─────────────────────────────────────────────────────────────
const container = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setClearColor(BG);
container.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.14;

const grid = new THREE.GridHelper(30, 30, 0x39445C, 0x1C2333);
scene.add(grid);

let D = null, tracks = [], selected = null;
let N = 0, playhead = 0, playing = false, speed = 1.0;
const personRefs = [];   // {tid, t, mats:[], bbox}
const hitObjs = [];      // raycast targets (lines + start/end markers)
const markerObjs = [];   // {tid, mesh, label, t}   (moving person markers)
const ALL_BBOX = new THREE.Box3();
let cloud = null, cloudFi = -1;
let lastCloudShown = null;
let meshFaces = null, meshAvailable = false;

function b64f32(s) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

function b64i32(s) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Int32Array(u8.buffer);
}

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function fitBox(box) {
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.length() / 2, 1.2);
  controls.target.copy(c);
  camera.near = Math.max(0.005, r / 200);
  camera.far = r * 400;
  camera.position.set(c.x + r * 0.9, c.y - r * 0.9, c.z + r * 0.65);
  camera.updateProjectionMatrix();
  controls.update();
}

function fitAll() {
  if (!ALL_BBOX.isEmpty() && personRefs.length) {
    fitBox(ALL_BBOX.clone().expandByScalar(0.5));
  } else {
    controls.target.set(0, 0, 0);
    camera.position.set(8, -10, 6);
    controls.update();
  }
}

// ─── build trajectories + markers ────────────────────────────────────────────
// diff >= SPAN_DIFF means that 10+ frames were missing between two trusted
// detections (an absence bridge, i.e. a former "sudden jump"): draw it as a
// faint connector, while the solid run is the actual observed/interpolated path
const SPAN_DIFF = 11;

function buildTracks() {
  for (const t of tracks) {
    const pts = t.pts;
    const runs = [];
    let run = [];
    const spanPairs = [];
    for (let i = 0; i < pts.length; i++) {
      if (i > 0 && pts[i][0] - pts[i - 1][0] >= SPAN_DIFF) {
        if (run.length) { runs.push(run); run = []; }
        spanPairs.push(new THREE.Vector3(pts[i - 1][1], pts[i - 1][2], pts[i - 1][3]));
        spanPairs.push(new THREE.Vector3(pts[i][1], pts[i][2], pts[i][3]));
      }
      run.push(new THREE.Vector3(pts[i][1], pts[i][2], pts[i][3]));
    }
    if (run.length) runs.push(run);

    const col = COL[t.st] ?? 0x8B949E;
    const ref = { tid: t.tid, t, mats: [], bbox: null };

    for (const r of runs) {
      const mat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 1 });
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(r), mat);
      line.userData.tid = t.tid;
      scene.add(line);
      ref.mats.push(mat);
      hitObjs.push(line);
    }
    if (spanPairs.length) {                       // absence bridges: faint
      const smat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.22 });
      const sgeo = new THREE.BufferGeometry().setFromPoints(spanPairs);
      const sline = new THREE.LineSegments(sgeo, smat);
      sline.userData.tid = t.tid;
      scene.add(sline);
      ref.mats.push(smat);
      hitObjs.push(sline);
    }
    const pts3 = t.pts.map(p => new THREE.Vector3(p[1], p[2], p[3]));
    if (pts3.length) ref.bbox = new THREE.Box3().setFromPoints(pts3);

    // start / end markers
    const mk = (p, c) => {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.07, 12, 12),
                               new THREE.MeshBasicMaterial({ color: c, transparent: true }));
      s.position.copy(p);
      s.userData.tid = t.tid;
      scene.add(s);
      ref.mats.push(s.material);
      hitObjs.push(s);
    };
    if (pts3.length) {
      mk(pts3[0], 0xFFFFFF);
      mk(pts3[pts3.length - 1], t.st === 'M' ? 0xFF7B72 : 0x8B949E);
    }

    // moving person marker + HTML label
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.11, 16, 16),
      new THREE.MeshBasicMaterial({ color: col, transparent: true }));
    mesh.renderOrder = 10;
    scene.add(mesh);
    const label = document.createElement('div');
    label.className = 'tag';
    label.textContent = 'T' + t.tid;
    label.style.borderColor = '#' + col.toString(16).padStart(6, '0');
    container.appendChild(label);
    markerObjs.push({ tid: t.tid, mesh, label, t });

    personRefs.push(ref);
    if (ref.bbox) ALL_BBOX.union(ref.bbox);
  }
}

function posAt(t, fi) {
  const pts = t.pts;
  if (!pts.length || fi < pts[0][0] || fi > pts[pts.length - 1][0]) return null;
  let lo = 0, hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (pts[mid][0] < fi) lo = mid + 1; else hi = mid;
  }
  if (pts[lo][0] === fi) return new THREE.Vector3(pts[lo][1], pts[lo][2], pts[lo][3]);
  const b = pts[lo], a = pts[lo - 1];
  const u = (fi - a[0]) / (b[0] - a[0]);
  return new THREE.Vector3(a[1] + (b[1] - a[1]) * u,
                           a[2] + (b[2] - a[2]) * u,
                           a[3] + (b[3] - a[3]) * u);
}

function updateMarkers(fi) {
  const w = container.clientWidth, h = container.clientHeight;
  for (const m of markerObjs) {
    const p = posAt(m.t, fi);
    const dimmed = selected !== null && selected !== m.tid;
    if (!p) { m.mesh.visible = false; m.label.style.display = 'none'; continue; }
    m.mesh.visible = true;
    m.mesh.material.opacity = dimmed ? 0.25 : 1;
    const v = p.clone().project(camera);
    if (v.z > 1 || v.z < -1) { m.label.style.display = 'none'; continue; }
    m.label.style.display = 'block';
    m.label.style.opacity = dimmed ? 0.25 : 1;
    m.label.style.left = ((v.x * 0.5 + 0.5) * w) + 'px';
    m.label.style.top = ((-v.y * 0.5 + 0.5) * h) + 'px';
  }
}

function focus(tid) {
  selected = tid;
  for (const o of personRefs) {
    const on = tid === null || o.tid === tid;
    for (const m of o.mats) m.opacity = on ? 1 : DIM_OPACITY;
  }
  document.querySelectorAll('#rows tr').forEach(tr =>
    tr.classList.toggle('sel', Number(tr.dataset.tid) === tid));
  if (tid !== null) {
    const o = personRefs.find(x => x.tid === tid);
    if (o && o.bbox) fitBox(o.bbox.clone().expandByScalar(0.3));
    const row = document.querySelector(`#rows tr[data-tid="${tid}"]`);
    if (row) row.scrollIntoView({ block: 'nearest' });
  }
}

// ─── click a trajectory in 3D ────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Line.threshold = 0.35;
let downXY = null;
renderer.domElement.addEventListener('pointerdown', e => {
  if (e.button === 0) downXY = [e.clientX, e.clientY];
});
renderer.domElement.addEventListener('pointerup', e => {
  if (e.button !== 0 || !downXY) return;
  const dx = e.clientX - downXY[0], dy = e.clientY - downXY[1];
  downXY = null;
  if (dx * dx + dy * dy > 36) return;        // was a drag, not a click
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                                -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(hitObjs, false);
  focus(hits.length ? hits[0].object.userData.tid : null);
});

// ─── point cloud context (pre-allocated dynamic buffer, latest-wins) ─────────
const MAX_CLOUD_PTS = 65536;
if (cloud === null) {
  const g = new THREE.BufferGeometry();
  const pos = new THREE.BufferAttribute(new Float32Array(MAX_CLOUD_PTS * 3), 3);
  pos.setUsage(THREE.DynamicDrawUsage);
  const col = new THREE.BufferAttribute(new Float32Array(MAX_CLOUD_PTS * 3), 3);
  col.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute('position', pos);
  g.setAttribute('color', col);
  cloud = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.02, vertexColors: true,
    sizeAttenuation: true, transparent: true, opacity: 0.8 }));
  cloud.frustumCulled = false;
  scene.add(cloud);
}

async function fetchCloud(fi) {
  const d = await fetch(`/api/framecloud?fi=${fi}`).then(r => r.json()).catch(() => null);
  if (!d || d.error) return false;
  const a = b64f32(d.pts_b64);
  const n = Math.min(a.length / 3, MAX_CLOUD_PTS);
  const pos = cloud.geometry.getAttribute('position').array;
  const col = cloud.geometry.getAttribute('color').array;
  for (let i = 0; i < n; i++) {
    pos[i * 3] = a[i * 3]; pos[i * 3 + 1] = a[i * 3 + 1]; pos[i * 3 + 2] = a[i * 3 + 2];
    const z = a[i * 3 + 2];
    const v = (z > -0.3 && z < 1.7) ? 0.62 : 0.33;
    col[i * 3] = v; col[i * 3 + 1] = v; col[i * 3 + 2] = v + 0.05;
  }
  cloud.geometry.setDrawRange(0, n);
  cloud.geometry.getAttribute('position').needsUpdate = true;
  cloud.geometry.getAttribute('color').needsUpdate = true;
  cloud.visible = true;
  cloudFi = fi;
  document.getElementById('cloudlabel').textContent = `cloud: frame ${fi}`;
  return true;
}

// single in-flight fetch; each completion immediately requests the LATEST
// wanted frame, so frames are never queued up and playback stays smooth
let cloudBusy = false, cloudWanted = -1;
function setCloudTarget(fi) {
  cloudWanted = fi;
  pumpCloud();
}
function pumpCloud() {
  if (cloudBusy || cloudWanted < 0 || cloudWanted === cloudFi) return;
  cloudBusy = true;
  fetchCloud(cloudWanted).then(ok => {
    cloudBusy = false;
    if (ok && cloudWanted !== cloudFi) pumpCloud();   // continue catching up
    else if (!ok && cloudWanted >= 0) {               // fetch failed -> retry
      document.getElementById('cloudlabel').textContent = 'cloud: ✗ fetch failed – retrying…';
      setTimeout(pumpCloud, 800);
    }
  });
}

function clearCloud() {
  if (cloud) cloud.visible = false;
  cloudWanted = -1;
  cloudFi = -1;
  document.getElementById('cloudlabel').textContent = 'cloud: —';
}

// ─── SMPL mesh (one per track, all rendered simultaneously so multiple
// people can be watched travelling at once — SMPL-mode sessions only) ───────
const meshPool = new Map();   // tid -> {mesh, busy, shownKey}

async function initMesh() {
  const d = await fetch('/api/meshfaces').then(r => r.ok ? r.json() : null).catch(() => null);
  if (!d) { meshAvailable = false; return; }
  meshFaces = Uint32Array.from(b64i32(d.faces_b64));   // WebGL index buffers need unsigned
  meshAvailable = true;
  for (const t of tracks) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6890 * 3), 3));
    g.setIndex(new THREE.BufferAttribute(meshFaces, 1));
    const col = COL[t.st] ?? 0x8B949E;
    const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.55,
      side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(g, mat);
    mesh.visible = false;
    mesh.frustumCulled = false;
    scene.add(mesh);
    meshPool.set(t.tid, { mesh, busy: false, shownKey: null });
    // hook into the existing focus()-dimming logic (personRefs[].mats)
    const ref = personRefs.find(r => r.tid === t.tid);
    if (ref) ref.mats.push(mat);
  }
}

function pumpMeshFor(tid, fi) {
  const st = meshPool.get(tid);
  if (!st || st.busy) return;
  const key = tid + ':' + fi;
  if (key === st.shownKey) return;
  st.busy = true;
  fetch(`/api/mesh?tid=${tid}&fi=${fi}`).then(r => r.ok ? r.json() : null).then(d => {
    st.busy = false;
    if (d && d.verts_b64) {
      const v = b64f32(d.verts_b64);
      st.mesh.geometry.getAttribute('position').array.set(v);
      st.mesh.geometry.getAttribute('position').needsUpdate = true;
      st.mesh.geometry.computeVertexNormals();
      st.mesh.visible = true;
      st.shownKey = key;
    } else {
      st.mesh.visible = false;
      st.shownKey = null;
    }
  }).catch(() => { st.busy = false; });
}

function updateMeshes(fi) {
  if (!meshAvailable) return;
  if (!meshon.checked) { clearAllMeshes(); return; }
  for (const t of tracks) {
    const st = meshPool.get(t.tid);
    if (!st) continue;
    const p = posAt(t, fi);
    if (!p) { st.mesh.visible = false; st.shownKey = null; continue; }
    pumpMeshFor(t.tid, fi);
  }
}

function clearAllMeshes() {
  for (const st of meshPool.values()) { st.mesh.visible = false; st.shownKey = null; }
}

// ─── table / stats ───────────────────────────────────────────────────────────
// absence spans = consecutive trusted (or interpolated) points with a gap of
// 2+ frames (i.e. diff >= SPAN_DIFF, the same condition used for the faint
// bridges); small 1-frame interp gaps are already in the pts stream
function gapsOf(t) {
  const out = [];
  for (let i = 1; i < t.pts.length; i++) {
    const a = t.pts[i - 1][0], b = t.pts[i][0];
    if (b - a >= SPAN_DIFF) out.push({ from: a + 1, to: b - 1, miss: b - a - 1 });
  }
  return out;
}

function toggleGaps(tr, gaps) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains('gaps')) { next.remove(); return; }
  const tr2 = document.createElement('tr');
  tr2.className = 'gaps';
  tr2.dataset.tid = tr.dataset.tid;
  const td = document.createElement('td');
  td.colSpan = 7;
  const tot = gaps.reduce((s, g) => s + g.miss, 0);
  const div = document.createElement('div');
  div.className = 'gaplist';
  div.innerHTML = `<b>${gaps.length} absence${gaps.length > 1 ? 's' : ''} · ` +
    `${tot.toLocaleString()} missing frames (≈ ${(tot / PPS).toFixed(1)} s) of the track</b>`;
  for (const g of gaps) {
    const it = document.createElement('span');
    it.className = 'gap';
    it.textContent = `f${g.from}–${g.to}  (${g.miss} f ≈ ${(g.miss / PPS).toFixed(1)} s)`;
    it.title = 'Jump the playhead to the end of the last trusted frame before this absence';
    it.onclick = () => {
      const start = Math.max(0, g.from - 1 - 5);
      playhead = start; slider.value = start;
      if (cbox.checked) setCloudTarget(start);
    };
    div.appendChild(it);
  }
  td.appendChild(div);
  tr2.appendChild(td);
  tr.after(tr2);
}

function buildTable() {
  const rows = document.getElementById('rows');
  rows.innerHTML = '';
  for (const t of tracks) {
    const tr = document.createElement('tr');
    tr.dataset.tid = t.tid;
    const gaps = gapsOf(t);
    const caret = t.nm
      ? ` <span class="caret" title="click to list every missing frame">⌄ ${t.nm}f</span>`
      : '';
    tr.innerHTML = `<td>T${t.tid}${caret}</td><td>${t.last}</td><td>${t.len + t.ni}</td>` +
      `<td>${t.ni || '·'}</td><td>${t.nm || '·'}</td><td>${t.nj || '·'}</td>` +
      `<td class="st-${t.st}">${t.st}</td>`;
    tr.onclick = () => focus(t.tid);
    const c = tr.querySelector('.caret');
    if (c) c.onclick = e => { e.stopPropagation(); toggleGaps(tr, gaps); };
    rows.appendChild(tr);
  }
}

function buildStats() {
  const s = D.stats, p = D.params;
  document.getElementById('sess').textContent =
    `${D.session} · ${D.n_frames} frames · s≥${p.min_score} · ≤${p.people} people · ${p.max_speed} m/s gate`;
  document.getElementById('stats').innerHTML =
    `people: <b>${s.n_tracks}</b> &nbsp; <span class="st-OK">${s.ok} clean</span> ` +
    `· <span class="st-I">${s.interp} w/interp</span> · <span class="st-M">${s.missing} w/absence</span><br>` +
    `frames: <b>${s.n_matched_frames.toLocaleString()}</b> matched · ` +
    `<b>${s.n_interp_frames.toLocaleString()}</b> interpolated · ` +
    `<span class="st-M">${s.n_missing_frames.toLocaleString()} missing</span><br>` +
    `<span class="st-I">${s.n_jumps.toLocaleString()} implausible jumps rejected</span> · ` +
    `${s.n_unassigned.toLocaleString()} boxes dropped as clutter<br>` +
    `longest: <b>${s.longest}</b> frames (${(s.longest / PPS).toFixed(0)} s)`;
}

// ─── playback ────────────────────────────────────────────────────────────────
const slider = document.getElementById('frameslider');
const cbox = document.getElementById('cloudon');
const meshon = document.getElementById('meshon');
const playbtn = document.getElementById('playbtn');
const speedSel = document.getElementById('speed');
const followEl = document.getElementById('follow');

function setPlay(on) {
  playing = on;
  playbtn.innerHTML = on ? '&#10074;&#10074;' : '&#9654;';
}

function advance(k) {
  playhead = Math.min(N - 1, playhead + k);
  slider.value = playhead;
  if (cbox.checked) setCloudTarget(playhead);
  if (playhead >= N - 1) setPlay(false);
}

slider.addEventListener('input', () => {
  playhead = Number(slider.value);
  if (cbox.checked) setCloudTarget(playhead);
});
cbox.addEventListener('change', () => {
  playhead = Number(slider.value);
  cbox.checked ? setCloudTarget(playhead) : clearCloud();
});
meshon.addEventListener('change', () => {
  if (meshon.checked) updateMeshes(playhead);
  else clearAllMeshes();
});
speedSel.addEventListener('change', () => { speed = Number(speedSel.value); });
playbtn.addEventListener('click', () => {
  if (!playing && playhead >= N - 1) {          // restarting from the end
    playhead = 0; slider.value = 0;
    if (cbox.checked) setCloudTarget(0);
  }
  setPlay(!playing);
});
renderer.domElement.addEventListener('dblclick', fitAll);

let lastT = performance.now(), acc = 0;
(function tick(now) {
  requestAnimationFrame(tick);
  const t0 = performance.now(), dt = t0 - lastT; lastT = t0;
  if (playing) {
    acc += dt * speed;                            // ms of session time
    const adv = Math.floor(acc / (1000 / PPS));   // frames
    if (adv > 0) { acc -= adv * (1000 / PPS); advance(adv); }
  }
  if (followEl.checked && selected !== null) {
    const o = personRefs.find(x => x.tid === selected);
    if (o) {
      const p = posAt(o.t, playhead);
      if (p) controls.target.lerp(p, 0.12);
    }
  }
  controls.update();
  updateMarkers(playhead);
  updateMeshes(playhead);
  renderer.render(scene, camera);
})();

// ─── debug / test hook ───────────────────────────────────────────────────────
window.__reid = {
  get playhead() { return playhead; },
  get playing() { return playing; },
  focus, posAt, camera,
  get personRefs() { return personRefs; },
};

// ─── init ────────────────────────────────────────────────────────────────────
(async () => {
  try {
    D = await (await fetch('/api/tracks')).json();
    tracks = D.tracks;
    N = D.n_frames;
    if (!tracks.length) {
      document.getElementById('stats').textContent = 'No person tracks found for this session/params.';
      return;
    }
    slider.max = N - 1;
    playhead = 0;
    slider.value = 0;
    buildTracks();
    buildTable();
    buildStats();
    resize();
    fitAll();
    await initMesh();
    if (cbox.checked) setCloudTarget(0);
  } catch (err) {
    document.getElementById('stats').innerHTML = `Failed to reach server: ${err}`;
  }
})();
