import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

// ─── palette ─────────────────────────────────────────────────────────────────
const BG      = 0x0D1117;
const TINT_A  = new THREE.Vector3(0.98, 0.78, 0.31);   // person A — yellow
const TINT_B  = new THREE.Vector3(0.26, 0.85, 0.75);   // person B — teal
const DIM     = new THREE.Vector3(0.30, 0.30, 0.30);    // grey for the rest
const TINT_R  = 1.3;                                    // radius of person tint (m)
const BOX_HI_A = 0xF9C74F, BOX_HI_B = 0x43D9AD;
const BOX_DIM  = 0x2A6FBF;
const PT_SIZE  = 0.03;

// ─── helpers ─────────────────────────────────────────────────────────────────
function b64f32(s) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

function boxEdgePositions(b) {
  const [cx, cy, cz, dx, dy, dz, yaw] = b;
  const hw = dx / 2, hh = dy / 2, ht = dz / 2;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const cs = [];
  for (const z of [cz - ht, cz + ht])
    for (const [lx, ly] of [[hw, hh], [hw, -hh], [-hw, -hh], [-hw, hh]])
      cs.push([c * lx - s * ly + cx, s * lx + c * ly + cy, z]);
  const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const pos = new Float32Array(24 * 3);
  E.forEach(([a, k], i) => {
    pos[i * 6]     = cs[a][0]; pos[i * 6 + 1] = cs[a][1]; pos[i * 6 + 2] = cs[a][2];
    pos[i * 6 + 3] = cs[k][0]; pos[i * 6 + 4] = cs[k][1]; pos[i * 6 + 5] = cs[k][2];
  });
  return pos;
}

// ─── one interactive 3D panel ────────────────────────────────────────────────
class Panel {
  constructor(cId, titleId, tagId, hiColor) {
    this.container = document.getElementById(cId);
    this.titleEl   = document.getElementById(titleId);
    this.tagEl     = document.getElementById(tagId);
    this.hiColor   = hiColor;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setClearColor(BG);
    this.container.appendChild(this.renderer.domElement);

    this.scene  = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.14;
    this.controls.minDistance = 0.2;

    this.objs = [];
    this.fitTarget = new THREE.Vector3();
    this.fitRadius = 3;

    this.renderer.domElement.addEventListener('dblclick', () =>
      this.fit(this.fitTarget, this.fitRadius));
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  fit(c, r) {
    this.controls.target.copy(c);
    this.camera.near = Math.max(0.005, r / 200);
    this.camera.far  = r * 200;
    this.camera.position.set(c.x + r * 0.85, c.y - r * 1.05, c.z + r * 0.62);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  clear() {
    for (const o of this.objs) {
      this.scene.remove(o);
      o.geometry?.dispose?.();
      o.material?.dispose?.();
    }
    this.objs = [];
  }

  load(ptsF32, hiIdx, peds, title, tag) {
    this.clear();
    const hi = peds[hiIdx].box;

    // points, tinted near the highlighted person
    const n = ptsF32.length / 3;
    const col = new Float32Array(n * 3);
    const t = this === panelA ? TINT_A : TINT_B;
    for (let i = 0; i < n; i++) {
      const x = ptsF32[i*3] - hi[0], y = ptsF32[i*3+1] - hi[1], z = ptsF32[i*3+2] - hi[2];
      const v = (x*x + y*y + z*z) <= TINT_R*TINT_R ? t : DIM;
      col[i*3] = v.x; col[i*3+1] = v.y; col[i*3+2] = v.z;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(ptsF32, 3));
    g.setAttribute('color',    new THREE.BufferAttribute(col, 3));
    const points = new THREE.Points(g, new THREE.PointsMaterial({
      size: PT_SIZE, vertexColors: true, sizeAttenuation: true }));
    this.scene.add(points); this.objs.push(points);

    // boxes: highlighted bold, others dim
    for (const p of peds)
      this.addBox(p.box, p.hi ? this.hiColor : BOX_DIM);

    // reference grid on the local floor (box bottom)
    const grid = new THREE.GridHelper(10, 20, 0x39445C, 0x1C2333);
    grid.position.z = hi[2] - hi[5] / 2;
    this.scene.add(grid); this.objs.push(grid);

    this.titleEl.textContent = title;
    this.tagEl.textContent = tag;
    this.fitTarget.set(hi[0], hi[1], hi[2]);
    this.fitRadius = Math.max(2.6, hi[5] * 2.4);
    this.fit(this.fitTarget, this.fitRadius);
  }

  /** Lightweight update: swap point cloud + boxes without refitting camera. */
  loadFrame(ptsF32, hiIdx, peds) {
    this.clear();
    const hi = peds[hiIdx].box;

    const n = ptsF32.length / 3;
    const col = new Float32Array(n * 3);
    const t = this === panelA ? TINT_A : TINT_B;
    for (let i = 0; i < n; i++) {
      const x = ptsF32[i*3] - hi[0], y = ptsF32[i*3+1] - hi[1], z = ptsF32[i*3+2] - hi[2];
      const v = (x*x + y*y + z*z) <= TINT_R*TINT_R ? t : DIM;
      col[i*3] = v.x; col[i*3+1] = v.y; col[i*3+2] = v.z;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(ptsF32, 3));
    g.setAttribute('color',    new THREE.BufferAttribute(col, 3));
    const points = new THREE.Points(g, new THREE.PointsMaterial({
      size: PT_SIZE, vertexColors: true, sizeAttenuation: true }));
    this.scene.add(points); this.objs.push(points);

    for (const p of peds)
      this.addBox(p.box, p.hi ? this.hiColor : BOX_DIM);

    const grid = new THREE.GridHelper(10, 20, 0x39445C, 0x1C2333);
    grid.position.z = hi[2] - hi[5] / 2;
    this.scene.add(grid); this.objs.push(grid);
  }

  addBox(b, color) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(boxEdgePositions(b), 3));
    const ls = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
    this.scene.add(ls); this.objs.push(ls);
  }

  render() { this.renderer.render(this.scene, this.camera); }
}

// ─── app state ───────────────────────────────────────────────────────────────
const panelA = new Panel('A-canvas', 'A-title', 'A-tag', BOX_HI_A);
const panelB = new Panel('B-canvas', 'B-title', 'B-tag', BOX_HI_B);

const statusEl = document.getElementById('status');
const doneEl   = document.getElementById('done');
const doneBox  = document.getElementById('done-box');

// info bar elements
const infoA    = document.getElementById('info-a');
const infoB    = document.getElementById('info-b');
const infoCos  = document.getElementById('info-cos');
const infoGap  = document.getElementById('info-gap');
const infoTrack= document.getElementById('info-track');
const playBtn  = document.getElementById('play-btn');
const playSpeed= document.getElementById('play-speed');

let INIT = null, idx = 0, counts = { yes: 0, no: 0, skip: 0 }, busy = false, finished = false;

// playback state
let playing = false, playAbort = null, playSeq = null, playFrameIdx = 0;

const key = p => `${p.f0}_${p.k0}_${p.f1}_${p.k1}`;

function setCounts(c) { counts = c || counts; }

// ─── info bar ────────────────────────────────────────────────────────────────
function updateInfoBar(meta) {
  const h0 = meta.h0, h1 = meta.h1, w0 = meta.w0, w1 = meta.w1;
  infoA.textContent = `f${meta.f0} k${meta.k0}  h=${(h0*100).toFixed(0)}cm  w=${(w0*100).toFixed(0)}cm`;
  infoB.textContent = `f${meta.f1} k${meta.k1}  h=${(h1*100).toFixed(0)}cm  w=${(w1*100).toFixed(0)}cm`;

  if (meta.cos_sim != null) {
    const cs = meta.cos_sim;
    const cls = cs >= 0.8 ? 'cos-high' : cs >= 0.5 ? 'cos-mid' : 'cos-low';
    infoCos.textContent = cs.toFixed(3);
    infoCos.className = 'cos ' + cls;
  } else {
    infoCos.textContent = 'n/a';
    infoCos.className = 'cos';
  }

  infoGap.textContent = `${meta.gap} frames`;
  infoTrack.textContent = '';
}

function statusPair(p, extra) {
  statusEl.innerHTML =
    `Pair <b>${idx+1}/${INIT.n_pairs}</b> &nbsp;·&nbsp; frame <b>${p.f0} → ${p.f1}</b> ` +
    `(+${p.gap*100} ms) &nbsp;·&nbsp; centre dist <b>${p.dist.toFixed(2)}</b>` +
    ` &nbsp;·&nbsp; ✓same <b>${counts.yes}</b> &nbsp; ✗diff <b>${counts.no}</b> &nbsp; →skip <b>${counts.skip}</b>` +
    (extra ? ` &nbsp;·&nbsp; ${extra}` : '');
}

function setPanel(P, side, d, meta) {
  const sd = d[side];                                   // side = 'a' | 'b' (JSON key)
  const pts = b64f32(sd.pts_b64);
  const hiIdx = Math.max(0, sd.peds.findIndex(p => p.hi));
  const hi = sd.peds[hiIdx];
  const title = `FRAME ${side === 'a' ? meta.f0 : meta.f1}`;
  const tag = `person #${side === 'a' ? meta.k0 : meta.k1} · s=${hi.score.toFixed(2)}`;
  P.load(pts, hiIdx, sd.peds, title, tag);
}

async function show(j) {
  if (!INIT || j < 0 || j >= INIT.n_pairs) return;
  statusEl.textContent = `Loading pair ${j+1}/${INIT.n_pairs} …`;
  const r = await fetch(`/api/pairdata?j=${j}`);
  const d = await r.json();
  if (d.error) { statusEl.textContent = `Error: ${d.error}`; return; }
  setPanel(panelA, 'a', d, d.meta);
  setPanel(panelB, 'b', d, d.meta);
  idx = j;
  updateInfoBar(d.meta);
  playBtn.disabled = false;
  const unans = Object.keys(INIT.answered).length < INIT.n_pairs;
  statusPair(d.meta, unans ? null : '<b>all pairs answered — ← to revisit</b>');
}

async function verdict(v) {
  if (busy || finished || !INIT) return;
  busy = true;
  try {
    const p = INIT.pairs[idx];
    const r = await fetch('/api/verdict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ j: idx, verdict: v }),
    });
    const res = await r.json();
    if (res.error) { statusEl.textContent = `Error: ${res.error}`; return; }
    setCounts(res.counts);
    INIT.answered[key(p)] = v;
    const label = { yes: '✓ same', no: '✗ different', skip: '→ skipped' }[v];
    if (idx + 1 < INIT.n_pairs) {
      statusPair(p, `recorded ${label} — next…`);
      await show(idx + 1);
    } else {
      finish();
    }
  } finally {
    busy = false;
  }
}

function prev() { if (idx > 0) show(idx - 1); }

function finish() {
  finished = true;
  doneBox.innerHTML =
    `Session <b>${INIT.session}</b> — all ${INIT.n_pairs} pair(s) handled<br>` +
    `<span style="color:#F9C74F">✓ same: ${counts.yes}</span> &nbsp; ` +
    `<span style="color:#FF7B72">✗ different: ${counts.no}</span> &nbsp; ` +
    `<span style="color:#8B949E">skipped: ${counts.skip}</span><br>` +
    `<span style="color:#8B949E;font-size:12px">saved → reid_data/sameperson_${INIT.session}.json</span>`;
  doneEl.style.display = 'flex';
  doneEl.onclick = () => doneEl.style.display = 'none';
}

// ─── playback: animate f0 → f1 on Panel B ───────────────────────────────────
async function startPlayback() {
  if (playing || !INIT) return;
  const p = INIT.pairs[idx];
  playing = true;
  playBtn.textContent = '■ Stop';
  playBtn.disabled = false;
  infoTrack.textContent = 'fetching sequence…';

  playAbort = new AbortController();
  try {
    const r = await fetch(`/api/sequence?f0=${p.f0}&k0=${p.k0}&f1=${p.f1}`,
                           { signal: playAbort.signal });
    const d = await r.json();
    if (!d.sequence || d.sequence.length === 0) {
      infoTrack.textContent = 'no sequence data';
      stopPlayback();
      return;
    }
    playSeq = d.sequence;
    playFrameIdx = 0;
    infoTrack.textContent = `playing 0/${playSeq.length}`;

    const fps = Math.max(2, Math.min(60, parseInt(playSpeed.value) || 10));
    const delay = 1000 / fps;

    for (let i = 0; i < playSeq.length && playing; i++) {
      playFrameIdx = i;
      const frame = playSeq[i];
      const pts = b64f32(frame.pts_b64);
      const hiIdx = frame.hi_idx >= 0 ? frame.hi_idx : 0;

      panelB.loadFrame(pts, hiIdx, frame.peds);
      panelB.titleEl.textContent = `FRAME B  ${frame.fi} / ${p.f1}` +
        (frame.track_lost ? '  ⚠ track lost' : '');

      const dz = frame.peds[hiIdx]?.box[5];
      infoTrack.textContent = `${i+1}/${playSeq.length}  f${frame.fi}` +
        (dz ? `  h=${(dz*100).toFixed(0)}cm` : '') +
        (frame.track_lost ? '  ⚠ lost' : '');

      await new Promise((res, reject) => {
        const t = setTimeout(res, delay);
        playAbort.signal.addEventListener('abort', () => { clearTimeout(t); reject(); }, { once: true });
      });
    }
  } catch (e) {
    if (e.name !== 'AbortError') infoTrack.textContent = `play error: ${e}`;
  } finally {
    if (playing) stopPlayback();
  }
}

function stopPlayback() {
  playing = false;
  if (playAbort) playAbort.abort();
  playAbort = null;
  playSeq = null;
  playBtn.textContent = '▶ Play A→B';
  // restore original Frame B
  if (INIT && idx >= 0 && idx < INIT.n_pairs) show(idx);
}

playBtn.addEventListener('click', () => {
  if (playing) stopPlayback();
  else startPlayback();
});

// ─── input wiring ────────────────────────────────────────────────────────────
document.getElementById('b-yes').onclick  = () => verdict('yes');
document.getElementById('b-no').onclick   = () => verdict('no');
document.getElementById('b-skip').onclick = () => verdict('skip');
document.getElementById('b-prev').onclick = prev;
document.getElementById('b-quit').onclick = finish;

window.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  if (e.key === 'y' || e.key === 'Y') verdict('yes');
  else if (e.key === 'n' || e.key === 'N') verdict('no');
  else if (e.key === ' ') { e.preventDefault(); verdict('skip'); }
  else if (e.key === 'q' || e.key === 'Q') finish();
  else if (e.key === 'ArrowLeft') prev();
  else if (e.key === 'p' || e.key === 'P') { e.preventDefault(); playBtn.click(); }
});

window.addEventListener('resize', () => { panelA.resize(); panelB.resize(); });

// debug/test handle (used by Playwright render checks)
window.__reid = {
  get A() { return panelA; }, get B() { return panelB; },
  snapshot() {
    const snap = {};
    for (const [nm, P] of [['A', panelA], ['B', panelB]]) {
      const pts = P.scene.children.find(o => o.isPoints);
      const col = pts ? pts.geometry.getAttribute('color') : null;
      let nonzero = 0, nan = false;
      if (col) {
        for (let i = 0; i < col.array.length; i++) if (!isFinite(col.array[i])) { nan = true; break; }
        for (let i = 0; i < col.count && !nan; i++)
          nonzero += (col.array[i*3] || col.array[i*3+1] || col.array[i*3+2]) > 0.05;
      }
      snap[nm] = {
        children: P.scene.children.map(o => o.type),
        points: pts ? { n: col.count, nonzero, nan } : null,
        info: P.renderer.info.render,
        cam: { pos: P.camera.position.toArray().map(v => +v.toFixed(2)),
               tgt: P.controls.target.toArray().map(v => +v.toFixed(2)) },
      };
    }
    snap.status = document.getElementById('status').textContent;
    snap.titleA = document.getElementById('A-title').textContent;
    snap.titleB = document.getElementById('B-title').textContent;
    return snap;
  },
};

(function loop() {
  requestAnimationFrame(loop);
  panelA.controls.update(); panelB.controls.update();
  panelA.render();          panelB.render();
})();

// ─── init ────────────────────────────────────────────────────────────────────
(async () => {
  try {
    INIT = await (await fetch('/api/init')).json();
    setCounts(INIT.counts);
    idx = INIT.pairs.findIndex(p => !(key(p) in INIT.answered));
    if (idx < 0) idx = INIT.n_pairs - 1;
    if (!INIT.n_pairs) { statusEl.textContent = 'No pairs found on the server side.'; return; }
    await show(idx);
  } catch (err) {
    statusEl.innerHTML = `Failed to reach server: ${err}`;
  }
})();
