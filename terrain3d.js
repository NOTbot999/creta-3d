// Streaming lidar terrain for Slovenia (FMDT tiles, D96/TM).  Quadtree level-of-detail:
// 125 m patches at full 1 m detail near the camera, up to 32 km patches at 32 m in the distance.
import * as THREE from 'three';

const SEG = 125;                 // segments per patch side (126 x 126 vertices)
const ROOT = 32000;              // root patch = one level-5 tile (32 km)
const MIN_PATCH = 125;           // smallest patch: 125 m at 1 m spacing

export class TerrainLOD {
  constructor(scene, { baseUrl = '/terrain', workerUrl = '/static/fmdt-worker.js', orthoUrl = null, centreE, centreN, size, exaggeration = 1 } = {}) {
    this.scene = scene; this.baseUrl = baseUrl; this.cE = centreE; this.cN = centreN;
    this.orthoUrl = orthoUrl || ((e0, n0, S, year) => `/ortho?e0=${e0}&n0=${n0}&s=${S}&px=256${year ? '&year=' + year : ''}`);
    // time model (years before today): 0 = measured terrain; see setTime()
    this.years = 0; this.tm = { human: 0, fill: 0, uplift: 0, gravel: 0, climate: 0, photoYear: '' };
    this.uClimate = { value: 0 };
    this.size = size; this.exag = exaggeration;
    this.area = [centreE - size / 2, centreN - size / 2, centreE + size / 2, centreN + size / 2];
    this.group = new THREE.Group(); this.group.name = 'Terrain'; scene.add(this.group);
    this.tiles = new Map();      // 'k/x_y' -> {state, h, min, max, used}
    this.patches = new Map();    // patch key -> {mesh, used}
    this.worker = new Worker(workerUrl);
    this.pending = new Map(); this.nextId = 1; this.inflight = 0;
    this.worker.onmessage = (e) => this._onTile(e.data);
    // layers: relief colours at the bottom, satellite photo on top with adjustable opacity
    this.uRelief = { value: 1 };             // 1 = relief colours on, 0 = neutral grey underneath
    this.uPhoto = { value: 1 };              // satellite opacity (0 = off)
    // ice-age tint: tundra / steppe colours replace forest as uClimate -> 1
    const CLIMATE = 'vec3 relief = mix(vColor.rgb, vec3(0.56, 0.54, 0.43) * (0.75 + 0.5 * vColor.g), uClimate);\n';
    const base = (shader) => {
      shader.uniforms.uRelief = this.uRelief; shader.uniforms.uClimate = this.uClimate;
      shader.fragmentShader = 'uniform float uRelief;\nuniform float uClimate;\n' + shader.fragmentShader
        .replace('#include <color_fragment>', CLIMATE + 'diffuseColor.rgb = mix(vec3(0.62, 0.62, 0.6), relief, uRelief);');
    };
    this.mat = new THREE.MeshLambertMaterial({ vertexColors: true });
    this.mat.onBeforeCompile = base;
    this._photoShader = (shader) => {
      shader.uniforms.uRelief = this.uRelief; shader.uniforms.uPhoto = this.uPhoto; shader.uniforms.uClimate = this.uClimate;
      shader.fragmentShader = 'uniform float uRelief;\nuniform float uPhoto;\nuniform float uClimate;\n' + shader.fragmentShader
        .replace('#include <map_fragment>', '')
        .replace('#include <color_fragment>', CLIMATE +
          'vec3 baseCol = mix(vec3(0.62, 0.62, 0.6), relief, uRelief);\n' +
          'vec4 photo = texture2D(map, vMapUv);\n' +
          'diffuseColor.rgb = mix(baseCol, photo.rgb, uPhoto);');
    };
    this.lodFactor = 2.2; this.frame = 0; this.stats = {};
    this.orthoInflight = 0; this.texLoader = new THREE.TextureLoader();
    this.photoStats = { gursOk: 0, gursFail: 0, esriOk: 0, failed: 0 };
  }

  // ---------------------------------------------------------------- coordinates
  toLocal(E, N, z) { return new THREE.Vector3(E - this.cE, z * this.exag, -(N - this.cN)); }
  toEN(v) { return { E: v.x + this.cE, N: -v.z + this.cN, z: v.y / this.exag }; }

  // ---------------------------------------------------------------- tiles
  _tile(k, x, y, layer = 'tiles') {
    const key = layer === 'tiles' ? `${k}/${x}_${y}` : `${layer}:${k}/${x}_${y}`;
    let t = this.tiles.get(key);
    if (!t) {
      t = { state: 'loading', used: this.frame };
      this.tiles.set(key, t);
      const index = layer === 'tiles' ? this.index : this.layerIndex?.[layer];
      if ((index && !index.levels[k]?.has(`${x}_${y}`)) || (layer !== 'tiles' && !index)) { t.state = 'missing'; return t; }
      const id = this.nextId++; this.pending.set(id, key); this.inflight++;
      this.worker.postMessage({ id, url: `${this.baseUrl}/${layer}/${k}/${x}_${y}.fmdt` });
    }
    t.used = this.frame; return t;
  }
  _onTile(m) {
    const key = this.pending.get(m.id); this.pending.delete(m.id); this.inflight--;
    const t = this.tiles.get(key); if (!t) return;
    if (m.missing || m.error) { t.state = 'missing'; return; }
    Object.assign(t, { state: 'ready', h: m.h, n: m.n, min: m.min, max: m.max });
    // lowest ground of the area (from overview tiles) = level of the main river plain, used by the time model
    if (!key.includes(':') && +key.split('/')[0] >= 3 && Number.isFinite(m.min)) this.zLow = Math.min(this.zLow ?? Infinity, m.min);
  }
  async init() {
    try {
      const j = await (await fetch(`${this.baseUrl}/index.json`)).json();
      this.index = { levels: Object.fromEntries(Object.entries(j.levels).map(([k, v]) => [k, new Set(v)])) };
    } catch { this.index = null; }
    this.layerIndex = {};                    // optional time-slider layers (natural surface, valley envelope)
    for (const layer of ['tiles_nat', 'tiles_env']) {
      try {
        const j = await (await fetch(`${this.baseUrl}/index_${layer}.json`)).json();
        this.layerIndex[layer] = { levels: Object.fromEntries(Object.entries(j.levels).map(([k, v]) => [k, new Set(v)])) };
      } catch { /* layer not built for this area */ }
    }
  }
  get timeAvailable() { return !!(this.layerIndex?.tiles_nat && this.layerIndex?.tiles_env); }

  // ---------------------------------------------------------------- time model
  // years before today -> how the landscape is rebuilt.  0–35: measured terrain + historical photos;
  // >= 40: human-made ground removed (reconstruction); deep time: rivers had cut less (valleys fill
  // towards the envelope at ~0.25 mm/yr), the land stood lower (uplift ~0.2 mm/yr), ice ages left
  // gravel plains on valley floors and tundra instead of forest (illustrative model).
  setTime(years) {
    years = Math.max(0, Math.round(years));
    const GLACIALS = [[20000, 15, 6000], [65000, 8, 8000], [140000, 20, 15000], [250000, 10, 15000], [340000, 12, 15000], [430000, 25, 20000]];
    let gravel = 0, climate = 0;
    for (const [t0, g, w] of GLACIALS) { const e = Math.exp(-(((years - t0) / w) ** 2)); gravel += g * e; climate = Math.max(climate, e); }
    if (years > 9000 && years < 14000) climate = Math.max(climate, 0.25);      // late-glacial / early Holocene
    const photoYear = years <= 5 ? '' : years <= 15 ? '2016' : years <= 24 ? '2006' : years <= 35 ? '1997' : null;
    this.tm = { human: years >= 40 ? 1 : 0, fill: 0.00025 * years, uplift: 0.0002 * years, gravel, climate, photoYear };
    this.uClimate.value = climate;
    const changed = years !== this.years; this.years = years;
    if (changed) this.rebuild();
    return this.tm;
  }
  rebuild() {
    for (const p of this.patches.values()) { p.mesh?.geometry.dispose(); p.mesh?.userData.photo?.map?.dispose(); p.mesh?.userData.photo?.dispose(); }
    this.patches.clear();
  }
  // height at one node given measured (d), natural (n) and envelope (e) values
  _model(d, n, e) {
    const tm = this.tm;
    if (!Number.isFinite(d)) return d;
    let z = tm.human && Number.isFinite(n) ? n : d;
    if (tm.fill > 0 && Number.isFinite(e)) {
      const D = Math.max(0, e - z);
      z += D * tm.fill / (D + tm.fill);                         // valleys fill smoothly towards the envelope
      if (tm.gravel > 0.05) {                                    // ice-age gravel: wide river plains + deep valley floors
        const plain = this.zLow !== undefined ? Math.min(1, Math.max(0, 1 - (z - this.zLow - 5) / 45)) : 0;
        const deep = Math.min(1, Math.max(0, (D - 40) / 110));
        z += tm.gravel * Math.max(plain, deep);
      }
    }
    return z - tm.uplift;
  }
  _timeLayers() { return this.years >= 40 && this.timeAvailable; }

  // ---------------------------------------------------------------- patches
  _src(S) { return Math.max(0, Math.min(5, Math.floor(Math.log2(S / SEG)))); }
  _buildPatch(e0, n0, S) {
    const k = this._src(S), T = 1000 * 2 ** k, sp = 2 ** k;
    const tx = Math.floor(e0 / T), ty = Math.floor(n0 / T);
    const t = this._tile(k, tx, ty);
    if (t.state !== 'ready') return t.state === 'missing' ? 'missing' : null;
    let h = t.h;
    if (this.years > 0) {                       // apply the time model to a copy of the tile window values
      let hn = null, he = null;
      if (this._timeLayers()) {
        const tn = this._tile(k, tx, ty, 'tiles_nat'), te = this._tile(k, tx, ty, 'tiles_env');
        if (tn.state === 'loading' || te.state === 'loading') return null;
        hn = tn.state === 'ready' ? tn.h : null; he = te.state === 'ready' ? te.h : null;
      }
      const key = `${this.years}`;
      if (t.modelKey !== key) {                 // one modelled copy per tile and time
        const m = new Float32Array(h.length);
        for (let i = 0; i < h.length; i++) m[i] = this._model(h[i], hn ? hn[i] : NaN, he ? he[i] : NaN);
        t.modelH = m; t.modelKey = key;
      }
      h = t.modelH;
    }
    const step = (S / SEG) / sp, n = t.n;
    const c0 = (e0 - tx * T) / sp, r0 = ((ty + 1) * T - (n0 + S)) / sp;
    const V = SEG + 1, skirt = 4 * V;
    const pos = new Float32Array((V * V + skirt) * 3), nor = new Float32Array(pos.length), col = new Float32Array(pos.length);
    const uv = new Float32Array((V * V + skirt) * 2), valid = new Uint8Array(V * V + skirt);
    const ce = e0 + S / 2, cn = n0 + S / 2, d = S / SEG, ex = this.exag;
    const H = (r, c) => h[Math.min(n - 1, Math.max(0, r)) * n + Math.min(n - 1, Math.max(0, c))];
    const colr = new THREE.Color();
    let k3 = 0, zmin = Infinity, zmax = -Infinity, any = false;
    const put = (j, i, drop) => {
      const r = r0 + j * step, c = c0 + i * step;
      let z = H(r, c); const good = Number.isFinite(z);
      valid[k3 / 3] = good ? 1 : 0;
      if (!good) z = Number.isFinite(t.min) ? t.min : 0;        // no data: park the vertex, its triangles are dropped
      const gx = (H(r, c + step) - H(r, c - step)) / (2 * d), gy = (H(r - step, c) - H(r + step, c)) / (2 * d);
      const inv = 1 / Math.hypot(gx * ex, 1, gy * ex);
      pos[k3] = e0 + i * d - ce; pos[k3 + 1] = (z - drop) * ex; pos[k3 + 2] = -(n0 + S - j * d - cn);
      nor[k3] = -gx * ex * inv || 0; nor[k3 + 1] = inv || 1; nor[k3 + 2] = gy * ex * inv || 0;   // three z = south
      // hypsometric colour + slope (rock on steep ground)
      const slope = Math.hypot(gx, gy);
      const tt = Math.min(1, Math.max(0, (z - 250) / 1300));
      colr.setHSL(0.27 - 0.17 * tt, 0.35 - 0.15 * tt, 0.30 + 0.25 * tt);
      if (slope > 0.9) colr.lerp(new THREE.Color(0.55, 0.53, 0.5), Math.min(1, (slope - 0.9) * 1.5));
      col[k3] = colr.r; col[k3 + 1] = colr.g; col[k3 + 2] = colr.b;
      uv[(k3 / 3) * 2] = i / SEG; uv[(k3 / 3) * 2 + 1] = 1 - j / SEG;        // photo row 0 = north edge
      if (good) { any = true; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
      k3 += 3; return k3 / 3 - 1;
    };
    for (let j = 0; j < V; j++) for (let i = 0; i < V; i++) put(j, i, 0);
    const [a0, b0, a1, b1] = this.area;
    const inArea = (i, j) => { const e = e0 + i * d, nn = n0 + S - j * d; return e >= a0 - 1e-6 && e <= a1 + 1e-6 && nn >= b0 - 1e-6 && nn <= b1 + 1e-6; };
    const idx = [];
    const ok = (a) => valid[a] === 1;
    for (let j = 0; j < SEG; j++) for (let i = 0; i < SEG; i++) {
      if (!inArea(i + 0.5, j + 0.5)) continue;
      const a = j * V + i, b = a + 1, c = a + V, e = c + 1;
      if (ok(a) && ok(b) && ok(c)) idx.push(a, c, b);
      if (ok(b) && ok(c) && ok(e)) idx.push(b, c, e);
    }
    // skirts hide cracks between neighbouring patches of different detail
    const drop = Math.max(2, d * 3);
    const edge = (cells) => {
      for (let q = 0; q < cells.length - 1; q++) {
        const [ja, ia] = cells[q], [jb, ib] = cells[q + 1];
        if (!inArea((ia + ib) / 2, (ja + jb) / 2)) continue;
        const ta = ja * V + ia, tb = jb * V + ib; if (!ok(ta) || !ok(tb)) continue;
        const ba = put(ja, ia, drop), bb = put(jb, ib, drop);
        idx.push(ta, ba, tb, tb, ba, bb, ta, tb, ba, tb, bb, ba);
      }
    };
    const R = [...Array(V).keys()];
    if (this.skirts === true) { edge(R.map((i) => [0, i])); edge(R.map((i) => [SEG, i])); edge(R.map((j) => [j, 0])); edge(R.map((j) => [j, SEG])); }
    if (!any || !idx.length) return 'empty';
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, k3), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor.subarray(0, k3), 3));
    g.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, k3), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv.subarray(0, (k3 / 3) * 2), 2));
    g.setIndex(idx); g.computeBoundingSphere();
    const mesh = new THREE.Mesh(g, this.mat);
    mesh.position.set(ce - this.cE, 0, -(cn - this.cN)); mesh.matrixAutoUpdate = false; mesh.updateMatrix();
    mesh.userData = { zmin, zmax, S, e0, n0, photo: null, photoState: 'none', k, edgeKey: '' };
    // keep the original edge heights so edges can be snapped to coarser neighbours and back
    const ys = g.attributes.position.array, orig = new Float32Array(4 * V);
    for (let q = 0; q < V; q++) { orig[q] = ys[q * 3 + 1]; orig[V + q] = ys[(SEG * V + q) * 3 + 1]; orig[2 * V + q] = ys[(q * V) * 3 + 1]; orig[3 * V + q] = ys[(q * V + SEG) * 3 + 1]; }
    mesh.userData.origEdges = orig;
    return mesh;
  }

  // height of the level-k data at a node position (exact; levels are decimations of the 1 m data)
  _nodeHeight(k, E, N) {
    const T = 1000 * 2 ** k, sp = 2 ** k;
    let tx = Math.floor(E / T); const ty = Math.floor(N / T);
    let t = this.tiles.get(`${k}/${tx}_${ty}`);
    let c = Math.round((E - tx * T) / sp), r = Math.round(((ty + 1) * T - N) / sp);
    if (t?.state !== 'ready') {            // node on a tile border: the neighbouring tile shares it
      const t2 = this.tiles.get(`${k}/${tx - (c === 0 ? 1 : 0)}_${ty + (r === 0 ? 1 : 0)}`);
      if (t2?.state !== 'ready') return null;
      if (c === 0) tx -= 1;
      t = t2; if (c === 0) c = t.n - 1; if (r === 0) r = t.n - 1;
    }
    const i = Math.min(t.n - 1, r) * t.n + Math.min(t.n - 1, c);
    let v;
    if (this.years === 0) v = t.h[i];
    else if (t.modelKey === `${this.years}`) v = t.modelH[i];
    else {                                   // neighbour tile not modelled yet: model this one node
      const key = (L) => `${L}:${k}/${tx}_${ty}`;
      const tn = this.tiles.get(key('tiles_nat')), te = this.tiles.get(key('tiles_env'));
      v = this._model(t.h[i], tn?.state === 'ready' ? tn.h[i] : NaN, te?.state === 'ready' ? te.h[i] : NaN);
    }
    return Number.isFinite(v) ? v : null;
  }
  // move one patch's edge vertices onto the straight lines of coarser neighbours (crack-free joins)
  _snapEdges(mesh, nb) {
    const u = mesh.userData, key = nb.join(',');
    if (u.edgeKey === key) return;
    u.edgeKey = key;
    const V = SEG + 1, d = u.S / SEG, ys = mesh.geometry.attributes.position.array, ex = this.exag;
    // edges: 0 = north (j=0), 1 = south (j=SEG), 2 = west (i=0), 3 = east (i=SEG)
    for (let s = 0; s < 4; s++) {
      for (let q = 0; q < V; q++) {
        const vi = s === 0 ? q : s === 1 ? SEG * V + q : s === 2 ? q * V : q * V + SEG;
        let y = u.origEdges[s * V + q];
        const D = nb[s];                                   // neighbour vertex spacing (0 = same or finer)
        if (D > d * 1.5) {
          const along = s < 2 ? u.e0 + q * d : u.n0 + u.S - q * d;              // E for N/S edges, N for W/E edges
          const fixed = s === 0 ? u.n0 + u.S : s === 1 ? u.n0 : s === 2 ? u.e0 : u.e0 + u.S;
          const a0 = Math.floor(along / D + 1e-9) * D, f = (along - a0) / D;
          if (f > 1e-6) {
            const k = Math.max(0, Math.min(5, Math.round(Math.log2(D))));
            const kk = Math.min(u.k, k);
            const h0 = s < 2 ? this._nodeHeight(kk, a0, fixed) : this._nodeHeight(kk, fixed, a0);
            const h1 = s < 2 ? this._nodeHeight(kk, a0 + D, fixed) : this._nodeHeight(kk, fixed, a0 + D);
            if (h0 !== null && h1 !== null) y = (h0 * (1 - f) + h1 * f) * ex;
          }
        }
        ys[vi * 3 + 1] = y;
      }
    }
    mesh.geometry.attributes.position.needsUpdate = true;
  }
  _patch(e0, n0, S) {
    const key = `${S}/${e0}/${n0}`;
    let p = this.patches.get(key);
    if (!p) { p = { mesh: null, state: 'new', used: this.frame }; this.patches.set(key, p); }
    p.used = this.frame;
    if (!p.mesh && p.state !== 'missing' && p.state !== 'empty' && this.builtThisFrame < 6) {
      const r = this._buildPatch(e0, n0, S);
      if (r instanceof THREE.Mesh) { p.mesh = r; p.state = 'ready'; this.builtThisFrame++; }
      else if (r) p.state = r;
    }
    return p;
  }

  // ---------------------------------------------------------------- per-frame LOD selection
  update(camera) {
    this.frame++; this.builtThisFrame = 0;
    const cam = this.toEN(camera.position);
    const show = new Set();
    const [a0, b0, a1, b1] = this.area;
    let tris = 0;
    const visit = (e0, n0, S) => {
      if (e0 > a1 || e0 + S < a0 || n0 > b1 || n0 + S < b0) return true;            // outside the area
      const p = this._patch(e0, n0, S);
      if (p.state === 'missing' || p.state === 'empty') return true;
      const zc = p.mesh ? (p.mesh.userData.zmin + p.mesh.userData.zmax) / 2 : cam.z - 50;
      const dx = Math.max(e0 - cam.E, 0, cam.E - (e0 + S)), dy = Math.max(n0 - cam.N, 0, cam.N - (n0 + S));
      const dist = Math.hypot(dx, dy, (cam.z - zc));
      if (S > MIN_PATCH && dist < S * this.lodFactor) {
        const h = S / 2;
        const kids = [[e0, n0 + h], [e0 + h, n0 + h], [e0, n0], [e0 + h, n0]];
        const ready = kids.map(([e, n]) => { const c = this._patch(e, n, h); return c.mesh || c.state === 'missing' || c.state === 'empty'; });
        if (ready.every(Boolean) || !p.mesh) { let all = true; for (const [e, n] of kids) all = visit(e, n, h) && all; if (all || !p.mesh) return all; }
      }
      if (p.mesh) { show.add(p.mesh); tris += p.mesh.geometry.index.count / 3; return true; }
      return false;
    };
    for (let e = Math.floor(a0 / ROOT) * ROOT; e < a1; e += ROOT)
      for (let n = Math.floor(b0 / ROOT) * ROOT; n < b1; n += ROOT) visit(e, n, ROOT);
    // attach visible, detach the rest
    for (const m of [...this.group.children]) if (!show.has(m)) this.group.remove(m);
    for (const m of show) if (m.parent !== this.group) this.group.add(m);
    // crack-free joins: find each visible patch's neighbour size on every side and snap edges to coarser ones
    const bySize = new Map();
    for (const m of show) { const u = m.userData; (bySize.get(u.S) || bySize.set(u.S, new Set()).get(u.S)).add(`${u.e0}/${u.n0}`); }
    const sizes = [...bySize.keys()].sort((a, b) => a - b);
    const sizeAt = (E, N) => { for (const S of sizes) if (bySize.get(S).has(`${Math.floor(E / S) * S}/${Math.floor(N / S) * S}`)) return S; return 0; };
    for (const m of show) {
      const u = m.userData, S = u.S, eps = 0.5;
      const cx = u.e0 + S / 2, cy = u.n0 + S / 2;
      const nb = [sizeAt(cx, u.n0 + S + eps), sizeAt(cx, u.n0 - eps), sizeAt(u.e0 - eps, cy), sizeAt(u.e0 + S + eps, cy)]
        .map((Sn) => (Sn > S ? Sn / SEG : 0));
      this._snapEdges(m, nb);
    }
    // satellite photos: request the nearest visible patches first (max 8 downloads at a time)
    const photoOn = this.uPhoto.value > 0.001 && this.tm.photoYear !== null;
    if (photoOn) {
      const want = [...show].filter((m) => m.userData.photoState === 'none')
        .map((m) => [m, m.position.distanceToSquared(camera.position)]).sort((a, b) => a[1] - b[1]);
      for (const [m] of want) { if (this.orthoInflight >= 8) break; this._loadPhoto(m); }
    }
    for (const m of show) {
      const u = m.userData, wantMat = photoOn && u.photo ? u.photo : this.mat;
      if (m.material !== wantMat) m.material = wantMat;
    }
    // evict old patches / tiles
    if (this.frame % 60 === 0) {
      for (const [k, p] of this.patches) if (this.frame - p.used > 300) {
        p.mesh?.geometry.dispose(); p.mesh?.userData.photo?.map?.dispose(); p.mesh?.userData.photo?.dispose(); this.patches.delete(k);
      }
      const ready = [...this.tiles.entries()].filter(([, t]) => t.state === 'ready').sort((a, b) => a[1].used - b[1].used);
      for (const [k] of ready.slice(0, Math.max(0, ready.length - 80))) this.tiles.delete(k);
    }
    this.stats = { patches: show.size, tris, tilesLoading: this.inflight, tilesCached: this.tiles.size };
  }

  // satellite sources: GURS orthophoto (primary, also historical years) with Esri World Imagery as backup.
  // Both serve images directly in D96/TM (EPSG:3794), so they line up with the terrain exactly.
  _photoUrls(u) {
    const year = this.tm?.photoYear || '';
    const gurs = this.orthoUrl(u.e0, u.n0, u.S, year);
    if (year) return [gurs];                                   // historical photos exist only at GURS
    const esri = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=${u.e0},${u.n0},${u.e0 + u.S},${u.n0 + u.S}&bboxSR=3794&imageSR=3794&size=256,256&format=jpg&f=image`;
    const src = this.photoSource || 'auto';
    if (src === 'gurs') return [gurs];
    if (src === 'esri') return [esri];
    const gursBad = (this.photoStats.gursFail >= 3 && this.photoStats.gursOk === 0);
    return gursBad ? [esri, gurs] : [gurs, esri];
  }
  _loadPhoto(m) {
    const u = m.userData; u.photoState = 'loading'; this.orthoInflight++;
    const urls = this._photoUrls(u); let k = 0, done = false;
    const tryNext = () => {
      if (k >= urls.length) { this.orthoInflight--; u.photoState = 'failed'; this.photoStats.failed++; return; }
      const url = urls[k++], isGurs = !url.includes('arcgisonline');
      let settled = false;
      const timer = setTimeout(() => { if (!settled) { settled = true; if (isGurs) this.photoStats.gursFail++; tryNext(); } }, 8000);
      this.texLoader.load(url, (tex) => {
        if (settled || done) { tex.dispose(); return; }
        settled = done = true; clearTimeout(timer); this.orthoInflight--;
        tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
        u.photo = new THREE.MeshLambertMaterial({ map: tex, vertexColors: true });
        u.photo.onBeforeCompile = this._photoShader;
        u.photo.customProgramCacheKey = () => 'terrain-photo';
        this._applyAlpha(u.photo);
        u.photoState = 'ready'; u.photoFrom = isGurs ? 'gurs' : 'esri';
        if (isGurs) this.photoStats.gursOk++; else this.photoStats.esriOk++;
      }, undefined, () => {
        if (settled) return; settled = true; clearTimeout(timer);
        if (isGurs) this.photoStats.gursFail++;
        tryNext();
      });
    };
    tryNext();
  }
  setPhotoSource(src) {                                         // 'auto' | 'gurs' | 'esri' — reload photos
    this.photoSource = src;
    for (const p of this.patches.values()) {
      const u = p.mesh?.userData; if (!u) continue;
      if (u.photo) { u.photo.map?.dispose(); u.photo.dispose(); u.photo = null; }
      if (u.photoState !== 'loading') u.photoState = 'none';
    }
  }
  // whole-terrain opacity (to look underground)
  setOpacity(a) {
    this.alpha = a;
    const mats = [this.mat, ...[...this.patches.values()].map((p) => p.mesh?.userData.photo).filter(Boolean)];
    for (const m of mats) this._applyAlpha(m);
  }
  _applyAlpha(m) {
    const a = this.alpha ?? 1;
    m.opacity = a; m.transparent = a < 0.999; m.depthWrite = a > 0.6; m.needsUpdate = true;
  }
  // relief: true/false · satellite: opacity 0..1 (0 = off)
  setLayers({ relief, satellite }) {
    if (relief !== undefined) this.uRelief.value = relief ? 1 : 0;
    if (satellite !== undefined) this.uPhoto.value = Math.max(0, Math.min(1, satellite));
  }

  // height at a map position from the finest tile already loaded (for the camera, placing objects …)
  heightAt(E, N) {
    for (let k = 0; k <= 5; k++) {
      const T = 1000 * 2 ** k, t = this.tiles.get(`${k}/${Math.floor(E / T)}_${Math.floor(N / T)}`);
      if (t?.state !== 'ready') continue;
      const sp = 2 ** k, c = (E - Math.floor(E / T) * T) / sp, r = ((Math.floor(N / T) + 1) * T - N) / sp;
      if (this.years > 0 && t.modelKey !== `${this.years}`) continue;        // not modelled for this time yet
      const h = this.years > 0 ? t.modelH : t.h;
      const c0 = Math.min(t.n - 2, Math.floor(c)), r0 = Math.min(t.n - 2, Math.floor(r)), fc = c - c0, fr = r - r0, n = t.n;
      const z = h[r0 * n + c0] * (1 - fc) * (1 - fr) + h[r0 * n + c0 + 1] * fc * (1 - fr) + h[(r0 + 1) * n + c0] * (1 - fc) * fr + h[(r0 + 1) * n + c0 + 1] * fc * fr;
      if (Number.isFinite(z)) return z;
    }
    return null;
  }
}
