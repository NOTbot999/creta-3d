// Streaming lidar terrain for Slovenia (FMDT tiles, D96/TM).  Quadtree level-of-detail:
// 125 m patches at full 1 m detail near the camera, up to 32 km patches at 32 m in the distance.
import * as THREE from 'three';

const SEG = 125;                 // segments per patch side (126 x 126 vertices)
const ROOT = 32000;              // root patch = one level-5 tile (32 km)
const MIN_PATCH = 125;           // smallest patch: 125 m at 1 m spacing

export class TerrainLOD {
  constructor(scene, { baseUrl = '/terrain', workerUrl = '/static/fmdt-worker.js', orthoUrl = null, basePhoto = null, basePhotoLevels = 0, centreE, centreN, size, exaggeration = 1 } = {}) {
    // pre-baked photo of the whole area shipped with the site (1 m/px at level 0, tiles of 1000*2^L m)
    this.basePhoto = basePhoto; this.basePhotoLevels = basePhotoLevels; this.baseCache = new Map(); this.baseInflight = 0;
    this.scene = scene; this.baseUrl = baseUrl; this.cE = centreE; this.cN = centreN;
    this.orthoUrl = orthoUrl || ((e0, n0, S) => `/ortho?e0=${e0}&n0=${n0}&s=${S}&px=256`);
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
    const base = (shader) => {
      shader.uniforms.uRelief = this.uRelief;
      shader.fragmentShader = 'uniform float uRelief;\n' + shader.fragmentShader
        .replace('#include <color_fragment>', 'diffuseColor.rgb = mix(vec3(0.62, 0.62, 0.6), vColor.rgb, uRelief);');
    };
    this.mat = new THREE.MeshLambertMaterial({ vertexColors: true });
    this.mat.onBeforeCompile = base;
    this._photoShader = (shader) => {
      shader.uniforms.uRelief = this.uRelief; shader.uniforms.uPhoto = this.uPhoto;
      shader.fragmentShader = 'uniform float uRelief;\nuniform float uPhoto;\n' + shader.fragmentShader
        .replace('#include <map_fragment>', '')
        .replace('#include <color_fragment>',
          'vec3 baseCol = mix(vec3(0.62, 0.62, 0.6), vColor.rgb, uRelief);\n' +
          'vec4 photo = texture2D(map, vMapUv);\n' +
          'diffuseColor.rgb = mix(baseCol, photo.rgb, uPhoto);');
    };
    this.lodFactor = 2.2; this.frame = 0; this.stats = {};
    this.orthoInflight = 0; this.texLoader = new THREE.TextureLoader();
    this.photoStats = { gursOk: 0, gursFail: 0, esriOk: 0, failed: 0 };
    this.photoCache = new Map();             // photo key -> {tex, from, used}; survives patch rebuilds and revisits
    this._frustum = new THREE.Frustum(); this._m4 = new THREE.Matrix4(); this._sphere = new THREE.Sphere();
  }

  // ---------------------------------------------------------------- coordinates
  toLocal(E, N, z) { return new THREE.Vector3(E - this.cE, z * this.exag, -(N - this.cN)); }
  toEN(v) { return { E: v.x + this.cE, N: -v.z + this.cN, z: v.y / this.exag }; }

  // ---------------------------------------------------------------- tiles
  _tile(k, x, y) {
    const key = `${k}/${x}_${y}`;
    let t = this.tiles.get(key);
    if (!t) {
      t = { state: 'loading', used: this.frame };
      this.tiles.set(key, t);
      if (this.index && !this.index.levels[k]?.has(`${x}_${y}`)) { t.state = 'missing'; return t; }
      const id = this.nextId++; this.pending.set(id, key); this.inflight++;
      this.worker.postMessage({ id, url: `${this.baseUrl}/tiles/${k}/${x}_${y}.fmdt` });
    }
    t.used = this.frame; return t;
  }
  _onTile(m) {
    const key = this.pending.get(m.id); this.pending.delete(m.id); this.inflight--;
    const t = this.tiles.get(key); if (!t) return;
    if (m.missing || m.error) { t.state = 'missing'; return; }
    Object.assign(t, { state: 'ready', h: m.h, n: m.n, min: m.min, max: m.max });
  }
  async init() {
    try {
      const j = await (await fetch(`${this.baseUrl}/index.json`)).json();
      this.index = { levels: Object.fromEntries(Object.entries(j.levels).map(([k, v]) => [k, new Set(v)])) };
    } catch { this.index = null; }
  }

  // ---------------------------------------------------------------- patches
  _src(S) { return Math.max(0, Math.min(5, Math.floor(Math.log2(S / SEG)))); }
  _buildPatch(e0, n0, S) {
    const k = this._src(S), T = 1000 * 2 ** k, sp = 2 ** k;
    const tx = Math.floor(e0 / T), ty = Math.floor(n0 / T);
    const t = this._tile(k, tx, ty);
    if (t.state !== 'ready') return t.state === 'missing' ? 'missing' : null;
    const h = t.h;
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
    const v = t.h[i];
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
    // satellite photos: patches on screen first, nearest first; until a patch has its own photo it shows the
    // matching part of its parent's photo (so moving closer never leaves holes)
    const photoOn = this.uPhoto.value > 0.001;
    if (photoOn) {
      this._frustum.setFromProjectionMatrix(this._m4.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
      const want = [];
      for (const m of show) {
        const u = m.userData;
        if (u.photoState === 'none') {
          const cached = this.photoCache.get(this._photoKey(u));
          if (cached) { this._setPhoto(u, cached.tex, cached.from); continue; }
          if (u.S >= 1000 && this.basePhoto && this.photoSource !== 'esri') {
            const b = this._baseTex(u, true);                    // big patches: the shipped 1 m photo is sharp enough
            if (b) { const m = this._baseMaterial(u, b); u.photo = m.mat; u.photoTex = m.tex; u.photoState = 'ready'; u.photoFrom = 'base'; this._dropPlaceholder(u); continue; }
            else {
              const L = this._baseLevel(u.S), T = 1000 * 2 ** L;
              if (this.baseCache.get(`${L}/${Math.floor(u.e0 / T)}_${Math.floor(u.n0 / T)}`)?.state !== 'missing') continue;   // still loading
            }
          }
          this._sphere.copy(m.geometry.boundingSphere).applyMatrix4(m.matrixWorld);
          const onScreen = this._frustum.intersectsSphere(this._sphere);
          want.push([m, m.position.distanceToSquared(camera.position) * (onScreen ? 1 : 25)]);
        }
      }
      want.sort((a, b) => a[1] - b[1]);
      for (const [m] of want) { if (this.orthoInflight >= 12) break; this._loadPhoto(m); }
    }
    for (const m of show) {
      const u = m.userData;
      let wantMat = this.mat;
      if (photoOn) wantMat = u.photo || this._placeholder(u) || this.mat;
      if (m.material !== wantMat) m.material = wantMat;
    }
    // evict old patches / tiles
    if (this.frame % 60 === 0) {
      for (const [k, p] of this.patches) if (this.frame - p.used > 300) {
        const u = p.mesh?.userData;
        p.mesh?.geometry.dispose(); this._freePhoto(u); this._dropPlaceholder(u); this.patches.delete(k);   // photo textures stay cached
      }
      if (this.photoCache.size > 500) {                        // keep the most recently used photos
        const old = [...this.photoCache.entries()].sort((a, b) => a[1].used - b[1].used).slice(0, this.photoCache.size - 500);
        for (const [key, c] of old) { c.tex.dispose(); this.photoCache.delete(key); }
      }
      const ready = [...this.tiles.entries()].filter(([, t]) => t.state === 'ready').sort((a, b) => a[1].used - b[1].used);
      for (const [k] of ready.slice(0, Math.max(0, ready.length - 80))) this.tiles.delete(k);
    }
    this.stats = { patches: show.size, tris, tilesLoading: this.inflight, tilesCached: this.tiles.size };
  }

  // satellite sources: GURS orthophoto (primary) with Esri World Imagery as backup.
  // Both serve images directly in D96/TM (EPSG:3794), so they line up with the terrain exactly.
  _photoUrls(u) {
    const gurs = this.orthoUrl(u.e0, u.n0, u.S);
    const esri = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=${u.e0},${u.n0},${u.e0 + u.S},${u.n0 + u.S}&bboxSR=3794&imageSR=3794&size=256,256&format=jpg&f=image`;
    const src = this.photoSource || 'auto';
    if (src === 'gurs') return [gurs];
    if (src === 'esri') return [esri];
    const gursBad = (this.photoStats.gursFail >= 3 && this.photoStats.gursOk === 0);
    return gursBad ? [esri, gurs] : [gurs, esri];
  }
  _photoKey(u) { return `${this.photoSource || 'auto'}|${u.S}/${u.e0}/${u.n0}`; }
  _setPhoto(u, tex, from) {
    const c = this.photoCache.get(this._photoKey(u)); if (c) c.used = this.frame;
    u.photo = new THREE.MeshLambertMaterial({ map: tex, vertexColors: true });
    u.photo.onBeforeCompile = this._photoShader;
    u.photo.customProgramCacheKey = () => 'terrain-photo';
    this._applyAlpha(u.photo);
    u.photoState = 'ready'; u.photoFrom = from;
    this._dropPlaceholder(u);
  }
  // pre-baked area photo covering patch u (null if not shipped / not loaded yet); starts loading if needed
  _baseEntry(L, E, N) {
    const T = 1000 * 2 ** L, x = Math.floor(E / T), y = Math.floor(N / T), key = `${L}/${x}_${y}`;
    let b = this.baseCache.get(key);
    if (!b) { b = { state: 'new', T, x, y, L }; this.baseCache.set(key, b); }
    if (b.state === 'new' && this.baseInflight < 12) {
      b.state = 'loading'; this.baseInflight++;
      this.texLoader.load(this.basePhoto(L, x, y), (tex) => {
        this.baseInflight--; tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
        b.tex = tex; b.state = 'ready';
      }, undefined, () => { this.baseInflight--; b.state = 'missing'; });
    }
    return b;
  }
  // finest shipped photo tile covering patch u that is ready (asks for the ideal one, uses a coarser one meanwhile)
  _baseTex(u, exact = false) {
    if (!this.basePhoto || this.photoSource === 'esri') return null;
    if (u.S > 1000 * 2 ** (this.basePhotoLevels - 1)) return null;   // patch bigger than the largest shipped tile
    const L0 = this._baseLevel(u.S);
    const want = this._baseEntry(L0, u.e0, u.n0);
    if (want.state === 'ready' || exact) return want.state === 'ready' ? want : null;
    for (let L = L0 + 1; L < this.basePhotoLevels; L++) { const b = this._baseEntry(L, u.e0, u.n0); if (b.state === 'ready') return b; }
    return null;
  }
  // load the coarse shipped photo tiles of the whole area up front (few, small) so nothing is ever plain
  preloadBase() {
    if (!this.basePhoto) return;
    const [a0, b0, a1, b1] = this.area;
    for (let L = this.basePhotoLevels - 1; L >= 1; L--) {
      const T = 1000 * 2 ** L;
      for (let E = Math.floor(a0 / T) * T; E < a1; E += T) for (let N = Math.floor(b0 / T) * T; N < b1; N += T) this._baseEntry(L, E, N);
    }
  }
  _baseMaterial(u, b) {
    const tex = b.tex.clone();                                  // shares the image: no new download or upload
    tex.repeat.set(u.S / b.T, u.S / b.T); tex.offset.set((u.e0 - b.x * b.T) / b.T, (u.n0 - b.y * b.T) / b.T); tex.needsUpdate = true;
    const mat = new THREE.MeshLambertMaterial({ map: tex, vertexColors: true });
    mat.onBeforeCompile = this._photoShader; mat.customProgramCacheKey = () => 'terrain-photo';
    this._applyAlpha(mat);
    return { mat, tex };
  }
  // stand-in until a patch's own photo arrives: the matching part of the nearest ancestor's photo
  _placeholder(u) {
    for (let Sa = u.S * 2; Sa <= ROOT; Sa *= 2) {
      const ea = Math.floor(u.e0 / Sa) * Sa, na = Math.floor(u.n0 / Sa) * Sa;
      const c = this.photoCache.get(`${this.photoSource || 'auto'}|${Sa}/${ea}/${na}`);
      if (!c) continue;
      if (u.ph && u.ph.Sa <= Sa) return u.ph.mat;               // already showing this or a sharper ancestor
      this._dropPlaceholder(u);
      c.used = this.frame;
      const tex = c.tex.clone();                               // shares the image: no new download or upload
      tex.repeat.set(u.S / Sa, u.S / Sa); tex.offset.set((u.e0 - ea) / Sa, (u.n0 - na) / Sa); tex.needsUpdate = true;
      const mat = new THREE.MeshLambertMaterial({ map: tex, vertexColors: true });
      mat.onBeforeCompile = this._photoShader; mat.customProgramCacheKey = () => 'terrain-photo';
      this._applyAlpha(mat);
      u.ph = { Sa, mat, tex };
      return mat;
    }
    if (u.ph && u.ph.baseL === undefined) return u.ph.mat;       // an ancestor's live photo: keep it
    const b = this._baseTex(u);                                  // otherwise the shipped area photo (finest ready)
    if (b && !(u.ph && u.ph.baseL <= b.L)) {
      this._dropPlaceholder(u);
      const m = this._baseMaterial(u, b); u.ph = { Sa: Infinity, baseL: b.L, mat: m.mat, tex: m.tex };
    }
    return u.ph?.mat || null;
  }
  _freePhoto(u) { if (!u) return; u.photo?.dispose(); u.photoTex?.dispose(); u.photo = null; u.photoTex = null; }
  _baseLevel(S) { return Math.min(this.basePhotoLevels - 1, Math.max(0, Math.ceil(Math.log2(S / 1000)))); }
  _dropPlaceholder(u) { if (u?.ph) { u.ph.mat.dispose(); u.ph.tex.dispose(); u.ph = null; } }
  _loadPhoto(m) {
    const u = m.userData; u.photoState = 'loading'; this.orthoInflight++;
    const key = this._photoKey(u);
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
        const from = isGurs ? 'gurs' : 'esri';
        this.photoCache.set(key, { tex, from, used: this.frame });
        if (key === this._photoKey(u)) this._setPhoto(u, tex, from); else u.photoState = 'none';   // source changed meanwhile
        if (isGurs) this.photoStats.gursOk++; else this.photoStats.esriOk++;
      }, undefined, () => {
        if (settled) return; settled = true; clearTimeout(timer);
        if (isGurs) this.photoStats.gursFail++;
        tryNext();
      });
    };
    tryNext();
  }
  // rebuild all patches (e.g. new height exaggeration); downloaded photos stay cached
  resetPatches() {
    for (const p of this.patches.values()) { p.mesh?.geometry.dispose(); this._freePhoto(p.mesh?.userData); this._dropPlaceholder(p.mesh?.userData); }
    this.patches.clear();
  }
  setPhotoSource(src) {                                         // 'auto' | 'gurs' | 'esri' — reload photos
    this.photoSource = src;
    for (const p of this.patches.values()) {
      const u = p.mesh?.userData; if (!u) continue;
      this._freePhoto(u);
      this._dropPlaceholder(u);
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
    m.opacity = a; m.transparent = a < 0.999; m.depthWrite = a > 0.6;
    m.side = this.twoSided ? THREE.DoubleSide : THREE.FrontSide; m.needsUpdate = true;
  }
  // draw the underside too (noclip: the ground is seen as a ceiling from below)
  setTwoSided(on) {
    this.twoSided = on;
    this.setOpacity(this.alpha ?? 1);
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
      const h = t.h;
      const c0 = Math.min(t.n - 2, Math.floor(c)), r0 = Math.min(t.n - 2, Math.floor(r)), fc = c - c0, fr = r - r0, n = t.n;
      const z = h[r0 * n + c0] * (1 - fc) * (1 - fr) + h[r0 * n + c0 + 1] * fc * (1 - fr) + h[(r0 + 1) * n + c0] * (1 - fc) * fr + h[(r0 + 1) * n + c0 + 1] * fc * fr;
      if (Number.isFinite(z)) return z;
    }
    return null;
  }
}
