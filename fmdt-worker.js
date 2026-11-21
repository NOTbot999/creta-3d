// Decodes FMDT terrain tiles off the main thread.  See pipeline/fmdt.py for the format.
async function inflate(bytes) {
  const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}
self.onmessage = async (e) => {
  const { id, url } = e.data;
  try {
    const r = await fetch(new URL(url, self.location.href));
    if (!r.ok) { self.postMessage({ id, missing: true }); return; }
    const buf = await r.arrayBuffer(), dv = new DataView(buf);
    const n = dv.getUint16(6, true), base = dv.getInt32(16, true), flags = dv.getUint8(20), blen = dv.getUint32(24, true);
    const d = new Int16Array((await inflate(new Uint8Array(buf, 28, blen))).buffer);
    const h = new Float32Array(n * n);
    let min = Infinity, max = -Infinity, rowStart = 0;
    for (let r0 = 0; r0 < n; r0++) {
      const o = r0 * n;
      rowStart += d[o]; let v = rowStart; h[o] = (v + base) / 100;
      for (let c = 1; c < n; c++) { v += d[o + c]; h[o + c] = (v + base) / 100; }
    }
    if (flags & 1) {
      const m = await inflate(new Uint8Array(buf, 28 + blen));
      for (let i = 0; i < n * n; i++) if ((m[i >> 3] >> (7 - (i & 7))) & 1) h[i] = NaN;
    }
    for (let i = 0; i < n * n; i++) { const v = h[i]; if (v < min) min = v; if (v > max) max = v; }
    self.postMessage({ id, n, h, min, max }, [h.buffer]);
  } catch (err) { self.postMessage({ id, error: String(err) }); }
};
