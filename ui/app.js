/* Resolve UI — replay console for one event.
   Reads window.RESOLVE_GEO (base map) and window.RESOLVE_DATA (schema resolve-ui/1, see SCHEMA.md).
   No build step, no network calls: works offline from a local folder or a static server. */
(function () {
  'use strict';

  const D = window.RESOLVE_DATA;
  const G = window.RESOLVE_GEO;
  const $ = (s, el) => (el || document).querySelector(s);
  const NS = 'http://www.w3.org/2000/svg';

  if (!D || !G || D.schema !== 'resolve-ui/1') {
    document.body.innerHTML = '<p class="fatal">Resolve could not find its data. Run <code>python -m resolve export-ui</code>, then reload this page.</p>';
    return;
  }

  // ---------------------------------------------------------------- constants
  const meta = D.meta;
  const grid = D.grid;
  const NLAT = grid.nlat, NLON = grid.nlon, STEP = grid.step, LAT0 = grid.lat0, LON0 = grid.lon0;
  const M = meta.members;
  const THR = meta.threshold_mm;
  const runs = D.runs;
  const windows = D.windows.slice().sort((a, b) => a.cells - b.cells);
  const maxKm = windows[windows.length - 1].km;
  const kmOf = n => { const w = windows.find(v => v.cells === n); return w ? w.km : Math.round(n * STEP * 111.32); };
  const BINS = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90];
  const binOf = p => { let b = -1; for (let i = 0; i < BINS.length; i++) if (p >= BINS[i] - 1e-9) b = i; return b; };
  const land = D.land;
  const obsSet = new Set(D.observed.cells);
  const lockIdx = D.lockon_init ? runs.findIndex(r => r.init === D.lockon_init) : -1;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const DOMAIN = {
    latS: LAT0 - STEP / 2, latN: LAT0 + (NLAT - 1) * STEP + STEP / 2,
    lonW: LON0 - STEP / 2, lonE: LON0 + (NLON - 1) * STEP + STEP / 2,
  };
  const PHI0 = (DOMAIN.latS + DOMAIN.latN) / 2;
  const COS0 = Math.cos(PHI0 * Math.PI / 180);
  const VIEWS = {
    india:  { latS: 5.5, latN: 38.5, lonW: 66.0, lonE: 100.0 },
    domain: { latS: DOMAIN.latS - 1.3, latN: DOMAIN.latN + 1.3, lonW: DOMAIN.lonW - 1.5, lonE: DOMAIN.lonE + 1.5 },
    event: null,
  };
  (function eventBox() {
    // observed cells plus every cell any run gave at least a 15% chance; padded, at least 7 degrees tall
    let s = 90, n = -90, w = 360, e = -360;
    const take = idx => { const r = Math.floor(idx / NLON), c = idx % NLON; const la = LAT0 + r * STEP, lo = LON0 + c * STEP; s = Math.min(s, la); n = Math.max(n, la); w = Math.min(w, lo); e = Math.max(e, lo); };
    D.observed.cells.forEach(take);
    const k15 = Math.ceil(0.15 * M);
    runs.forEach(r => r.counts.forEach((v, i) => { if (v >= k15) take(i); }));
    if (s > n) { VIEWS.event = VIEWS.domain; return; }
    const pad = 1.4, minSpan = 7;
    let latS = s - pad, latN = n + pad, lonW = w - pad, lonE = e + pad;
    if (latN - latS < minSpan) { const c = (latN + latS) / 2; latS = c - minSpan / 2; latN = c + minSpan / 2; }
    if ((lonE - lonW) * COS0 < minSpan) { const c = (lonE + lonW) / 2; lonW = c - minSpan / 2 / COS0; lonE = c + minSpan / 2 / COS0; }
    VIEWS.event = { latS, latN, lonW, lonE };
  })();

  const PLACES = [
    { n: 'Hyderabad', lat: 17.385, lon: 78.487 },
    { n: 'Visakhapatnam', lat: 17.687, lon: 83.218 },
    { n: 'Kakinada', lat: 16.989, lon: 82.247 },
    { n: 'Vijayawada', lat: 16.506, lon: 80.648, end: true },
    { n: 'Chennai', lat: 13.083, lon: 80.270 },
    { n: 'Bengaluru', lat: 12.972, lon: 77.594 },
    { n: 'Bhubaneswar', lat: 20.296, lon: 85.825 },
    { n: 'Raipur', lat: 21.251, lon: 81.630 },
    { n: 'Kolkata', lat: 22.573, lon: 88.364 },
  ];
  const STATES = [
    { n: 'Andhra Pradesh', lat: 14.95, lon: 78.95 },
    { n: 'Telangana', lat: 18.55, lon: 78.75 },
    { n: 'Odisha', lat: 20.75, lon: 84.1 },
    { n: 'Chhattisgarh', lat: 22.55, lon: 82.4 },
    { n: 'Tamil Nadu', lat: 10.95, lon: 78.35 },
    { n: 'Karnataka', lat: 14.55, lon: 75.95 },
    { n: 'Maharashtra', lat: 19.65, lon: 75.9 },
    { n: 'Madhya Pradesh', lat: 23.45, lon: 78.2 },
  ];
  const WATER = [
    { n: 'Bay of Bengal', lat: 15.3, lon: 86.7 },
    { n: 'Arabian Sea', lat: 14.2, lon: 70.9 },
  ];

  // ---------------------------------------------------------------- helpers
  const mk = (tag, attrs, parent) => {
    const el = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  };
  const utc = iso => new Date(iso.endsWith('Z') ? iso : iso + 'Z');
  const dShort = iso => utc(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
  const dLong = iso => utc(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
  const pct = p => Math.round(p * 100) + '%';
  const f2 = v => (v == null || !isFinite(v)) ? '—' : v.toFixed(2);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const latTxt = v => Math.abs(v).toFixed(2).replace(/0$/, '') + '°' + (v >= 0 ? 'N' : 'S');
  const lonTxt = v => Math.abs(v).toFixed(2).replace(/0$/, '') + '°E';
  const leadTxt = L => L === 1 ? '1 day before the rain day' : L + ' days before the rain day';
  const fssAt = (run, n) => { const i = windows.findIndex(w => w.cells === n); return i < 0 ? null : run.fss[i]; };
  const tileValues = (run, n) => (n === 1 ? run.counts : run.nmep[String(n)]);

  // ---------------------------------------------------------------- state
  let defaultRun = runs.length - 1;
  (function () {
    let best = null;
    runs.forEach((r, i) => { if (r.earned_cells != null && (best == null || r.earned_cells <= runs[best].earned_cells)) best = i; });
    if (best != null) defaultRun = best;
  })();
  const state = { run: defaultRun, mode: 'earned', observed: true, playing: false, timer: null, view: 'chart', zoom: 'event', panZoom: null };

  // ---------------------------------------------------------------- header
  function renderHeader() {
    const target = meta.target_date + 'T00:00Z';
    const ctx = [
      ['Event', meta.event],
      ['Threshold', '≥ ' + THR + ' mm in 24 h'],
      ['Rain day', dLong(target) + ', 24 h to 08:30 IST'],
      ['Forecast', meta.forecast.replace(/^ECMWF IFS ENS, 0\.25°, /, 'ECMWF ensemble, ')],
    ];
    $('#context').innerHTML = ctx.map(([k, v]) => '<div><dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd></div>').join('');
    const live = meta.label === 'live';
    const pill = $('#mode-pill');
    pill.textContent = live ? 'Live' : 'Replay';
    pill.dataset.live = String(live);
    pill.title = live ? 'Latest 00 UTC run, processed on arrival.' : 'Archived runs, replayed using only data available at each issue time.';
    document.title = 'Resolve ' + meta.event.replace(/^Cyclone /, '') + (live ? ' Live' : ' Replay');
    if (meta.preview) {
      $('#preview-notice').hidden = false;
      $('#preview-text').innerHTML = '<strong>Preview data.</strong> ' + esc(meta.preview_note || '') +
        ' Run <code>python -m resolve export-ui</code> to load the real fields.';
    }
    $('#credit').textContent = 'Forecast: ' + meta.forecast_credit + '. Observed: IMD. Base map: Natural Earth.';
    $('#tl-sub').textContent = runs.length + ' runs at 00 UTC, ' + dShort(runs[0].init) + ' to ' + dShort(runs[runs.length - 1].init);
  }

  // ---------------------------------------------------------------- map
  const mapEl = $('#map');
  const svg = $('#map-svg');
  const layers = {};
  ['grat', 'land', 'states', 'borders', 'veil', 'data', 'obs', 'domain', 'track', 'tc', 'labels'].forEach(k => {
    layers[k] = mk('g', { class: 'g-' + k }, svg);
  });
  let P = null;

  function projection(W, H) {
    const V = state.panZoom || VIEWS[state.zoom] || VIEWS.domain;
    const latSpan = V.latN - V.latS;
    const lonSpan = V.lonE - V.lonW;
    const s = Math.min(H / latSpan, W / (lonSpan * COS0));
    const cx = (V.lonW + V.lonE) / 2, cy = (V.latS + V.latN) / 2;
    return {
      s, W, H,
      x: lon => W / 2 + (lon - cx) * COS0 * s,
      y: lat => H / 2 - (lat - cy) * s,
      lon: px => cx + (px - W / 2) / (COS0 * s),
      lat: py => cy - (py - H / 2) / s,
    };
  }
  const pt = ([lo, la]) => P.x(lo).toFixed(1) + ',' + P.y(la).toFixed(1);
  const ringPath = rings => rings.map(r => 'M' + r.map(pt).join('L') + 'Z').join('');
  const linePath = lines => lines.map(r => 'M' + r.map(pt).join('L')).join('');
  const clear = g => { while (g.firstChild) g.removeChild(g.firstChild); };

  function drawBase() {
    ['grat', 'land', 'states', 'borders', 'veil', 'domain', 'tc', 'labels'].forEach(k => clear(layers[k]));
    const { W, H } = P;
    const lonL = P.lon(0), lonR = P.lon(W), latT = P.lat(0), latB = P.lat(H);

    // graticule (sits under land, so it reads over the sea only)
    let g = '';
    for (let lo = Math.ceil(lonL / 2) * 2; lo <= lonR; lo += 2) g += 'M' + P.x(lo).toFixed(1) + ',0V' + H;
    for (let la = Math.ceil(latB / 2) * 2; la <= latT; la += 2) g += 'M0,' + P.y(la).toFixed(1) + 'H' + W;
    mk('path', { d: g }, layers.grat);

    mk('path', { d: ringPath(G.land) }, layers.land);
    mk('path', { d: linePath(G.states) }, layers.states);
    mk('path', { d: linePath(G.borders) }, layers.borders);

    // veil outside the analysis domain
    const x0 = P.x(DOMAIN.lonW), x1 = P.x(DOMAIN.lonE), y0 = P.y(DOMAIN.latN), y1 = P.y(DOMAIN.latS);
    mk('path', { d: 'M0,0H' + W + 'V' + H + 'H0Z' + 'M' + x0 + ',' + y0 + 'V' + y1 + 'H' + x1 + 'V' + y0 + 'Z' }, layers.veil);

    // domain frame with 2-degree ticks
    const dom = layers.domain;
    mk('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, dom);
    const leftIn = x0 > 44, bottomIn = y1 < H - 22;
    for (let la = Math.ceil(DOMAIN.latS / 2) * 2; la <= DOMAIN.latN; la += 2) {
      const y = P.y(la);
      if (y < 60 || y > H - 8) continue;
      if (leftIn) {
        mk('line', { x1: x0 - 5, x2: x0, y1: y, y2: y }, dom);
        const t = mk('text', { x: x0 - 8, y: y + 3.5, 'text-anchor': 'end' }, dom); t.textContent = la + '°N';
      } else {
        const t = mk('text', { x: 10, y: y + 3.5, 'text-anchor': 'start', class: 'inside' }, dom); t.textContent = la + '°N';
      }
    }
    for (let lo = Math.ceil(DOMAIN.lonW / 2) * 2; lo <= DOMAIN.lonE; lo += 2) {
      const x = P.x(lo);
      if (x < 44 || x > W - 20) continue;
      if (bottomIn) {
        mk('line', { x1: x, x2: x, y1: y1, y2: y1 + 5 }, dom);
        const t = mk('text', { x: x, y: y1 + 17, 'text-anchor': 'middle' }, dom); t.textContent = lo + '°E';
      } else {
        const t = mk('text', { x: x, y: H - 10, 'text-anchor': 'middle', class: 'inside' }, dom); t.textContent = lo + '°E';
      }
    }
    if (y0 > 20 && x1 < W - 4) {
      const lab = mk('text', { x: x1, y: y0 - 8, 'text-anchor': 'end' }, dom);
      lab.textContent = 'Analysis domain';
    }

    // IMD cyclone positions
    (D.track || []).forEach((tp, i, arr) => {
      const cx = P.x(tp.lon), cy = P.y(tp.lat);
      const gg = mk('g', { transform: 'translate(' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ')' }, layers.tc);
      const arms = 'M0,-4.6C4.5,-9 10,-8.5 12,-4.5M0,4.6C-4.5,9 -10,8.5 -12,4.5';
      mk('path', { d: arms, class: 'halo' }, gg); mk('circle', { r: 4.6, class: 'halo' }, gg);
      mk('path', { d: arms }, gg); mk('circle', { r: 4.6 }, gg);
      if (i === arr.length - 1 || arr.length <= 3) {
        const t = mk('text', { x: 16, y: 16 }, gg); t.textContent = 'IMD position, ' + tp.label;
      }
    });

    // labels
    const L = layers.labels;
    STATES.forEach(s => {
      if (s.lon < lonL || s.lon > lonR) return;
      const t = mk('text', { x: P.x(s.lon), y: P.y(s.lat), 'text-anchor': 'middle', class: 'state' }, L);
      t.textContent = s.n.toUpperCase();
    });
    WATER.forEach(s => {
      if (s.lon - 2 < lonL || s.lon + 2 > lonR) return;
      const t = mk('text', { x: P.x(s.lon), y: P.y(s.lat), 'text-anchor': 'middle', class: 'water' }, L);
      t.textContent = s.n;
    });
    PLACES.forEach(p => {
      const x = P.x(p.lon), y = P.y(p.lat);
      mk('circle', { cx: x, cy: y, r: 2.3, class: 'city-dot' }, L);
      const t = mk('text', { x: p.end ? x - 6 : x + 6, y: y + 4, 'text-anchor': p.end ? 'end' : 'start', class: 'city' }, L);
      t.textContent = p.n;
    });
  }

  function cellBox(r0, r1, c0, c1) {
    const latS = LAT0 + r0 * STEP - STEP / 2, latN = LAT0 + r1 * STEP + STEP / 2;
    const lonW = LON0 + c0 * STEP - STEP / 2, lonE = LON0 + c1 * STEP + STEP / 2;
    return { latS, latN, lonW, lonE, x: P.x(lonW), y: P.y(latN), w: P.x(lonE) - P.x(lonW), h: P.y(latS) - P.y(latN) };
  }

  function drawData(animate) {
    const run = runs[state.run];
    const holder = layers.data;
    const old = Array.from(holder.querySelectorAll('g.set:not(.leaving)'));
    const g = mk('g', { class: 'set' + (state.mode === 'raw' ? ' raw' : '') }, holder);
    const n = state.mode === 'raw' ? 1 : run.earned_cells;
    if (n != null) {
      const vals = tileValues(run, n);
      const nbr = Math.ceil(NLAT / n), nbc = Math.ceil(NLON / n);
      const gap = n > 1 ? 2 : 0;
      for (let i = 0; i < nbr; i++) {
        for (let j = 0; j < nbc; j++) {
          const v = vals[i * nbc + j];
          const b = binOf(v / M);
          if (b < 0) continue;
          const bx = cellBox(i * n, Math.min(NLAT, (i + 1) * n) - 1, j * n, Math.min(NLON, (j + 1) * n) - 1);
          const a = { x: (bx.x + gap / 2).toFixed(1), y: (bx.y + gap / 2).toFixed(1), width: Math.max(0, bx.w - gap).toFixed(1), height: Math.max(0, bx.h - gap).toFixed(1), class: 'f' + (b + 1) };
          if (gap) a.rx = 2;
          mk('rect', a, g);
        }
      }
    }
    const fade = animate && !reduceMotion.matches && old.length;
    if (fade) {
      g.classList.add('entering');
      requestAnimationFrame(() => requestAnimationFrame(() => g.classList.remove('entering')));
      old.forEach(o => { o.classList.add('leaving'); setTimeout(() => o.remove(), 380); });
    } else {
      old.forEach(o => o.remove());
    }
    drawObserved();
    drawTrack();
    const empty = state.mode === 'earned' && run.earned_cells == null;
    $('#map-empty').hidden = !empty;
    if (empty) {
      $('#empty-body').textContent = 'At ' + run.lead_days + (run.lead_days === 1 ? ' day' : ' days') +
        ' out, no tile size up to ' + maxKm + ' km beat the useful-skill line. Resolve draws nothing rather than a guess.';
    }
  }

  function drawObserved() {
    clear(layers.obs);
    if (!state.observed) return;
    let d = '';
    const has = (r, c) => r >= 0 && r < NLAT && c >= 0 && c < NLON && obsSet.has(r * NLON + c);
    obsSet.forEach(idx => {
      const r = Math.floor(idx / NLON), c = idx % NLON;
      const b = cellBox(r, r, c, c);
      const xl = b.x.toFixed(1), xr = (b.x + b.w).toFixed(1), yt = b.y.toFixed(1), yb = (b.y + b.h).toFixed(1);
      if (!has(r - 1, c)) d += 'M' + xl + ',' + yb + 'H' + xr;
      if (!has(r + 1, c)) d += 'M' + xl + ',' + yt + 'H' + xr;
      if (!has(r, c - 1)) d += 'M' + xl + ',' + yt + 'V' + yb;
      if (!has(r, c + 1)) d += 'M' + xr + ',' + yt + 'V' + yb;
    });
    mk('path', { d, class: 'halo', 'stroke-linecap': 'square' }, layers.obs);
    mk('path', { d, class: 'line', 'stroke-linecap': 'square' }, layers.obs);
  }

  function drawTrack() {
    clear(layers.track);
    const pts = runs.map((r, i) => (r.object ? { i, x: P.x(r.object.lon), y: P.y(r.object.lat) } : null)).filter(Boolean);
    if (pts.length > 1) mk('path', { d: 'M' + pts.map(p => p.x.toFixed(1) + ',' + p.y.toFixed(1)).join('L'), class: 'path' }, layers.track);
    pts.forEach(p => { if (p.i !== state.run) mk('circle', { cx: p.x, cy: p.y, r: 3 }, layers.track); });
    const cur = pts.find(p => p.i === state.run);
    if (cur) {
      mk('circle', { cx: cur.x, cy: cur.y, r: 5, class: 'now' }, layers.track);
      const t = mk('text', { x: cur.x - 9, y: cur.y - 9, 'text-anchor': 'end' }, layers.track);
      t.textContent = dShort(runs[state.run].init) + ' run';
    }
  }

  function drawScalebar() {
    const sb = $('#scalebar');
    while (sb.firstChild) sb.removeChild(sb.firstChild);
    const kmPerPx = 111.32 / P.s;
    const len = [50, 100, 200, 300, 500].find(k => k / kmPerPx >= 90) || 500;
    const w = len / kmPerPx;
    const x0 = 188 - w, y = 6;
    mk('rect', { x: x0, y: y, width: w / 2, height: 5, style: 'fill: var(--surface)', 'stroke-width': 1 }, sb);
    mk('rect', { x: x0 + w / 2, y: y, width: w / 2, height: 5, style: 'fill: var(--ink-2)', 'stroke-width': 1 }, sb);
    let t = mk('text', { x: x0, y: 23, 'text-anchor': 'start' }, sb); t.textContent = '0';
    t = mk('text', { x: x0 + w, y: 23, 'text-anchor': 'end' }, sb); t.textContent = len + ' km';
  }

  function layout() {
    const W = mapEl.clientWidth, H = mapEl.clientHeight;
    if (!W || !H) return;
    P = projection(W, H);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    drawBase();
    drawData(false);
    drawScalebar();
  }

  // ---------------------------------------------------------------- legend
  function renderLegend() {
    const title = state.mode === 'raw'
      ? 'Chance of ≥ ' + THR + ' mm in each 0.25° cell'
      : 'Chance of ≥ ' + THR + ' mm somewhere in each tile';
    const ramp = BINS.map((_, i) => '<span class="f' + (i + 1) + '" style="background: var(--p' + (i + 1) + ')"></span>').join('');
    const ticks = BINS.map(b => '<span>' + Math.round(b * 100) + (b === BINS[0] ? '%' : '') + '</span>').join('');
    const keys = [
      ['<svg width="22" height="12"><rect x="1.5" y="1.5" width="19" height="9" fill="none" stroke="var(--halo)" stroke-width="4"/><rect x="1.5" y="1.5" width="19" height="9" fill="none" stroke="var(--ink)" stroke-width="1.8"/></svg>', 'Observed ≥ ' + THR + ' mm (IMD)'],
      ['<svg width="22" height="12"><path d="M1 9L11 3L21 6" fill="none" stroke="var(--ink-3)" stroke-width="1.2"/><circle cx="11" cy="3" r="2.6" fill="var(--ink-3)"/><circle cx="21" cy="6" r="3.4" fill="var(--ink)"/></svg>', 'Footprint centre in each run'],
      ['<svg width="22" height="14" viewBox="-11 -7 22 14"><path d="M0,-3.4C3.3,-6.6 7.3,-6.2 8.8,-3.3M0,3.4C-3.3,6.6 -7.3,6.2 -8.8,3.3" fill="none" stroke="var(--ink)" stroke-width="1.5" stroke-linecap="round"/><circle r="3.4" fill="none" stroke="var(--ink)" stroke-width="1.5"/></svg>', 'Cyclone position from IMD'],
    ];
    $('#legend').innerHTML =
      '<div class="legend-title">' + esc(title) + '</div>' +
      '<div class="legend-ramp" aria-hidden="true">' + ramp + '</div>' +
      '<div class="legend-ticks" aria-hidden="true">' + ticks + '</div>' +
      '<div class="legend-keys">' + keys.map(([s, t]) => '<div class="legend-key">' + s + '<span>' + esc(t) + '</span></div>').join('') + '</div>';
  }

  // ---------------------------------------------------------------- tooltip
  const tip = $('#tip');
  function showTip(html, px, py) {
    tip.innerHTML = html;
    tip.hidden = false;
    const W = mapEl.clientWidth, H = mapEl.clientHeight;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = px + 14, y = py + 14;
    if (x + tw > W - 8) x = px - tw - 14;
    if (y + th > H - 8) y = py - th - 14;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function hideTip() { tip.hidden = true; }

  let drag = null;
  svg.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    const V = state.panZoom || { ...(VIEWS[state.zoom] || VIEWS.domain) };
    drag = { sx: e.clientX, sy: e.clientY, V, moved: false };
    svg.setPointerCapture(e.pointerId);
  });
  const endDrag = () => { drag = null; svg.classList.remove('dragging'); };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);
  svg.addEventListener('wheel', e => {
    e.preventDefault();
    if (!P) return;
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const curLon = P.lon(px), curLat = P.lat(py);
    const factor = e.deltaY > 0 ? 1.18 : 1 / 1.18;
    const V = state.panZoom || { ...(VIEWS[state.zoom] || VIEWS.domain) };
    state.panZoom = {
      latS: curLat + (V.latS - curLat) * factor,
      latN: curLat + (V.latN - curLat) * factor,
      lonW: curLon + (V.lonW - curLon) * factor,
      lonE: curLon + (V.lonE - curLon) * factor,
    };
    state.zoom = 'custom';
    ['event', 'domain', 'india'].forEach(n => $('#zoom-' + n).setAttribute('aria-pressed', 'false'));
    layout();
  }, { passive: false });

  svg.addEventListener('pointermove', e => {
    if (!P) return;
    if (drag) {
      const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        if (!drag.moved) {
          drag.moved = true;
          svg.classList.add('dragging');
          ['event', 'domain', 'india'].forEach(n => $('#zoom-' + n).setAttribute('aria-pressed', 'false'));
        }
        state.zoom = 'custom';
        state.panZoom = {
          latS: drag.V.latS + dy / P.s,
          latN: drag.V.latN + dy / P.s,
          lonW: drag.V.lonW - dx / P.s / COS0,
          lonE: drag.V.lonE - dx / P.s / COS0,
        };
        hideTip();
        layout();
        return;
      }
    }
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const lo = P.lon(px), la = P.lat(py);
    const c = Math.round((lo - LON0) / STEP), r = Math.round((la - LAT0) / STEP);
    if (r < 0 || r >= NLAT || c < 0 || c >= NLON) return hideTip();
    const run = runs[state.run];
    const n = state.mode === 'raw' ? 1 : run.earned_cells;
    let html = '';
    if (n == null) {
      html = '<b>No earned footprint</b><br><span class="muted">This run has no skilful tile size.</span>';
    } else {
      const i = Math.floor(r / n), j = Math.floor(c / n), nbc = Math.ceil(NLON / n);
      const v = tileValues(run, n)[i * nbc + j];
      const r0 = i * n, r1 = Math.min(NLAT, (i + 1) * n) - 1, c0 = j * n, c1 = Math.min(NLON, (j + 1) * n) - 1;
      const bx = cellBox(r0, r1, c0, c1);
      let obsN = 0, landN = 0;
      for (let rr = r0; rr <= r1; rr++) for (let cc = c0; cc <= c1; cc++) {
        const k = rr * NLON + cc; if (land[k]) landN++; if (obsSet.has(k)) obsN++;
      }
      if (n === 1) {
        html = '<b>' + pct(v / M) + '</b> chance of ≥ ' + THR + ' mm in this cell' +
          '<br><span class="muted">' + v + ' of ' + M + ' members. ' + latTxt(LAT0 + r * STEP) + ', ' + lonTxt(LON0 + c * STEP) + '</span>';
        if (land[r * NLON + c]) html += '<br>Observed: ' + (obsN ? 'reached ' + THR + ' mm' : 'below ' + THR + ' mm');
      } else {
        html = '<b>' + pct(v / M) + '</b> chance of ≥ ' + THR + ' mm somewhere in this ' + kmOf(n) + ' km tile' +
          '<br><span class="muted">' + v + ' of ' + M + ' members. ' + latTxt(bx.latS) + ' to ' + latTxt(bx.latN) + '</span>';
        if (landN) html += '<br>Observed: ' + (obsN ? obsN + (obsN === 1 ? ' cell' : ' cells') + ' reached ' + THR + ' mm' : 'no cell reached ' + THR + ' mm');
      }
    }
    showTip(html, px, py);
  });
  svg.addEventListener('pointerleave', hideTip);

  // ---------------------------------------------------------------- timeline
  const runsEl = $('#runs');
  runsEl.style.setProperty('--n', runs.length);
  const runBtns = runs.map((r, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'run';
    const p = r.peak_count / M;
    const h = Math.max(1, Math.round(p * 70));
    const km = r.earned_cells != null ? kmOf(r.earned_cells) : null;
    const sq = km != null ? Math.max(3, Math.round(16 * km / maxKm)) : 0;
    b.innerHTML =
      '<span class="run-date">' + esc(dShort(r.init)) + '<small>D−' + r.lead_days + '</small></span>' +
      '<span class="run-bar"><span class="track"></span><span class="fill" style="height:' + h + 'px"></span>' +
      '<span class="val" style="bottom:' + (h + 3) + 'px">' + pct(p) + '</span></span>' +
      (km != null
        ? '<span class="run-scale"><span class="sq" style="width:' + sq + 'px;height:' + sq + 'px"></span><span>' + km + ' km</span></span>'
        : '<span class="run-scale none"><span>no skill</span></span>');
    b.setAttribute('aria-label', dLong(r.init) + ' run, ' + r.lead_days + ' days out. Peak chance ' + pct(p) +
      '. ' + (km != null ? 'Earned scale ' + km + ' km.' : 'No earned scale.') + (r.locked ? ' Locked on.' : ''));
    b.addEventListener('click', () => { stop(); select(i, true); });
    runsEl.appendChild(b);
    return b;
  });
  if (lockIdx >= 0) {
    const band = document.createElement('div');
    band.className = 'lock-band';
    band.style.gridColumn = (lockIdx + 1) + ' / -1';
    band.textContent = 'Locked on from the ' + dShort(runs[lockIdx].init) + ' run';
    runsEl.appendChild(band);
  }

  // ---------------------------------------------------------------- rail
  function renderRail() {
    const run = runs[state.run];
    const i = state.run;
    $('#run-title').textContent = 'Run issued ' + dLong(run.init) + ', 00 UTC';
    $('#run-sub').textContent = leadTxt(run.lead_days) + (run.window_offset_hours ? '. Rain summed to 00 UTC, ' + Math.abs(run.window_offset_hours) + ' h early, because forecast steps are 6-hourly beyond 144 h.' : '');

    const hv = $('#hero-value');
    if (run.earned_cells != null) {
      const f = fssAt(run, run.earned_cells);
      hv.className = 'hero-value';
      hv.innerHTML = kmOf(run.earned_cells) + '<span class="unit">km</span>';
      $('#hero-note').textContent = 'Smallest tile where this run beat the useful-skill line: FSS ' + f2(f) + ' against ' + f2(run.useful) + '.';
    } else {
      hv.className = 'hero-value none';
      hv.textContent = 'None yet';
      $('#hero-note').textContent = 'No tile size up to ' + maxKm + ' km beat the useful-skill line (' + f2(run.useful) + ').';
    }

    $('#ladder').innerHTML = windows.map((w, k) => {
      const s = Math.max(4, Math.round(46 * w.km / maxKm));
      const pass = run.fss[k] != null && run.fss[k] >= run.useful;
      const cls = 'rung' + (pass ? ' pass' : '') + (w.cells === run.earned_cells ? ' earned' : '');
      return '<div class="' + cls + '" title="FSS ' + f2(run.fss[k]) + '"><div class="box" style="width:' + s + 'px;height:' + s + 'px"></div><span>' + w.km + ' km</span></div>';
    }).join('');

    const p = run.peak_count / M;
    const stat = (label, value, sub, na) =>
      '<div class="stat"><span class="stat-label">' + label + '</span><span class="stat-value' + (na ? ' na' : '') + '">' + value + '</span><span class="stat-sub">' + sub + '</span></div>';
    const firstRun = i === 0;
    $('#stats').innerHTML =
      stat('Peak chance', pct(p), run.peak_count + ' of ' + M + ' members in one cell') +
      (run.drift_km != null
        ? stat('Footprint shift', Math.round(run.drift_km) + ' km', 'since the previous run')
        : stat('Footprint shift', '—', firstRun ? 'first run in the replay' : 'not computed for this run', true)) +
      (run.iou != null
        ? stat('Overlap', Math.round(run.iou * 100) + '%', 'with the previous run')
        : stat('Overlap', '—', firstRun ? 'first run in the replay' : 'not computed for this run', true));

    const lock = $('#lock');
    if (run.locked) {
      lock.className = 'lock on';
      lock.innerHTML = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 7.4l2.6 2.6L11 4.4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>Locked on since the ' + esc(dShort(D.lockon_init)) + ' run';
    } else if (lockIdx >= 0 && i < lockIdx) {
      lock.className = 'lock';
      lock.textContent = 'Still shifting between runs';
    } else {
      lock.className = 'lock';
      lock.textContent = 'Not locked on';
    }

    renderSkill();

    const target = meta.target_date + 'T00:00Z';
    $('#obs-title').textContent = 'Observed on ' + dShort(target);
    $('#obs-line').innerHTML = '<b>' + D.observed.n + '</b> IMD land cells reached ' + THR + ' mm, ' +
      (D.observed.f_obs * 100).toFixed(1) + '% of land cells in the domain' +
      (D.observed.max_mm != null ? '. Wettest cell ' + Math.round(D.observed.max_mm) + ' mm.' : '.');
    $('#obs-note').textContent = 'The rain day is ' + meta.rain_day + '. These scores describe one event. General skill needs the event library.';
    $('#provenance').innerHTML = 'Forecast: ' + esc(meta.forecast) + ', ' + esc(meta.forecast_credit) + '. Observations: ' + esc(meta.truth) +
      '. Built ' + esc(dLong(meta.generated_utc)) + (meta.commit ? ' from commit <code>' + esc(meta.commit) + '</code>' : '') + '.';
  }

  // ---------------------------------------------------------------- skill chart / table
  const chartEl = $('#chart');
  function renderSkill() {
    const showChart = state.view === 'chart';
    $('#view-chart').setAttribute('aria-pressed', String(showChart));
    $('#view-table').setAttribute('aria-pressed', String(!showChart));
    chartEl.hidden = !showChart;
    $('#table').hidden = showChart;
    const u = runs[state.run].useful;
    $('#chart-cap').textContent = 'Fractions skill score against IMD observed rain, by tile size. Dashed line: useful skill, 0.5 + f/2 = ' + f2(u) + '.' +
      (showChart ? ' Grey lines: the other runs.' : ' Bold: at or above useful skill. Underlined: earned scale.');
    if (showChart) renderChart(); else renderTable();
  }

  function renderChart() {
    const run = runs[state.run];
    const W = Math.max(260, chartEl.clientWidth || 390), H = 196;
    const m = { l: 34, r: 16, t: 18, b: 34 };
    const allF = runs.flatMap(r => r.fss).filter(v => v != null);
    const yMax = Math.max(0.8, Math.ceil((Math.max(...allF) + 0.05) * 5) / 5);
    const x = km => m.l + (km / (maxKm * 1.04)) * (W - m.l - m.r);
    const y = v => m.t + (1 - v / yMax) * (H - m.t - m.b);
    chartEl.innerHTML = '';
    const s = mk('svg', { viewBox: '0 0 ' + W + ' ' + H, height: H, role: 'img', 'aria-label': 'Fractions skill score by tile size for the selected run' }, chartEl);
    const gGrid = mk('g', { class: 'grid' }, s), gAx = mk('g', { class: 'axis' }, s);
    for (let v = 0; v <= yMax + 1e-9; v += 0.2) {
      mk('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, gGrid);
      const t = mk('text', { x: m.l - 7, y: y(v) + 3.5, 'text-anchor': 'end' }, gAx); t.textContent = v === 0 ? '0' : v.toFixed(1);
    }
    windows.forEach(w => { const t = mk('text', { x: x(w.km), y: H - m.b + 15, 'text-anchor': 'middle' }, gAx); t.textContent = w.km; });
    const xt = mk('text', { x: W - m.r, y: H - 3, 'text-anchor': 'end', class: 'axis-title' }, s); xt.textContent = 'Tile size, km';
    const yt = mk('text', { x: m.l - 7, y: 9, 'text-anchor': 'end', class: 'axis-title' }, s); yt.textContent = 'FSS';

    const line = r => 'M' + windows.map((w, k) => x(w.km).toFixed(1) + ',' + y(r.fss[k]).toFixed(1)).join('L');
    runs.forEach((r, i) => { if (i !== state.run) mk('path', { d: line(r), class: 'other' }, s); });
    mk('line', { x1: m.l, x2: W - m.r, y1: y(run.useful), y2: y(run.useful), class: 'useful' }, s);
    const ul = mk('text', { x: m.l + 4, y: y(run.useful) - 6, class: 'useful-label' }, s); ul.textContent = 'Useful skill ' + f2(run.useful);
    mk('path', { d: line(run), class: 'sel' }, s);
    windows.forEach((w, k) => {
      const v = run.fss[k];
      const pass = v >= run.useful;
      const earned = w.cells === run.earned_cells;
      mk('circle', { cx: x(w.km), cy: y(v), r: earned ? 5.5 : 4, class: 'pt' + (pass ? ' pass' : '') + (earned ? ' earned' : '') }, s);
      if (earned) { const t = mk('text', { x: x(w.km), y: y(v) - 11, 'text-anchor': 'middle', class: 'earned-label' }, s); t.textContent = 'Earned'; }
      const hit = mk('circle', { cx: x(w.km), cy: y(v), r: 12, class: 'hit' }, s);
      hit.addEventListener('pointerenter', () => {
        let t = chartEl.querySelector('.tip');
        if (!t) { t = document.createElement('div'); t.className = 'tip'; chartEl.appendChild(t); }
        t.innerHTML = '<b>' + w.km + ' km tile</b><br>FSS ' + f2(v) + (pass ? ', above useful skill' : ', below useful skill');
        const sx = chartEl.clientWidth / W;
        t.style.left = Math.min(chartEl.clientWidth - 150, x(w.km) * sx + 10) + 'px';
        t.style.top = Math.max(0, y(v) - 44) + 'px';
        t.hidden = false;
      });
      hit.addEventListener('pointerleave', () => { const t = chartEl.querySelector('.tip'); if (t) t.hidden = true; });
    });
  }

  function renderTable() {
    const head = '<tr><th>Run</th>' + windows.map(w => '<th>' + w.km + ' km</th>').join('') + '</tr>';
    const body = runs.map((r, i) => '<tr class="' + (i === state.run ? 'sel' : '') + '"><td>' + esc(dShort(r.init)) + ' <span style="color:var(--ink-3)">D−' + r.lead_days + '</span></td>' +
      windows.map((w, k) => {
        const v = r.fss[k];
        const cls = (v >= r.useful ? 'pass' : '') + (w.cells === r.earned_cells ? ' earned' : '');
        return '<td class="' + cls + '">' + f2(v) + '</td>';
      }).join('') + '</tr>').join('');
    $('#table').innerHTML = '<table class="skill"><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
  }

  // ---------------------------------------------------------------- selection & controls
  function select(i, animate) {
    state.run = Math.max(0, Math.min(runs.length - 1, i));
    runBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(k === state.run)));
    if (P) drawData(animate);
    renderRail();
  }
  function setMode(mode) {
    if (state.mode === mode) return;
    state.mode = mode;
    $('#mode-earned').setAttribute('aria-pressed', String(mode === 'earned'));
    $('#mode-raw').setAttribute('aria-pressed', String(mode === 'raw'));
    renderLegend();
    if (P) drawData(true);
  }
  function toggleObs() {
    state.observed = !state.observed;
    $('#obs-btn').setAttribute('aria-pressed', String(state.observed));
    drawObserved();
  }
  function setPlayUi() {
    $('#play-btn').setAttribute('aria-pressed', String(state.playing));
    $('#play-label').textContent = state.playing ? 'Pause' : 'Play runs';
    $('#play-icon').innerHTML = state.playing ? '<path d="M2.5 1.5h2.6v9H2.5zM6.9 1.5h2.6v9H6.9z"/>' : '<path d="M2.5 1.2v9.6L10.5 6z"/>';
  }
  function stop() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null; state.playing = false; setPlayUi();
  }
  function play() {
    if (state.run >= runs.length - 1) select(0, true);
    state.playing = true; setPlayUi();
    state.timer = setInterval(() => {
      if (state.run >= runs.length - 1) { stop(); return; }
      select(state.run + 1, true);
    }, reduceMotion.matches ? 2200 : 1700);
  }
  $('#play-btn').addEventListener('click', () => (state.playing ? stop() : play()));
  $('#mode-earned').addEventListener('click', () => setMode('earned'));
  $('#mode-raw').addEventListener('click', () => setMode('raw'));
  $('#empty-raw').addEventListener('click', () => setMode('raw'));
  $('#obs-btn').addEventListener('click', toggleObs);
  function setZoom(z) {
    if (state.zoom === z && !state.panZoom) return;
    state.zoom = z;
    state.panZoom = null;
    $('#zoom-event').setAttribute('aria-pressed', String(z === 'event'));
    $('#zoom-domain').setAttribute('aria-pressed', String(z === 'domain'));
    $('#zoom-india').setAttribute('aria-pressed', String(z === 'india'));
    hideTip();
    layout();
  }
  $('#zoom-event').addEventListener('click', () => setZoom('event'));
  $('#zoom-domain').addEventListener('click', () => setZoom('domain'));
  $('#zoom-india').addEventListener('click', () => setZoom('india'));
  $('#view-chart').addEventListener('click', () => { state.view = 'chart'; renderSkill(); });
  $('#view-table').addEventListener('click', () => { state.view = 'table'; renderSkill(); });

  document.addEventListener('keydown', e => {
    if (e.target.closest && e.target.closest('input, textarea, select')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowRight') { stop(); select(state.run + 1, true); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { stop(); select(state.run - 1, true); e.preventDefault(); }
    else if (e.key === ' ' && !(e.target.closest && e.target.closest('button'))) { state.playing ? stop() : play(); e.preventDefault(); }
    else if (e.key === 'r' || e.key === 'R') setMode('raw');
    else if (e.key === 'e' || e.key === 'E') setMode('earned');
    else if (e.key === 'o' || e.key === 'O') toggleObs();
    else if (e.key === 'z' || e.key === 'Z') { const cycle = { event: 'domain', domain: 'india', india: 'event' }; setZoom(cycle[state.zoom] || 'event'); }
  });

  // ---------------------------------------------------------------- theme
  const themeBtn = $('#theme-btn');
  const root = document.documentElement;
  const currentTheme = () => root.getAttribute('data-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const setThemeUi = () => { $('#theme-label').textContent = currentTheme() === 'dark' ? 'Light' : 'Dark'; };
  try { const saved = localStorage.getItem('resolve-theme'); if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved); } catch (e) { /* storage unavailable */ }
  themeBtn.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('resolve-theme', next); } catch (e) { /* storage unavailable */ }
    setThemeUi();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', setThemeUi);
  new MutationObserver(setThemeUi).observe(root, { attributes: true, attributeFilter: ['data-theme'] });

  // ---------------------------------------------------------------- boot
  renderHeader();
  renderLegend();
  setThemeUi();
  select(state.run, false);
  let raf = 0;
  const ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { layout(); if (state.view === 'chart') renderChart(); }); });
  ro.observe(mapEl);
  ro.observe(chartEl);
  const boot = () => layout();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(boot); else boot();
  boot();
})();
