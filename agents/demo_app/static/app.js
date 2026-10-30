// Lõi giao diện: đọc /api/manifest, dựng thanh chọn agent/demo và nạp tệp demos/<js> khi cần.
// Mỗi demo tự đăng ký: DEMOS[id] = { mount(root, demoMeta) }.
window.DEMOS = {};

const H = window.H = {
  el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== undefined) e.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
    return e;
  },
  async api(id, params = {}) {
    const q = new URLSearchParams(params).toString();
    const r = await fetch(`/api/demo/${id}?${q}`);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  },
  select(label, id, options, value) {
    const s = H.el('select', { id }, options.map(([v, t]) => H.el('option', { value: v }, t)));
    if (value !== undefined) s.value = value;
    return H.el('label', {}, label, s);
  },
  number(label, id, value, step = 'any', min, max) {
    return H.el('label', {}, label, H.el('input', { id, type: 'number', value, step, min, max, style: 'width:90px' }));
  },
  table(header, rows) {
    return H.el('div', { class: 'scroll' }, H.el('table', {}, H.el('thead', {}, H.el('tr', {}, header.map(h => H.el('th', {}, h)))),
      H.el('tbody', {}, rows.map(r => H.el('tr', {}, r.map(c => H.el('td', {}, c ?? '')))))));
  },
  stat(value, label) { return H.el('div', { class: 'stat' }, H.el('b', {}, value), H.el('span', {}, label)); },
  canvas(w, h) { const c = H.el('canvas', { width: w, height: h }); return c; },
  css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); },

  // Biểu đồ đường: series = [{name, color, data:[...]}]
  line(canvas, series, { yLabel = '', min, max } = {}) {
    const g = canvas.getContext('2d'), W = canvas.width, Hh = canvas.height, pad = { l: 52, r: 12, t: 12 + 19 * series.length, b: 30 };
    g.clearRect(0, 0, W, Hh);
    const all = series.flatMap(s => s.data);
    const lo = min ?? Math.min(...all), hi = max ?? Math.max(...all), span = (hi - lo) || 1;
    g.font = '14px sans-serif'; g.fillStyle = H.css('--ink'); g.strokeStyle = '#c9d9e8';
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (Hh - pad.t - pad.b) * i / 4; g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
      g.fillText((hi - span * i / 4).toFixed(span < 5 ? 2 : 0), 4, y + 4);
    }
    g.fillText(yLabel, pad.l, Hh - 8);
    series.forEach((s, k) => {
      g.strokeStyle = s.color; g.lineWidth = 2; g.beginPath();
      s.data.forEach((v, i) => {
        const x = pad.l + (W - pad.l - pad.r) * i / Math.max(1, s.data.length - 1), y = pad.t + (Hh - pad.t - pad.b) * (1 - (v - lo) / span);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke(); g.fillStyle = s.color; g.fillRect(W - 190, 6 + 19 * k, 13, 13); g.fillStyle = H.css('--ink'); g.fillText(s.name, W - 172, 18 + 19 * k);
    });
  },

  // Biểu đồ cột nhóm: groups = [{label, values:[...]}], names = tên từng cột trong nhóm
  bars(canvas, groups, names, colors, { fmt = v => v.toFixed(2), max } = {}) {
    const g = canvas.getContext('2d'), W = canvas.width, Hh = canvas.height, pad = { l: 12, r: 12, t: 14 + 18 * names.length, b: 34 };
    g.clearRect(0, 0, W, Hh);
    const hi = (max ?? Math.max(...groups.flatMap(x => x.values)) * 1.15) || 1, gw = (W - pad.l - pad.r) / groups.length;
    const bw = gw * 0.8 / names.length;
    g.font = 'bold 14px sans-serif';
    groups.forEach((grp, i) => {
      grp.values.forEach((v, j) => {
        const h = (Hh - pad.t - pad.b) * Math.max(0, v) / hi, x = pad.l + gw * i + gw * 0.1 + bw * j, y = Hh - pad.b - h;
        g.fillStyle = colors[j]; g.fillRect(x, y, bw - 2, h); g.fillStyle = H.css('--ink'); g.textAlign = 'center'; g.fillText(fmt(v), x + bw / 2, y - 4);
      });
      g.fillStyle = H.css('--ink'); g.textAlign = 'center'; g.fillText(grp.label, pad.l + gw * (i + 0.5), Hh - 10);
    });
    g.textAlign = 'left';
    names.forEach((n, j) => { g.fillStyle = colors[j]; g.fillRect(W - 200, 4 + 18 * j, 12, 12); g.fillStyle = H.css('--ink'); g.fillText(n, W - 183, 15 + 18 * j); });
  },

  // Vẽ bản đồ lưới với mèo, chuột và vệt đường đi (map = {w,h,grid})
  drawMap(canvas, map, cat, mouse, trail = []) {
    const cell = Math.min(Math.floor(canvas.parentNode.clientWidth / map.w) || 24, 34);
    canvas.width = cell * map.w; canvas.height = cell * map.h;
    const g = canvas.getContext('2d');
    for (let r = 0; r < map.h; r++) for (let c = 0; c < map.w; c++) {
      g.fillStyle = map.grid[r][c] === '#' ? H.css('--wall') : ((r + c) % 2 ? H.css('--floor') : H.css('--floor2'));
      g.fillRect(c * cell, r * cell, cell, cell);
    }
    g.fillStyle = 'rgba(240,96,58,.55)';
    trail.forEach(([r, c]) => g.fillRect(c * cell + cell * .25, r * cell + cell * .25, cell * .5, cell * .5));
    g.font = `${Math.floor(cell * .9)}px "Segoe UI Emoji","Apple Color Emoji",sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('🐭', mouse[1] * cell + cell / 2, mouse[0] * cell + cell / 2 + 1);
    g.fillText('🐱', cat[1] * cell + cell / 2, cat[0] * cell + cell / 2 + 1);
    return cell;
  },
};

const state = { manifest: null, agent: null, demo: null };

function show() {
  const agent = state.manifest.agents.find(a => a.id === state.agent);
  document.getElementById('title').textContent = agent.title;
  document.getElementById('subtitle').textContent = agent.subtitle;
  const tabs = document.getElementById('agentTabs'); tabs.replaceChildren();
  state.manifest.agents.forEach(a => tabs.append(H.el('button', { class: a.id === state.agent ? 'on' : '', onclick: () => go(a.id) }, a.id === 'agent01' ? 'Agent 01' : 'Agent 02')));
  const nav = document.getElementById('nav'); nav.replaceChildren();
  agent.demos.forEach(d => nav.append(H.el('button', { class: d.id === state.demo ? 'on' : '', onclick: () => go(state.agent, d.id) }, d.title, H.el('small', {}, d.description))));
  const demo = agent.demos.find(d => d.id === state.demo), panel = document.getElementById('panel');
  panel.replaceChildren(H.el('h2', {}, demo.title), H.el('p', { class: 'desc' }, demo.description));
  const body = H.el('div'); panel.append(body);
  const mount = () => window.DEMOS[demo.js].mount(body, demo);
  if (window.DEMOS[demo.js]) return mount();
  const s = document.createElement('script'); s.src = `demos/${demo.js}`; s.onload = mount;
  s.onerror = () => body.append(H.el('p', { class: 'err' }, `Không nạp được demos/${demo.js}`));
  document.head.append(s);
}

function go(agent, demo) {
  const a = state.manifest.agents.find(x => x.id === agent);
  state.agent = agent; state.demo = demo || a.demos[0].id;
  location.hash = `${state.agent}/${state.demo}`;
}

window.addEventListener('hashchange', () => {
  const [agent, demo] = location.hash.slice(1).split('/');
  if (state.manifest && state.manifest.agents.some(a => a.id === agent)) { state.agent = agent; state.demo = demo || state.manifest.agents.find(a => a.id === agent).demos[0].id; show(); }
});

fetch('/api/manifest').then(r => r.json()).then(m => {
  state.manifest = m; state.agent = m.agents[0].id; state.demo = m.agents[0].demos[0].id;
  if (location.hash) window.dispatchEvent(new Event('hashchange')); else go(state.agent);
});
