/* Phòng khám AI - logic giao diện. Phụ thuộc: scene.js (Clinic). ?mock=1 đọc dữ liệu mẫu trong fixtures/. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var MOCK = /[?&]mock=1/.test(location.search);
  var COLORS = { doctor: '#2589e0', predictor: '#7060d8', knowledge: '#0d9f8f', coach: '#f0603a', guardian: '#e8a317', reporter: '#e8508a' };
  var SHIRTS = ['#2589e0', '#f0603a', '#16a34a', '#8b5cf6', '#e8508a', '#e8a317', '#0d9f8f', '#475569'];
  var ZONES = Clinic.ZONES;

  var S = { meta: null, agents: {}, values: {}, persona: null, shirt: SHIRTS[0], avName: '', kg: null, mode: 'consult', consult: null,
    fast: false, paused: false, playToken: 0, playing: false, life: null, glbWebgl: true };

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function pct(v) { return Math.round((v || 0) * 100); }
  function fmt1(v) { return (Math.round(v * 10) / 10).toString().replace('.', ','); }

  /* ---------- API ---------- */
  var FIX = { '/api/meta': 'meta', '/api/consult': 'consult', '/api/kg': 'kg', '/api/life/start': 'life_start', '/api/life/step': 'life_step', '/api/life/compare': 'life_compare' };
  var fixCache = {};
  function loadFixture(name) {
    if (fixCache[name]) return Promise.resolve(JSON.parse(JSON.stringify(fixCache[name])));
    return fetch('fixtures/' + name + '.json').then(function (r) { if (!r.ok) throw new Error('fixture'); return r.json(); }).then(function (j) { fixCache[name] = j; return JSON.parse(JSON.stringify(j)); });
  }
  function api(path, body) {
    if (MOCK) return mockApi(path, body);
    var opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    return fetch(path, opt).then(function (r) {
      if (!r.ok) return r.text().then(function (t) { var m = t; try { m = JSON.parse(t).detail || JSON.parse(t).error || t; } catch (e) {} throw new Error('Máy chủ báo lỗi ' + r.status + (m ? ': ' + String(m).slice(0, 160) : '')); });
      return r.json();
    }, function () { throw new Error('Không kết nối được máy chủ (http://localhost:8200). Hãy kiểm tra máy chủ đã chạy chưa.'); });
  }
  function mockApi(path, body) {
    var delay = function (v) { return new Promise(function (r) { setTimeout(function () { r(v); }, 350); }); };
    if (path === '/api/kg/search') return loadFixture('kg').then(function (kg) { return loadFixture('consult').then(function (c) { return delay(mockSearch(kg, c, body.q)); }); });
    if (path === '/api/life/step') return loadFixture('life_start').then(function (st) { return delay(mockLifeStep(body.state, body.action, st.actions)); });
    if (path === '/api/life/start' || path === '/api/life/compare' || FIX[path]) return loadFixture(FIX[path]).then(delay);
    return Promise.reject(new Error('Không có dữ liệu mẫu cho ' + path));
  }
  function fold(s) { return String(s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd'); }
  function mockSearch(kg, consult, q) {
    var toks = fold(q).split(/\s+/).filter(function (t) { return t.length > 1; }), hits = [];
    kg.nodes.forEach(function (n) { var txt = fold(n.vi + ' ' + n.label + ' ' + n.desc + ' ' + n.type); var sc = toks.filter(function (t) { return txt.indexOf(t) >= 0; }).length; if (sc) hits.push([sc, n.id]); });
    hits.sort(function (a, b) { return b[0] - a[0]; });
    var ids = hits.slice(0, 8).map(function (h) { return h[1]; });
    return { nodes: ids, passages: consult.summary.citations.map(function (c, i) { return { id: c.id, quote: c.quote, url: c.url, score: 0.82 - i * 0.12 }; }), answer: ids.length ? 'Tìm thấy ' + ids.length + ' khái niệm liên quan đến "' + q + '" (dữ liệu mẫu).' : 'Không có khái niệm khớp. Hãy thử từ khóa khác.' };
  }
  function zoneOf(g) { return g < 70 ? 'low' : g <= 180 ? 'ok' : g <= 250 ? 'high' : 'danger'; }
  function mockLifeStep(state, action, actions) {
    var d = { rest: -2, sweet: 32, light_meal: 20, normal_meal: 42, heavy_meal: 78, walk: -24, workout: -40, skip: 0 };
    var g0 = state.glucose, g1 = clamp(g0 + (d[action] == null ? 0 : d[action]) + (112 - g0) * 0.08 + (Math.random() * 8 - 4), 48, 330);
    var log = (state.log || []).concat([{ hour: state.hour, glucose: g1, zone: zoneOf(g1) }]);
    var hour = state.hour + 1, done = hour >= 30, tir = log.filter(function (l) { return l.zone === 'ok'; }).length / log.length;
    var ns = { hour: hour, glucose: g1, carbs_on_board: Math.max(0, (state.carbs_on_board || 0) * 0.5 + (d[action] > 10 ? d[action] : 0)), activity_on_board: 0, log: log };
    var coach = g1 < 85 ? 'light_meal' : g1 > 190 ? 'walk' : (hour === 12 || hour === 18) ? 'normal_meal' : 'rest';
    var rule = g1 < 70 ? 'sweet' : g1 > 250 ? 'walk' : 'rest';
    var why = { light_meal: 'Đường hơi thấp, ăn nhẹ để lấy lại năng lượng.', walk: 'Đường đang cao, đi bộ giúp hạ.', normal_meal: 'Đến giờ ăn chính, ăn cân đối.', rest: 'Đường ổn, nghỉ ngơi và theo dõi.', sweet: 'Đường quá thấp, ăn nhanh chút đồ ngọt.' };
    return { state: ns, glucose_before: g0, glucose_after: g1, reward: 1, zone: zoneOf(g1), events: ['Hành động "' + action + '": đường huyết ' + Math.round(g0) + ' → ' + Math.round(g1) + ' mg/dL.'],
      advice: { coach: { action: coach, why: why[coach] }, rule: { action: rule, why: why[rule] } }, done: done,
      score: { time_in_range: tir, hypo_hours: log.filter(function (l) { return l.zone === 'low'; }).length, hyper_hours: log.filter(function (l) { return l.zone === 'high' || l.zone === 'danger'; }).length } };
  }

  /* ---------- thông báo ---------- */
  function banner(msg, opt) {
    opt = opt || {}; var b = $('#banner'); b.className = 'banner' + (opt.info ? ' info' : ''); b.hidden = false;
    b.innerHTML = '<div>' + esc(msg) + '</div>';
    if (opt.retry) { var r = document.createElement('button'); r.className = 'btn sm'; r.textContent = '↻ Thử lại'; r.onclick = function () { b.hidden = true; opt.retry(); }; b.appendChild(r); }
    if (opt.mock && !MOCK) { var m = document.createElement('a'); m.className = 'btn sm'; m.style.textDecoration = 'none'; m.textContent = 'Mở chế độ demo (?mock=1)'; m.href = location.pathname + '?mock=1'; b.appendChild(m); }
    var c = document.createElement('button'); c.className = 'btn sm'; c.textContent = 'Đóng'; c.onclick = function () { b.hidden = true; }; b.appendChild(c);
  }
  function hideBanner() { $('#banner').hidden = true; }

  /* ---------- hồ sơ / biểu mẫu ---------- */
  var GROUPS = [
    ['Thông tin cơ bản', '👤', ['age_years', 'height_cm', 'weight_kg']],
    ['Bệnh lý và xét nghiệm', '🩺', ['high_bp', 'high_chol', 'chol_check', 'stroke', 'heart_disease', 'diff_walking']],
    ['Lối sống', '🏃', ['smoker', 'phys_activity', 'fruits_daily', 'veggies_daily', 'heavy_alcohol']],
    ['Sức khỏe tổng quát', '💚', ['gen_health', 'mental_bad_days', 'physical_bad_days']],
    ['Tiếp cận y tế và xã hội', '🏥', ['any_healthcare', 'no_doc_cost', 'education', 'income']],
    ['Triệu chứng và đường huyết', '🩸', ['fasting_glucose', 'symptoms']]
  ];
  var SYMPTOM_FIELD = { key: 'symptoms', label: 'Triệu chứng gần đây', type: 'multi', default: [], options: [['thirst', 'Khát nhiều'], ['frequent_urination', 'Tiểu nhiều'], ['fatigue', 'Mệt mỏi'], ['blurred_vision', 'Nhìn mờ'], ['slow_healing', 'Vết thương lâu lành']] };
  var SEX_FIELD = { key: 'sex', label: 'Giới tính', type: 'select', options: [['female', 'Nữ'], ['male', 'Nam']], default: 'female' };

  function fieldDefaults() {
    var v = {}; S.meta.fields.forEach(function (f) { v[f.key] = f.default == null ? (f.type === 'multi' ? [] : null) : (Array.isArray(f.default) ? f.default.slice() : f.default); }); return v;
  }
  function bmiOf(v) { var h = Number(v.height_cm) / 100, w = Number(v.weight_kg); return h > 0 && w > 0 ? w / (h * h) : null; }
  function bmiInfo(b) { if (b == null) return ['', '#888']; return b < 18.5 ? ['Gầy', '#2f7bf0'] : b < 23 ? ['Bình thường', '#16a34a'] : b < 25 ? ['Hơi thừa cân', '#e8a317'] : b < 30 ? ['Thừa cân', '#f0603a'] : ['Béo phì', '#e5342d']; }

  function syncAvatar() {
    var v = S.values, b = bmiOf(v), inf = bmiInfo(b);
    $('#bmiLine').innerHTML = b == null ? '' : 'BMI: <b style="background:' + inf[1] + '">' + fmt1(b) + ' · ' + inf[0] + '</b>';
    Clinic.setPatient({ name: S.avName || 'Bạn', sex: v.sex, color: S.shirt, bmi: b });
    $$('#avSex button').forEach(function (bt) { bt.classList.toggle('on', bt.dataset.v === v.sex); });
  }

  /* giữ trang đứng yên: focus vào ô ẩn không được cuộn cả tài liệu (gây màn hình trống) */
  window.addEventListener('scroll', function () { if (window.scrollY || window.scrollX) window.scrollTo(0, 0); });
  [document.documentElement, document.body].forEach(function (el) { el.addEventListener('scroll', function () { el.scrollTop = 0; el.scrollLeft = 0; }); });

  function buildForm() {
    var meta = S.meta, fields = meta.fields.slice();
    if (!fields.some(function (f) { return f.key === 'symptoms'; })) fields.push(SYMPTOM_FIELD);
    if (!fields.some(function (f) { return f.key === 'sex'; })) fields.push(SEX_FIELD);
    meta.fields = fields; S.values = fieldDefaults();
    // vai
    var ph = '<button class="persona custom active" data-id=""><b>✍️ Tự nhập hồ sơ</b><span>Điền biểu mẫu bên dưới</span></button>';
    meta.personas.forEach(function (p) { ph += '<button class="persona" data-id="' + esc(p.id) + '"><b>' + esc(p.name) + '</b><span>' + esc(p.story) + '</span></button>'; });
    $('#personas').innerHTML = ph;
    $$('#personas .persona').forEach(function (b) { b.onclick = function () { choosePersona(b.dataset.id); }; });
    // màu áo
    $('#avColors').innerHTML = SHIRTS.map(function (c, i) { return '<button class="swatch' + (i === 0 ? ' on' : '') + '" style="background:' + c + '" data-c="' + c + '" aria-label="Màu áo ' + c + '"></button>'; }).join('');
    $$('#avColors .swatch').forEach(function (b) { b.onclick = function () { S.shirt = b.dataset.c; $$('#avColors .swatch').forEach(function (x) { x.classList.toggle('on', x === b); }); syncAvatar(); }; });
    $$('#avSex button').forEach(function (b) { b.onclick = function () { S.values.sex = b.dataset.v; renderFormValues(); syncAvatar(); markCustom(); }; });
    $('#avName').oninput = function (e) { S.avName = e.target.value; syncAvatar(); };
    // nhóm
    var byKey = {}; fields.forEach(function (f) { byKey[f.key] = f; });
    var used = { sex: true }, html = '';
    used.name = true;
    var groups;
    if (fields.some(function (f) { return f.group; })) {
      var order = [], map = {}, icons = ['👤', '📏', '🩺', '🏃', '💚', '🏥', '🩸', '➕'];
      fields.forEach(function (f) { if (used[f.key]) return; var g = f.group || 'Khác'; if (!map[g]) { map[g] = []; order.push(g); } map[g].push(f.key); });
      groups = order.map(function (g, i) { return [g, icons[i % icons.length], map[g]]; });
    } else groups = GROUPS.map(function (g) { return [g[0], g[1], g[2].filter(function (k) { return byKey[k]; })]; });
    var rest = fields.filter(function (f) { return !used[f.key] && !groups.some(function (g) { return g[2].indexOf(f.key) >= 0; }); }).map(function (f) { return f.key; });
    if (rest.length) groups.push(['Khác', '➕', rest]);
    groups.forEach(function (g, gi) {
      if (!g[2].length) return;
      html += '<details class="group"' + (gi < 2 ? ' open' : '') + '><summary><span>' + g[1] + '</span>' + esc(g[0]) + '</summary><div class="gbody">' + g[2].map(function (k) { return fieldHtml(byKey[k]); }).join('') + '</div></details>';
    });
    $('#formGroups').innerHTML = html;
    bindForm(); renderFormValues();
    $('#avName').value = ''; syncAvatar();
  }

  function fieldHtml(f) {
    var k = esc(f.key), help = f.help ? '<span class="help">' + esc(f.help) + '</span>' : '';
    if (f.type === 'bool') return '<div class="bool-row"><div><div class="lbl">' + esc(f.label) + '</div>' + help + '</div><label class="switch"><input type="checkbox" data-k="' + k + '"><span class="slider"></span></label></div>';
    if (f.type === 'select') return '<div class="fld"><label><span>' + esc(f.label) + '</span><select data-k="' + k + '">' + f.options.map(function (o) { return '<option value="' + esc(o[0]) + '">' + esc(o[1]) + '</option>'; }).join('') + '</select></label>' + help + '</div>';
    if (f.type === 'multi') return '<div class="fld"><span>' + esc(f.label) + '</span><div class="multi" data-k="' + k + '">' + f.options.map(function (o) { return '<button type="button" class="chip" data-v="' + esc(o[0]) + '">' + esc(o[1]) + '</button>'; }).join('') + '</div>' + help + '</div>';
    if (f.type === 'text') return '<div class="fld"><label><span>' + esc(f.label) + '</span><input type="text" data-k="' + k + '"></label>' + help + '</div>';
    if (f.optional || f.min == null || f.max == null) return '<div class="fld"><label><span>' + esc(f.label) + (f.unit ? ' (' + esc(f.unit) + ')' : '') + '</span><input type="number" data-k="' + k + '" placeholder="Bỏ trống nếu chưa có"' + (f.min != null ? ' min="' + f.min + '"' : '') + (f.max != null ? ' max="' + f.max + '"' : '') + '></label>' + help + '</div>';
    return '<div class="fld"><span>' + esc(f.label) + '</span><div class="range-row"><input type="range" data-k="' + k + '" min="' + f.min + '" max="' + f.max + '" step="' + (f.step || 1) + '"><span class="range-val" data-val="' + k + '"></span></div>' + help + '</div>';
  }
  function fieldOf(k) { return S.meta.fields.filter(function (f) { return f.key === k; })[0]; }
  function bindForm() {
    $$('#formGroups [data-k]').forEach(function (el) {
      var k = el.dataset.k, f = fieldOf(k);
      if (el.classList.contains('multi')) { $$('.chip', el).forEach(function (c) { c.onclick = function () { var a = S.values[k] || [], v = c.dataset.v, i = a.indexOf(v); if (i >= 0) a.splice(i, 1); else a.push(v); S.values[k] = a; renderFormValues(); markCustom(); }; }); return; }
      el.addEventListener(el.tagName === 'SELECT' || el.type === 'checkbox' ? 'change' : 'input', function () {
        var v;
        if (el.type === 'checkbox') v = el.checked; else if (el.tagName === 'SELECT') { var sample = f.options[0][0]; v = typeof sample === 'number' ? Number(el.value) : el.value; }
        else if (f.type === 'text') v = el.value; else v = el.value === '' ? null : Number(el.value);
        S.values[k] = v; renderFormValues(true); syncAvatar(); markCustom();
      });
    });
  }
  function renderFormValues(skipInputs) {
    S.meta.fields.forEach(function (f) {
      var v = S.values[f.key], el = $('#formGroups [data-k="' + f.key + '"]');
      if (f.key === 'sex') return;
      if (!el) return;
      if (el.classList.contains('multi')) { $$('.chip', el).forEach(function (c) { c.classList.toggle('on', (v || []).indexOf(c.dataset.v) >= 0); }); return; }
      if (el.type === 'checkbox') el.checked = !!v;
      else if (el.tagName === 'SELECT') el.value = v;
      else if (!skipInputs || el.type === 'range') { if (document.activeElement !== el) el.value = v == null ? '' : v; }
      var vv = $('[data-val="' + f.key + '"]'); if (vv) vv.textContent = v + (f.unit ? ' ' + f.unit : '');
    });
  }
  function markCustom() { if (S.persona) { S.persona = null; $$('#personas .persona').forEach(function (b) { b.classList.toggle('active', b.dataset.id === ''); }); } }
  function choosePersona(id) {
    S.persona = id || null; $$('#personas .persona').forEach(function (b) { b.classList.toggle('active', b.dataset.id === id); });
    if (!id) return;
    var p = S.meta.personas.filter(function (x) { return x.id === id; })[0]; if (!p) return;
    S.values = fieldDefaults(); Object.keys(p.profile).forEach(function (k) { S.values[k] = Array.isArray(p.profile[k]) ? p.profile[k].slice() : p.profile[k]; });
    S.avName = p.profile.name || p.name.split(',')[0]; $('#avName').value = S.avName;
    if (p.profile.sex === 'female') S.shirt = SHIRTS[4]; else S.shirt = SHIRTS[0];
    $$('#avColors .swatch').forEach(function (x) { x.classList.toggle('on', x.dataset.c === S.shirt); });
    renderFormValues(); syncAvatar();
  }
  function buildProfile() {
    var p = {}; Object.keys(S.values).forEach(function (k) { var v = S.values[k]; if (v === null || v === undefined || v === '') return; p[k] = v; });
    if (S.avName) p.name = S.avName; else delete p.name;
    return p;
  }

  /* ---------- chuyển chế độ ---------- */
  function switchMode(m) {
    if (S.mode === m) return;
    if (S.playing) { skipConsult(true); }
    S.mode = m; hideBanner();
    $$('.tab').forEach(function (t) { t.classList.toggle('active', t.dataset.mode === m); t.setAttribute('aria-selected', t.dataset.mode === m); });
    ['consult', 'life', 'kg'].forEach(function (v) { $('#view-' + v).hidden = v !== m; });
    $('#panel').scrollTop = 0;
    $('#hud').hidden = !(m === 'consult' && S.consult && $('#consultRun').hidden === false);
    Clinic.setMode(m === 'kg' ? 'kg' : m === 'life' ? 'life' : 'clinic');
    if (m === 'consult') { Clinic.hideAllBubbles(); if (S.consult) { var d = S.consult.events[S.consult.events.length - 1]; if (d && !S.playing) Clinic.say('doctor', d.text, { compact: true }); } Clinic.flyHome(900); }
    if (m === 'life') { restoreLifeBubbles(); }
    if (m === 'kg') { Clinic.highlightKg(null); Clinic.kgAutoRotate(true); ensureKg(); }
    else if (S.consult && m === 'consult') { var kn = (S.consult.summary && S.consult.summary.kg_nodes) || []; Clinic.highlightKg(kn); }
    $('#stageChip').textContent = m === 'kg' ? 'Kéo để xoay · cuộn để phóng to · bấm vào nút để đọc mô tả' : 'Kéo chuột để xoay, cuộn để phóng to';
  }

  /* ---------- KHÁM BỆNH ---------- */
  var sleepers = [];
  function sleep(ms) {
    return new Promise(function (res) { var o = { r: res }; o.t = setTimeout(function () { var i = sleepers.indexOf(o); if (i >= 0) sleepers.splice(i, 1); res(); }, ms); sleepers.push(o); });
  }
  function wakeAll() { sleepers.splice(0).forEach(function (o) { clearTimeout(o.t); o.r(); }); }
  function agentName(id) { return (S.agents[id] && S.agents[id].name) || id; }
  function agentColor(id) { return (S.agents[id] && S.agents[id].color) || COLORS[id] || '#2589e0'; }

  function startConsult() {
    var btn = $('#btnConsult'); hideBanner();
    btn.disabled = true; btn.textContent = '⏳ Đang gọi bác sĩ...';
    Clinic.hideAllBubbles(); Clinic.stopAll(); Clinic.patientReset(); Clinic.flyHome(800);
    var profile = buildProfile(); S.lastProfile = profile;
    var enter = Clinic.patientEnter(), req = api('/api/consult', { profile: profile });
    req.then(function (data) { return enter.then(function () { return data; }); }).then(function (data) {
      btn.disabled = false; btn.textContent = '🚪 Vào khám';
      S.consult = data; playConsult(data);
    }, function (err) {
      btn.disabled = false; btn.textContent = '🚪 Vào khám';
      banner('Không hội chẩn được. ' + err.message, { retry: startConsult, mock: true });
    });
  }

  function addChat(ev) {
    var c = agentColor(ev.agent), log = $('#chatLog'), d = document.createElement('div');
    d.className = 'msg' + (ev.type === 'alert' ? ' alert-' + (ev.level || 'warn') : ''); d.style.setProperty('--c', c);
    var tag = { handoff: 'chuyển giao', tool: 'công cụ', cite: 'trích dẫn', alert: 'cảnh báo', final: 'kết luận' }[ev.type];
    var extra = '';
    if (ev.type === 'tool' && ev.payload && ev.payload.risk) { var r = ev.payload.risk; extra = '<span class="quote">Không ' + pct(r.none) + '% · Tiền tiểu đường ' + pct(r.pre) + '% · Tiểu đường ' + pct(r.diabetes) + '%</span>'; }
    if (ev.type === 'cite' && ev.payload && ev.payload.passages) extra = ev.payload.passages.slice(0, 2).map(function (p) { return '<span class="quote">“' + esc(p.quote) + '”</span>'; }).join('');
    d.innerHTML = '<b>' + esc(agentName(ev.agent)) + (tag ? '<em class="tag" style="font-style:normal">' + tag + '</em>' : '') + '</b><span>' + esc(ev.text) + '</span>' + extra;
    log.appendChild(d); log.scrollTop = log.scrollHeight; $('#panel').scrollTop = 0;
  }

  /* ---------- giọng nói (Azure TTS qua máy chủ) và phụ đề ---------- */
  var Voice = { available: false, on: true, cache: {}, audio: null };
  try { Voice.on = localStorage.getItem('clinic.voice') !== 'off'; } catch (e) {}
  function voiceFetch(ev) {
    var k = ev.agent + '|' + ev.text; if (Voice.cache[k]) return Voice.cache[k];
    Voice.cache[k] = fetch('/api/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: ev.text, agent: ev.agent }) })
      .then(function (r) { if (!r.ok) throw new Error('tts'); return r.blob(); })
      .then(function (b) { return URL.createObjectURL(b); })
      .catch(function () { return null; });
    return Voice.cache[k];
  }
  function voicePrefetch(events, from) { if (!Voice.available || !Voice.on) return; for (var j = from; j < Math.min(events.length, from + 3); j++) voiceFetch(events[j]); }
  function loadAudio(url) {
    return new Promise(function (res) {
      var a = new Audio(url), done = false; function fin(v) { if (!done) { done = true; res(v); } }
      a.addEventListener('loadedmetadata', function () { fin(a); }); a.addEventListener('error', function () { fin(null); }); setTimeout(function () { fin(null); }, 4000);
    });
  }
  function stopVoice() { if (Voice.audio) { try { Voice.audio.pause(); } catch (e) {} Voice.audio = null; } }
  function syncVoiceButton() {
    var b = $('#btnVoice'); b.hidden = !Voice.available; b.classList.toggle('voice-off', !Voice.on); b.textContent = Voice.on ? '🔊 Giọng nói' : '🔇 Đã tắt tiếng';
  }
  var capTimer = null;
  function showCaption(ev, typeMs) {
    var el = $('#hudCaption'); if (capTimer) { clearInterval(capTimer); capTimer = null; }
    el.style.setProperty('--c', agentColor(ev.agent)); el.classList.add('on');
    var text = ev.text, i = 0, step = 30, per = Math.max(1, Math.ceil(text.length / Math.max(1, typeMs / step)));
    function paint(n) { el.innerHTML = '<b>' + esc(agentName(ev.agent)) + '</b>' + esc(text.slice(0, n)); }
    paint(0);
    capTimer = setInterval(function () { if (S.paused) return; i += per; paint(i); if (i >= text.length) { clearInterval(capTimer); capTimer = null; } }, step);
  }
  function hideCaption() { if (capTimer) { clearInterval(capTimer); capTimer = null; } $('#hudCaption').classList.remove('on'); }
  function shortText(t) { return t.length > 64 ? t.slice(0, 61).replace(/\s+\S*$/, '') + '…' : t; }
  async function waitResume(token) { while (S.paused && token === S.playToken) await sleep(150); }
  function setPaused(p) {
    S.paused = !!p; $('#btnPause').textContent = S.paused ? '▶ Tiếp tục' : '⏸ Tạm dừng'; $('#btnPause').style.background = S.paused ? '#ffe9a8' : '';
    if (Voice.audio) { if (S.paused) Voice.audio.pause(); else Voice.audio.play().catch(function () {}); }
  }

  async function playConsult(data) {
    var token = ++S.playToken; S.playing = true;
    $('#consultForm').hidden = true; $('#consultRun').hidden = false; $('#consultResult').innerHTML = ''; $('#chatLog').innerHTML = '';
    $('#hud').hidden = false; $('#btnReplay').disabled = true;
    $('#hudDots').innerHTML = data.events.map(function (e) { return '<i style="--c:' + agentColor(e.agent) + '"></i>'; }).join('');
    var dots = $$('#hudDots i'); Clinic.highlightKg(null);
    setPaused(false); voicePrefetch(data.events, 0);
    for (var i = 0; i < data.events.length; i++) {
      var ev = data.events[i]; if (token !== S.playToken) return;
      await waitResume(token); if (token !== S.playToken) return;
      var sp = S.fast ? 2.6 : 1, col = agentColor(ev.agent);
      voicePrefetch(data.events, i + 1);
      // giọng nói quyết định nhịp: nói xong mới sang câu tiếp; không có tiếng thì cho đủ thời gian đọc (~60 ms/ký tự)
      var audio = null;
      if (Voice.available && Voice.on && !S.fast) { var url = await voiceFetch(ev); if (token !== S.playToken) return; audio = url ? await loadAudio(url) : null; if (token !== S.playToken) return; }
      var voiceMs = audio && isFinite(audio.duration) ? audio.duration * 1000 : 0;
      var ms = Math.max(Math.max(600, (ev.ms || 2000)) / sp, voiceMs ? voiceMs + 500 : Math.max(2000, ev.text.length * 60) / sp);
      dots.forEach(function (d, j) { d.className = j < i ? 'done' : j === i ? 'cur' : ''; });
      addChat(ev); Clinic.speak(ev.agent, true);
      Clinic.hideAllBubbles(); Clinic.say(ev.agent, shortText(ev.text), { mini: true });
      showCaption(ev, voiceMs ? voiceMs * 0.9 : ms * 0.8);
      stopVoice(); if (audio) { Voice.audio = audio; audio.play().catch(function () {}); }
      if (ev.type === 'handoff' && ev.to) { Clinic.flyToChar(ev.agent, 700 / sp); Clinic.beam(ev.agent, ev.to, col, ms * 0.9); setTimeout(function (to, t) { if (t === S.playToken) Clinic.flyToChar(to, 800 / sp); }, ms * 0.5, ev.to, token); }
      else if (ev.type === 'cite') { Clinic.flyToBoard(1100 / sp); Clinic.highlightKg((ev.payload && ev.payload.kg_nodes) || []); }
      else Clinic.flyToChar(ev.focus || ev.agent, 1100 / sp);
      if (ev.type === 'tool') Clinic.work(ev.agent, true);
      if (ev.type === 'alert') Clinic.alert(ev.level || 'warn', ms);
      await sleep(ms); if (token !== S.playToken) return;
      await waitResume(token); if (token !== S.playToken) return;
      if (audio && !audio.ended && !audio.paused) await new Promise(function (res) { audio.onended = res; setTimeout(res, 4000); });
      Clinic.work(ev.agent, false);
    }
    if (token !== S.playToken) return;
    finishConsult(data, false);
  }

  function skipConsult(silent) {
    if (!S.playing) return;
    S.playToken++; wakeAll(); stopVoice(); hideCaption(); Clinic.stopAll(); Clinic.hideAllBubbles(); finishConsult(S.consult, true, silent);
  }

  function finishConsult(data, skipped, silent) {
    S.playing = false; stopVoice(); hideCaption(); setPaused(false); Clinic.speak(null, false); Clinic.stopAll(); Clinic.patientAtSpot();
    var last = data.events[data.events.length - 1]; Clinic.hideAllBubbles();
    $$('#hudDots i').forEach(function (d) { d.className = 'done'; }); $('#btnReplay').disabled = false;
    if (skipped) { $('#chatLog').innerHTML = ''; data.events.forEach(addChat); }
    var cites = data.events.filter(function (e) { return e.type === 'cite' && e.payload; }); var nodes = (data.summary && data.summary.kg_nodes) || (cites[0] && cites[0].payload.kg_nodes) || [];
    Clinic.highlightKg(nodes);
    if (!silent && S.mode === 'consult') { Clinic.flyHome(1200); Clinic.say('doctor', (data.summary && data.summary.headline) || last.text, { compact: true, sticky: true }); }
    renderResult(data); $('#panel').scrollTop = $('#panel').scrollHeight; setTimeout(function () { var r = $('#consultResult'); if (r.firstChild) r.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 80);
  }

  function gaugeSVG(risk) {
    var v = clamp((risk.diabetes || 0) + 0.5 * (risk.pre || 0), 0, 1), cx = 150, cy = 140, r = 110;
    function pt(a, rr) { return [cx + rr * Math.cos(Math.PI * (1 - a)), cy - rr * Math.sin(Math.PI * (1 - a))]; }
    function arc(a0, a1, col) { var p0 = pt(a0, r), p1 = pt(a1, r); return '<path d="M' + p0[0].toFixed(1) + ' ' + p0[1].toFixed(1) + ' A' + r + ' ' + r + ' 0 0 1 ' + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1) + '" stroke="' + col + '" stroke-width="26" fill="none"/>'; }
    var n = pt(v, 86), s = '<svg class="gauge" viewBox="0 0 300 170" role="img" aria-label="Đồng hồ nguy cơ">';
    s += arc(0, 0.34, '#22b05a') + arc(0.34, 0.67, '#f5a623') + arc(0.67, 1, '#e5342d');
    s += '<line x1="' + cx + '" y1="' + cy + '" x2="' + n[0].toFixed(1) + '" y2="' + n[1].toFixed(1) + '" stroke="#14203d" stroke-width="7" stroke-linecap="round" style="transition:all 1s"/><circle cx="' + cx + '" cy="' + cy + '" r="12" fill="#14203d"/>';
    s += '<text x="20" y="165" font-size="14" font-weight="800" fill="#2b3a5e">Thấp</text><text x="236" y="165" font-size="14" font-weight="800" fill="#2b3a5e">Cao</text></svg>';
    return s;
  }

  function renderResult(data) {
    var sm = data.summary || {}, risk = sm.risk || {}, col = sm.level_color || '#e5342d', h = '';
    h += '<div class="card res-head"><h3>Kết quả hội chẩn</h3>' + gaugeSVG(risk) + '<div class="level-badge" style="background:' + esc(col) + '">Nguy cơ ' + esc(sm.level || '') + '</div><p>' + esc(sm.headline || '') + '</p>';
    h += '<div class="stack"><i style="width:' + pct(risk.none) + '%;background:#22b05a"></i><i style="width:' + pct(risk.pre) + '%;background:#f5a623"></i><i style="width:' + pct(risk.diabetes) + '%;background:#e5342d"></i></div>';
    h += '<div class="risk-rows">' + [['none', 'Không tiểu đường', '#22b05a'], ['pre', 'Tiền tiểu đường', '#f5a623'], ['diabetes', 'Tiểu đường', '#e5342d']].map(function (r) { return '<div class="risk-row"><span>' + r[1] + '</span><div class="risk-track"><div class="risk-fill" style="width:0;background:' + r[2] + '" data-w="' + pct(risk[r[0]]) + '"></div></div><span class="risk-pct">' + pct(risk[r[0]]) + '%</span></div>'; }).join('') + '</div></div>';
    if (sm.red_flags && sm.red_flags.length) h += '<div class="card"><h3>🚩 Cờ đỏ</h3><ul class="clean flags">' + sm.red_flags.map(function (x) { return '<li>' + esc(typeof x === 'string' ? x : x.text) + '</li>'; }).join('') + '</ul></div>';
    if (sm.advice && sm.advice.length) h += '<div class="card"><h3>💡 Lời khuyên</h3><ul class="clean">' + sm.advice.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul></div>';
    if (sm.citations && sm.citations.length) h += '<div class="card"><h3>📚 Trích dẫn tài liệu</h3>' + sm.citations.map(function (c) { return '<div class="cite"><span class="src">NIDDK</span><br>“' + esc(c.quote) + '”<br>' + (c.url ? '<a href="' + esc(c.url) + '" target="_blank" rel="noopener noreferrer">' + esc(c.url) + '</a>' : '') + '</div>'; }).join('') + '</div>';
    var rep = data.report || {};
    if ((rep.figures && rep.figures.length) || (rep.tables && rep.tables.length)) {
      h += '<div class="card"><h3>📄 Báo cáo tự sinh</h3>';
      (rep.figures || []).forEach(function (f) { h += '<figure class="fig"><b>' + esc(f.title) + '</b><img alt="' + esc(f.title) + '" src="data:image/png;base64,' + f.png_base64 + '"><figcaption>' + esc(f.caption) + '</figcaption></figure>'; });
      (rep.tables || []).forEach(function (t) { h += '<div class="tbl-title">' + esc(t.title) + '</div>' + tableHtml(t.columns, t.rows); });
      h += '<div class="btn-row"><button class="btn primary" id="btnDownload">⬇ Tải báo cáo (HTML)</button></div></div>';
    }
    h += '<div class="btn-row" style="margin-bottom:12px"><button class="btn" id="btnAgain">🔁 Khám lại</button><button class="btn accent" id="btnToLife">🎮 Thử sống 24 giờ</button></div>';
    $('#consultResult').innerHTML = h;
    setTimeout(function () { $$('.risk-fill').forEach(function (e) { e.style.width = e.dataset.w + '%'; }); }, 60);
    var d = $('#btnDownload'); if (d) d.onclick = function () { downloadReport(data); };
    $('#btnAgain').onclick = function () { S.playToken++; wakeAll(); S.playing = false; $('#consultRun').hidden = true; $('#consultForm').hidden = false; $('#hud').hidden = true; Clinic.hideAllBubbles(); Clinic.highlightKg(null); Clinic.patientReset(); Clinic.flyHome(900); $('#panel').scrollTop = 0; };
    $('#btnToLife').onclick = function () { switchMode('life'); };
  }
  function tableHtml(cols, rows) {
    return '<div class="tbl-wrap"><table class="data"><thead><tr>' + cols.map(function (c) { return '<th>' + esc(c) + '</th>'; }).join('') + '</tr></thead><tbody>' + rows.map(function (r) { return '<tr>' + r.map(function (c) { return '<td>' + esc(c) + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table></div>';
  }

  function downloadReport(data) {
    var sm = data.summary || {}, risk = sm.risk || {}, rep = data.report || {}, col = sm.level_color || '#e5342d', name = S.avName || 'Bệnh nhân';
    var css = 'body{font-family:"Segoe UI",Arial,sans-serif;background:#eef5ff;color:#14203d;margin:0;padding:24px}.wrap{max-width:860px;margin:0 auto}h1{margin:0 0 4px}.sub{color:#4a5a7c;margin-bottom:18px}.card{background:#fff;border-radius:16px;padding:18px 20px;margin-bottom:16px;box-shadow:0 4px 16px rgba(37,70,140,.14);border:2px solid #e3ecfa}.badge{display:inline-block;color:#fff;font-weight:800;border-radius:999px;padding:4px 18px;font-size:20px}.bar{display:flex;align-items:center;gap:10px;margin:6px 0;font-weight:700}.bar .t{flex:1;height:16px;background:#e8eefa;border-radius:99px;overflow:hidden}.bar .f{height:100%}li{margin:6px 0}img{max-width:100%;border:2px solid #dbe6f8;border-radius:12px}table{border-collapse:collapse;width:100%;margin:8px 0 14px}th{background:#2589e0;color:#fff;text-align:left;padding:6px 10px}td{padding:6px 10px;border-top:1px solid #e1e9f7}.disc{background:#fff8dc;border:2px solid #f2d77a;border-radius:12px;padding:10px 14px;font-weight:700}a{color:#0b7f72}';
    var h = '<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Báo cáo hội chẩn - ' + esc(name) + '</title><style>' + css + '</style></head><body><div class="wrap">';
    h += '<h1>Báo cáo hội chẩn - Phòng khám AI</h1><div class="sub">Bệnh nhân: <b>' + esc(name) + '</b> · Lập lúc ' + new Date().toLocaleString('vi-VN') + '</div>';
    h += '<div class="card"><span class="badge" style="background:' + esc(col) + '">Nguy cơ ' + esc(sm.level || '') + '</span><p><b>' + esc(sm.headline || '') + '</b></p>';
    [['Không tiểu đường', risk.none, '#22b05a'], ['Tiền tiểu đường', risk.pre, '#f5a623'], ['Tiểu đường', risk.diabetes, '#e5342d']].forEach(function (r) { h += '<div class="bar"><span style="width:150px">' + r[0] + '</span><div class="t"><div class="f" style="width:' + pct(r[1]) + '%;background:' + r[2] + '"></div></div><b>' + pct(r[1]) + '%</b></div>'; });
    h += '</div>';
    if (sm.red_flags && sm.red_flags.length) h += '<div class="card"><h2>Cờ đỏ</h2><ul>' + sm.red_flags.map(function (x) { return '<li>' + esc(typeof x === 'string' ? x : x.text) + '</li>'; }).join('') + '</ul></div>';
    if (sm.advice && sm.advice.length) h += '<div class="card"><h2>Lời khuyên</h2><ul>' + sm.advice.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul></div>';
    h += '<div class="card"><h2>Diễn biến hội chẩn</h2><ol>' + data.events.map(function (e) { return '<li><b style="color:' + agentColor(e.agent) + '">' + esc(agentName(e.agent)) + ':</b> ' + esc(e.text) + '</li>'; }).join('') + '</ol></div>';
    if ((rep.figures || []).length || (rep.tables || []).length) { h += '<div class="card"><h2>Hình và bảng</h2>'; (rep.figures || []).forEach(function (f) { h += '<h3>' + esc(f.title) + '</h3><img alt="' + esc(f.title) + '" src="data:image/png;base64,' + f.png_base64 + '"><p><i>' + esc(f.caption) + '</i></p>'; }); (rep.tables || []).forEach(function (t) { h += '<h3>' + esc(t.title) + '</h3>' + tableHtml(t.columns, t.rows); }); h += '</div>'; }
    if (sm.citations && sm.citations.length) h += '<div class="card"><h2>Tài liệu tham khảo (NIDDK)</h2><ul>' + sm.citations.map(function (c) { return '<li>“' + esc(c.quote) + '”' + (c.url ? '<br><a href="' + esc(c.url) + '">' + esc(c.url) + '</a>' : '') + '</li>'; }).join('') + '</ul></div>';
    h += '<div class="disc">Chỉ để học tập, không phải chẩn đoán y khoa. Hãy gặp bác sĩ nếu bạn lo lắng về sức khỏe.</div></div></body></html>';
    var blob = new Blob([h], { type: 'text/html;charset=utf-8' }), a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'bao-cao-hoi-chan.html'; document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  /* ---------- SỐNG 24 GIỜ ---------- */
  function hourLabel(h) { var x = Math.floor(h) % 24; return (x < 10 ? '0' : '') + x + ':00'; }
  function startLife() {
    var btn = $('#btnLifeStart'); hideBanner(); btn.disabled = true; btn.textContent = '⏳ Đang chuẩn bị...';
    var seed = Number($('#lifeSeed').value) || 7, profile = buildProfile();
    S.lastProfile = profile;
    api('/api/life/start', { profile: profile, seed: seed }).then(function (d) {
      btn.disabled = false; btn.textContent = '▶ Bắt đầu ngày mới';
      S.life = { seed: seed, profile: profile, state: d.state, actions: d.actions, advice: d.advice, patient: d.patient, series: [{ hour: d.state.hour, glucose: d.state.glucose }], busy: false, done: false, zone: zoneOf(d.state.glucose), last: null };
      $('#lifeIntro').hidden = true; $('#lifeEnd').hidden = true; $('#lifePlay').hidden = false; $('#lifeEvents').innerHTML = '';
      Clinic.patientAtSpot(); Clinic.setPatient({ name: S.avName || (d.patient && d.patient.name) || 'Bạn' });
      renderLifeActions(); renderLife(); Clinic.flyLife(1000);
    }, function (e) { btn.disabled = false; btn.textContent = '▶ Bắt đầu ngày mới'; banner('Không bắt đầu được ngày mới. ' + e.message, { retry: startLife, mock: true }); });
  }
  function actionById(id) { return (S.life.actions || []).filter(function (a) { return a.id === id; })[0] || { id: id, label: id, emoji: '❔' }; }
  function renderLifeActions() {
    var L = S.life, c = L.advice && L.advice.coach && L.advice.coach.action, r = L.advice && L.advice.rule && L.advice.rule.action;
    $('#actionGrid').innerHTML = L.actions.map(function (a) {
      var cls = 'act' + (a.id === c ? ' rec-coach' : '') + (a.id === r ? ' rec-rule' : '');
      return '<button class="' + cls + '" data-id="' + esc(a.id) + '" title="' + esc(a.desc || '') + '"><span class="rec">' + (a.id === c ? '🏅' : '') + (a.id === r ? '🛡️' : '') + '</span><span class="em">' + a.emoji + '</span><span class="lb">' + esc(a.label) + '</span></button>';
    }).join('');
    $$('#actionGrid .act').forEach(function (b) { b.onclick = function () { lifeStep(b.dataset.id); }; b.disabled = L.busy || L.done; });
  }
  function renderLife() {
    var L = S.life, st = L.state, z = ZONES[L.zone] || ZONES.ok;
    $('#lifeClock').textContent = hourLabel(st.hour); var g = $('#lifeGl'); g.textContent = Math.round(st.glucose); g.style.color = z.color;
    var zp = $('#lifeZonePill'); zp.textContent = z.label; zp.style.background = z.color;
    $('#zoneMarker').style.left = (clamp((st.glucose - 40) / 260, 0, 1) * 100) + '%';
    $('#lifeProg').style.width = clamp((st.hour - 6) / 24, 0, 1) * 100 + '%';
    var adv = L.advice || {}, cards = '';
    if (adv.coach) { var a = actionById(adv.coach.action); cards += '<div class="advice" style="--c:#f0603a"><span class="em">' + a.emoji + '</span><div><b>🏅 Huấn luyện viên (RL): ' + esc(a.label) + '</b><span>' + esc(adv.coach.why) + '</span></div></div>'; }
    if (adv.rule) { var b = actionById(adv.rule.action); cards += '<div class="advice" style="--c:#c98a0c"><span class="em">' + b.emoji + '</span><div><b>🛡️ Hộ lý (luật): ' + esc(b.label) + '</b><span>' + esc(adv.rule.why) + '</span></div></div>'; }
    $('#adviceCards').innerHTML = cards;
    Clinic.setGlucose({ value: st.glucose, zone: L.zone, hour: st.hour, series: L.series });
    drawLifeChart(); restoreLifeBubbles();
  }
  function restoreLifeBubbles() {
    if (S.mode !== 'life' || !S.life || S.life.done) return; var adv = S.life.advice || {};
    Clinic.hideAllBubbles();
    if (adv.coach) Clinic.say('coach', actionById(adv.coach.action).emoji + ' ' + adv.coach.why, { sticky: true, compact: true });
    if (adv.rule) Clinic.say('guardian', actionById(adv.rule.action).emoji + ' ' + adv.rule.why, { sticky: true, compact: true });
  }
  function chartBase(cv, xmin, xmax, ymin, ymax) {
    var ctx = cv.getContext('2d'), W = cv.width, H = cv.height, m = { l: 38, r: 10, t: 10, b: 24 };
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#f6faff'; ctx.fillRect(0, 0, W, H);
    var X = function (x) { return m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r); }, Y = function (y) { return H - m.b - (clamp(y, ymin, ymax) - ymin) / (ymax - ymin) * (H - m.t - m.b); };
    ctx.fillStyle = 'rgba(34,176,90,.22)'; ctx.fillRect(m.l, Y(180), W - m.l - m.r, Y(70) - Y(180));
    ctx.fillStyle = 'rgba(245,166,35,.14)'; ctx.fillRect(m.l, Y(250), W - m.l - m.r, Y(180) - Y(250));
    ctx.fillStyle = 'rgba(229,52,45,.12)'; ctx.fillRect(m.l, Y(ymax), W - m.l - m.r, Y(250) - Y(ymax));
    ctx.fillStyle = 'rgba(47,123,240,.12)'; ctx.fillRect(m.l, Y(70), W - m.l - m.r, Y(ymin) - Y(70));
    ctx.font = 'bold 13px "Segoe UI",Arial'; ctx.fillStyle = '#2b3a5e'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [70, 180, 250].forEach(function (v) { ctx.strokeStyle = 'rgba(20,32,61,.25)'; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(W - m.r, Y(v)); ctx.stroke(); ctx.setLineDash([]); ctx.fillText(v, m.l - 5, Y(v)); });
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    return { ctx: ctx, X: X, Y: Y, W: W, H: H, m: m };
  }
  function drawLifeChart() {
    var cv = $('#lifeChart'), c = chartBase(cv, 6, 30, 40, 300), ctx = c.ctx; ctx.fillStyle = '#2b3a5e';
    for (var h = 6; h <= 30; h += 4) ctx.fillText(hourLabel(h).slice(0, 2) + 'h', c.X(h), c.H - c.m.b + 5);
    var s = S.life.series; if (!s.length) return;
    ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.strokeStyle = '#2f5bd8'; ctx.beginPath(); s.forEach(function (p, i) { if (i) ctx.lineTo(c.X(p.hour), c.Y(p.glucose)); else ctx.moveTo(c.X(p.hour), c.Y(p.glucose)); }); ctx.stroke();
    s.forEach(function (p) { ctx.fillStyle = ZONES[zoneOf(p.glucose)].color; ctx.beginPath(); ctx.arc(c.X(p.hour), c.Y(p.glucose), 5, 0, 7); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); });
  }

  async function lifeStep(actionId) {
    var L = S.life; if (!L || L.busy || L.done) return; L.busy = true; hideBanner(); renderLifeActions();
    Clinic.hideBubble('coach'); Clinic.hideBubble('guardian');
    var act = actionById(actionId); Clinic.say('patient', act.emoji + ' ' + act.label, { compact: true });
    var anim = Clinic.patientAct(actionId, 1700), req = api('/api/life/step', { state: L.state, action: actionId, seed: L.seed });
    try {
      var res = await req; await anim; Clinic.hideBubble('patient');
      L.state = res.state; L.zone = res.zone || zoneOf(res.glucose_after); L.advice = res.advice; L.done = !!res.done; L.last = res;
      L.series.push({ hour: res.state.hour, glucose: res.glucose_after != null ? res.glucose_after : res.state.glucose });
      var ev = $('#lifeEvents'); ev.innerHTML = (res.events || []).map(function (e) { return '<div>' + esc(e) + '</div>'; }).join('') + '<div style="border-color:' + ZONES[L.zone].color + '">' + Math.round(res.glucose_before) + ' → <b>' + Math.round(res.glucose_after) + '</b> mg/dL</div>';
      L.busy = false; renderLife(); renderLifeActions();
      if (L.zone === 'danger') Clinic.alert('danger', 2200); else if (L.zone === 'low') Clinic.alert('info', 1800);
      if (L.done) return endLife();
      if ($('#autoPlay').checked && L.advice && L.advice.coach) setTimeout(function () { if ($('#autoPlay').checked && S.life === L && !L.busy && !L.done && S.mode === 'life') lifeStep(L.advice.coach.action); }, 600);
    } catch (e) {
      await anim; L.busy = false; Clinic.hideBubble('patient'); renderLifeActions(); $('#autoPlay').checked = false;
      banner('Lượt chơi bị lỗi. ' + e.message, { mock: true });
    }
  }

  function gradeOf(t) { return t >= 0.85 ? ['A', 'Xuất sắc', '#16a34a'] : t >= 0.7 ? ['B', 'Tốt', '#2589e0'] : t >= 0.5 ? ['C', 'Tạm được', '#e8a317'] : ['D', 'Cần cố gắng', '#e5342d']; }
  function endLife() {
    var L = S.life, sc = (L.last && L.last.score) || { time_in_range: 0, hypo_hours: 0, hyper_hours: 0 }, g = gradeOf(sc.time_in_range);
    Clinic.hideAllBubbles(); Clinic.say('coach', 'Hết ngày rồi! Bạn đạt xếp loại ' + g[0] + ' (' + g[1] + ').', { compact: true }); Clinic.speak('coach', true); setTimeout(function () { Clinic.speak(null, false); }, 2500);
    $('#lifePlay').hidden = true; var e = $('#lifeEnd'); e.hidden = false;
    e.innerHTML = '<div class="card"><h3>🏁 Kết thúc ngày</h3><div class="score-row"><div class="grade" style="color:' + g[2] + '">' + g[0] + '</div><div><b style="font-size:18px">' + g[1] + '</b><br><span class="muted">Thời gian trong vùng an toàn (time-in-range)</span></div></div>'
      + '<div class="score-row"><div class="score-box"><b style="color:' + g[2] + '">' + pct(sc.time_in_range) + '%</b><small>Trong vùng an toàn</small></div><div class="score-box"><b style="color:#2f7bf0">' + fmt1(sc.hypo_hours) + '</b><small>Giờ hạ đường</small></div><div class="score-box"><b style="color:#e5342d">' + fmt1(sc.hyper_hours) + '</b><small>Giờ đường cao</small></div></div>'
      + '<canvas id="endChart" class="cmp-chart" width="560" height="200"></canvas>'
      + '<div class="btn-row"><button class="btn primary" id="btnCompare">📊 So sánh huấn luyện viên</button><button class="btn accent" id="btnReplayLife">🔄 Chơi lại (seed khác)</button></div></div><div id="cmpBox"></div>';
    drawEndChart();
    $('#btnCompare').onclick = compareCoaches; $('#btnReplayLife').onclick = function () { var sd = $('#lifeSeed'); sd.value = (Number(sd.value) || 7) + 1 + Math.floor(Math.random() * 20); $('#lifeEnd').hidden = true; $('#lifeIntro').hidden = false; startLife(); };
  }
  function drawEndChart() {
    var cv = $('#endChart'), c = chartBase(cv, 6, 30, 40, 300), ctx = c.ctx, s = S.life.series; ctx.fillStyle = '#2b3a5e';
    for (var h = 6; h <= 30; h += 4) ctx.fillText(hourLabel(h).slice(0, 2) + 'h', c.X(h), c.H - c.m.b + 5);
    ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.strokeStyle = '#2f5bd8'; ctx.beginPath(); s.forEach(function (p, i) { if (i) ctx.lineTo(c.X(p.hour), c.Y(p.glucose)); else ctx.moveTo(c.X(p.hour), c.Y(p.glucose)); }); ctx.stroke();
  }
  var COACH_COL = { rl: '#f0603a', rule: '#e8a317', none: '#64748b' };
  function compareCoaches() {
    var btn = $('#btnCompare'); btn.disabled = true; btn.textContent = '⏳ Đang mô phỏng 30 ngày...';
    api('/api/life/compare', { profile: S.life.profile, days: 30, seed: S.life.seed }).then(function (d) {
      btn.disabled = false; btn.textContent = '📊 So sánh huấn luyện viên';
      var h = '<div class="card"><h3>So sánh 3 người chơi (trung bình 30 ngày)</h3><canvas id="cmpChart" class="cmp-chart" width="560" height="260"></canvas><div class="legend" style="margin-bottom:10px">' + d.coaches.map(function (c) { return '<span><i style="background:' + (COACH_COL[c.id] || '#888') + '"></i>' + esc(c.name) + ' · ' + pct(c.time_in_range) + '%</span>'; }).join('') + '</div>';
      if (d.table) h += tableHtml(d.table.columns, d.table.rows);
      else h += tableHtml(['Người chơi', 'Trong vùng an toàn', 'Đường TB', 'Giờ hạ', 'Giờ cao'], d.coaches.map(function (c) { return [c.name, pct(c.time_in_range) + '%', Math.round(c.mean_glucose), fmt1(c.hypo_hours), fmt1(c.hyper_hours)]; }));
      $('#cmpBox').innerHTML = h + '</div>';
      var cv = $('#cmpChart'), c = chartBase(cv, 0, 23, 40, 300), ctx = c.ctx; ctx.fillStyle = '#2b3a5e';
      for (var i = 0; i < 24; i += 4) ctx.fillText(hourLabel(6 + i).slice(0, 2) + 'h', c.X(i), c.H - c.m.b + 5);
      d.coaches.slice().reverse().forEach(function (co) { ctx.strokeStyle = COACH_COL[co.id] || '#888'; ctx.lineWidth = co.id === 'rl' ? 5 : 3.5; ctx.lineJoin = 'round'; ctx.beginPath(); co.curve.forEach(function (v, i) { if (i) ctx.lineTo(c.X(i), c.Y(v)); else ctx.moveTo(c.X(i), c.Y(v)); }); ctx.stroke(); });
      $('#cmpBox').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, function (e) { btn.disabled = false; btn.textContent = '📊 So sánh huấn luyện viên'; banner('Không so sánh được. ' + e.message, { retry: compareCoaches, mock: true }); });
  }

  /* ---------- BẢN ĐỒ TRI THỨC ---------- */
  var kgPromise = null;
  function ensureKg() {
    if (S.kg) return Promise.resolve(S.kg);
    if (kgPromise) return kgPromise;
    kgPromise = api('/api/kg').then(function (d) { S.kg = d; Clinic.setKG(d); Clinic.onNodePick(showNode); buildLegend(); return d; }, function (e) { kgPromise = null; banner('Không tải được đồ thị tri thức. ' + e.message, { retry: ensureKg, mock: true }); });
    return kgPromise;
  }
  function buildLegend() {
    $('#legendItems').innerHTML = Object.keys(Clinic.TYPE_COLORS).map(function (k) { return '<span><i style="background:' + Clinic.TYPE_COLORS[k] + '"></i>' + Clinic.TYPE_LABELS[k] + '</span>'; }).join('');
    var sug = ['triệu chứng tiểu đường', 'huyết áp cao', 'ăn uống và vận động', 'biến chứng', 'xét nghiệm A1C'];
    $('#kgSuggest').innerHTML = sug.map(function (s) { return '<button type="button" class="chip">' + esc(s) + '</button>'; }).join('');
    $$('#kgSuggest .chip').forEach(function (c) { c.onclick = function () { $('#kgQ').value = c.textContent; kgSearch(c.textContent); }; });
  }
  function showNode(id) {
    var n = S.kg && S.kg.nodes.filter(function (x) { return x.id === id; })[0]; if (!n) return;
    var col = Clinic.TYPE_COLORS[n.type] || '#667', rel = S.kg.edges.filter(function (e) { return e.source === id || e.target === id; });
    var byId = {}; S.kg.nodes.forEach(function (x) { byId[x.id] = x; });
    var el = $('#kgNodeInfo'); el.hidden = false;
    el.innerHTML = '<div class="node-title"><i style="background:' + col + '"></i><h3 style="margin:0">' + esc(n.vi || n.label) + '</h3></div><span class="node-type" style="background:' + col + '">' + (Clinic.TYPE_LABELS[n.type] || n.type) + '</span> <span class="muted">' + esc(n.label) + '</span><p style="margin-top:8px">' + esc(n.desc || '') + '</p>'
      + (rel.length ? '<b>Quan hệ</b><ul class="rel-list">' + rel.map(function (e) { var out = e.source === id, o = byId[out ? e.target : e.source]; return o ? '<li data-id="' + esc(o.id) + '">' + (out ? '<b>' + esc(n.vi) + '</b> ' + Clinic.relVi(e.rel) + ' → ' + esc(o.vi) : esc(o.vi) + ' ' + Clinic.relVi(e.rel) + ' → <b>' + esc(n.vi) + '</b>') + '</li>' : ''; }).join('') + '</ul>' : '');
    $$('#kgNodeInfo li').forEach(function (li) { li.onclick = function () { Clinic.kgSelect(li.dataset.id); Clinic.kgFocus(li.dataset.id); showNode(li.dataset.id); }; });
    Clinic.kgSelect(id);
  }
  function kgSearch(q) {
    q = (q || '').trim(); if (!q) { Clinic.highlightKg(null); $('#kgSearchResult').innerHTML = ''; return; }
    var box = $('#kgSearchResult'); box.innerHTML = '<div class="card"><div class="spinner" style="width:26px;height:26px;border-width:4px"></div></div>';
    ensureKg().then(function () { return api('/api/kg/search', { q: q }); }).then(function (d) {
      Clinic.highlightKg(d.nodes || []); Clinic.kgAutoRotate(false);
      var byId = {}; (S.kg ? S.kg.nodes : []).forEach(function (x) { byId[x.id] = x; });
      var h = '<div class="card"><h3>Kết quả cho “' + esc(q) + '”</h3>';
      if (d.answer) h += '<div class="answer">' + esc(d.answer) + '</div>';
      if (d.nodes && d.nodes.length) h += '<div class="chips" style="margin:0 0 8px">' + d.nodes.map(function (id) { var n = byId[id]; return n ? '<button class="chip on" data-id="' + esc(id) + '" style="background:' + (Clinic.TYPE_COLORS[n.type] || '#667') + ';border-color:transparent">' + esc(n.vi) + '</button>' : ''; }).join('') + '</div>';
      else h += '<p class="muted">Không có nút nào khớp.</p>';
      (d.passages || []).forEach(function (p) { h += '<div class="cite"><span class="src">RAG · ' + (p.score != null ? Math.round(p.score * 100) + '%' : 'đoạn trích') + '</span><br>“' + esc(p.quote) + '”' + (p.url ? '<br><a href="' + esc(p.url) + '" target="_blank" rel="noopener noreferrer">' + esc(p.url) + '</a>' : '') + '</div>'; });
      box.innerHTML = h + '</div>';
      $$('#kgSearchResult .chip[data-id]').forEach(function (c) { c.onclick = function () { Clinic.kgFocus(c.dataset.id); showNode(c.dataset.id); }; });
    }, function (e) { box.innerHTML = ''; banner('Tìm kiếm thất bại. ' + e.message, { mock: true }); });
  }

  /* ---------- khởi động ---------- */
  function wire() {
    $$('.tab').forEach(function (t) { t.onclick = function () { switchMode(t.dataset.mode); }; });
    $('#btnConsult').onclick = startConsult; $('#btnLifeStart').onclick = startLife;
    $('#btnSkip').onclick = function () { skipConsult(false); };
    $('#btnPause').onclick = function () { setPaused(!S.paused); };
    $('#btnVoice').onclick = function () { Voice.on = !Voice.on; try { localStorage.setItem('clinic.voice', Voice.on ? 'on' : 'off'); } catch (e) {} if (!Voice.on) stopVoice(); syncVoiceButton(); };
    fetch('/api/tts/status').then(function (r) { return r.json(); }).then(function (j) { Voice.available = !!j.enabled; syncVoiceButton(); }, function () { syncVoiceButton(); });
    $('#btnFast').onclick = function () { S.fast = !S.fast; $('#btnFast').textContent = S.fast ? '⏩ Tốc độ x2,6' : '⏩ Tua nhanh'; $('#btnFast').style.background = S.fast ? '#ffe9a8' : ''; };
    $('#btnReplay').onclick = function () { if (S.consult) { S.playToken++; wakeAll(); stopVoice(); hideCaption(); Clinic.stopAll(); Clinic.hideAllBubbles(); playConsult(S.consult); } };
    $('#kgForm').onsubmit = function (e) { e.preventDefault(); kgSearch($('#kgQ').value); };
    $('#kgEdgeToggle').onchange = function (e) { Clinic.kgEdgeLabels(e.target.checked); };
    $('#helpBtn').onclick = function () { $('#helpModal').hidden = false; }; $('#helpClose').onclick = function () { $('#helpModal').hidden = true; };
    $('#helpModal').onclick = function (e) { if (e.target === this) this.hidden = true; };
    document.addEventListener('keydown', function (e) {
      var tg = e.target.tagName; if (tg === 'INPUT' || tg === 'SELECT' || tg === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
      var k = e.key.toLowerCase();
      if (k === '1') switchMode('consult'); else if (k === '2') switchMode('life'); else if (k === '3') switchMode('kg');
      else if (k === '?' ) $('#helpModal').hidden = false; else if (k === 'escape') $('#helpModal').hidden = true;
      else if (k === ' ' && S.playing) { e.preventDefault(); setPaused(!S.paused); } else if (k === 'v' && Voice.available) $('#btnVoice').click();
      else if (k === 's') skipConsult(false); else if (k === 'f' && !$('#hud').hidden) $('#btnFast').click(); else if (k === 'r' && !$('#hud').hidden && !$('#btnReplay').disabled) $('#btnReplay').click();
      else if (k === 'h') Clinic[S.mode === 'life' ? 'flyLife' : 'flyHome'](900);
      else if (k === 'enter') { if (S.mode === 'consult' && !$('#consultForm').hidden) startConsult(); else if (S.mode === 'life' && !$('#lifeIntro').hidden) startLife(); }
    });
  }

  function init() {
    if (MOCK) { $('#mockBadge').hidden = false; }
    wire();
    try { Clinic.init($('#stage'), $('#bubbles')); }
    catch (e) { S.glbWebgl = false; $('#webglError').hidden = false; console.warn('WebGL lỗi', e); return loadMetaOnly(); }
    setTimeout(function () { $('#stageLoading').classList.add('gone'); }, 500);
    return loadMeta();
  }
  function loadMetaOnly() { $('#stageLoading').classList.add('gone'); return loadMeta(); }
  function loadMeta() {
    return api('/api/meta').then(function (meta) {
      S.meta = meta; (meta.agents || []).forEach(function (a) { S.agents[a.id] = a; });
      if (S.glbWebgl) Clinic.setAgents(meta.agents);
      buildForm();
      if (meta.model) $('#footModel').textContent = 'Mô hình ' + meta.model.name + ' · huấn luyện trên ' + (meta.model.train_rows + meta.model.test_rows).toLocaleString('vi-VN') + ' hồ sơ · độ chính xác ' + pct(meta.model.accuracy) + '%';
      ensureKg();
    }, function (e) {
      $('#formGroups').innerHTML = '<p class="muted">Chưa tải được biểu mẫu.</p>';
      banner('Không tải được dữ liệu ban đầu. ' + e.message, { retry: loadMeta, mock: true });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
  window.__app = S;
})();
