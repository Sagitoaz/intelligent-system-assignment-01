/* Tab "Hàng đợi" (nhiều bệnh nhân + xu hướng 4 tuần) và nút tải báo cáo PDF sau khi khám. */
(function () {
  'use strict';
  var App = window.ClinicApp;
  if (!App) return;
  var esc = App.esc;
  var Q = { list: null, order: 'orig', sel: null, follow: {}, week: 1, loading: false };
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var LV = { 'thấp': '#0d9f8f', 'trung bình': '#e8a317', 'cao': '#e8503a' };

  function root() { return $('#view-queue'); }

  function shell() {
    var r = root();
    if (r.dataset.ready) return;
    r.dataset.ready = '1';
    r.innerHTML =
      '<div class="card hero q-hero"><h2>👥 Hàng đợi phòng khám</h2>' +
      '<p class="muted">Phòng khám có nhiều người cùng chờ. <b>Agent phân loại</b> xem nhanh hồ sơ (không chạy cả buổi hội chẩn) rồi sắp ai nên khám trước. ' +
      'Bấm <b>Khám</b> để chuyển sang tab khám bệnh với hồ sơ đó.</p>' +
      '<div class="btn-row"><button class="btn primary" id="qPrio">🧭 Agent phân loại: xếp theo mức khẩn</button><button class="btn" id="qOrig" hidden>↩ Về thứ tự đến khám</button></div>' +
      '<div id="qRule" class="q-rule" hidden></div></div>' +
      '<div id="qList" class="q-list"></div>' +
      '<div class="card q-follow" id="qFollow" hidden></div>' +
      '<p class="q-foot">Chỉ để học tập, không phải chẩn đoán y khoa.</p>';
    $('#qPrio').onclick = function () { Q.order = 'prio'; renderList(); };
    $('#qOrig').onclick = function () { Q.order = 'orig'; renderList(); };
  }

  function origOrder(list) {
    var meta = App.meta && App.meta(), ids = meta && meta.personas ? meta.personas.map(function (p) { return p.id; }) : [];
    return list.slice().sort(function (a, b) {
      var i = ids.indexOf(a.id), j = ids.indexOf(b.id);
      return (i < 0 ? 99 : i) - (j < 0 ? 99 : j);
    });
  }

  function load() {
    shell();
    if (Q.list || Q.loading) { renderList(); return; }
    Q.loading = true;
    $('#qList').innerHTML = '<div class="card muted">Đang gọi agent phân loại…</div>';
    App.api('/api/queue', {}).then(function (d) {
      Q.list = d.queue; Q.rule = d.rule; Q.loading = false;
      renderList();
      if (!Q.sel && d.queue.length) selectPatient(origOrder(d.queue)[0].id);
    }).catch(function (e) {
      Q.loading = false;
      $('#qList').innerHTML = '<div class="banner">' + esc(e.message || 'Không tải được hàng đợi') + '</div>';
    });
  }

  function renderList() {
    if (!Q.list) return;
    var items = Q.order === 'prio' ? Q.list.slice().sort(function (a, b) { return a.rank - b.rank; }) : origOrder(Q.list);
    $('#qPrio').hidden = Q.order === 'prio';
    $('#qOrig').hidden = Q.order !== 'prio';
    var rule = $('#qRule');
    rule.hidden = Q.order !== 'prio';
    rule.textContent = Q.rule || '';
    $('#qList').innerHTML = items.map(function (p, i) {
      var col = p.level_color || LV[p.level] || '#7f95ad';
      var flags = p.red_flags.length ? p.red_flags.map(function (f) {
        return '<span class="q-flag ' + (f.level === 'danger' ? 'danger' : 'warn') + '" title="' + esc(f.text) + '">' + (f.level === 'danger' ? '🚩 ' : '⚠️ ') + esc(f.id) + '</span>';
      }).join('') : '<span class="q-flag none">Không có cờ đỏ</span>';
      var num = Q.order === 'prio' ? '<span class="q-rank" style="background:' + col + '">' + p.rank + '</span>' : '<span class="q-rank q-wait">' + (i + 1) + '</span>';
      var mini = ['none', 'pre', 'diabetes'].map(function (k, j) {
        var c = ['#0d9f8f', '#e8a317', '#e8503a'][j], v = p.risk[k] * 100;
        return '<i title="' + ['Không tiểu đường', 'Tiền tiểu đường', 'Tiểu đường'][j] + ' ' + v.toFixed(0) + '%" style="flex:' + Math.max(v, 1.5) + ';background:' + c + '"></i>';
      }).join('');
      return '<div class="q-card' + (Q.sel === p.id ? ' sel' : '') + (p.danger ? ' danger' : '') + '" data-id="' + esc(p.id) + '" style="--lc:' + col + '">' +
        num + '<div class="q-main"><div class="q-top"><b>' + esc(p.name) + '</b><span class="q-lv" style="background:' + col + '">Nguy cơ ' + esc(p.level) + '</span></div>' +
        '<div class="q-story">' + esc(p.story) + '</div>' +
        '<div class="q-mini" aria-label="Ba xác suất: không, tiền, tiểu đường">' + mini + '</div>' +
        '<div class="q-meta">Chỉ số nguy cơ <b>' + (p.risk_index * 100).toFixed(0) + '%</b> · BMI ' + p.bmi + (p.fasting_glucose ? ' · đường huyết đói ' + Math.round(p.fasting_glucose) + ' mg/dL' : '') + '</div>' +
        '<div class="q-flags">' + flags + '</div>' +
        (Q.order === 'prio' ? '<div class="q-why">🧭 ' + esc(p.reason) + '</div>' : '') +
        '<div class="btn-row"><button class="btn primary sm" data-act="see">🩺 Khám</button><button class="btn sm" data-act="trend">📈 Xu hướng 4 tuần</button></div></div></div>';
    }).join('');
    Array.prototype.forEach.call(document.querySelectorAll('#qList .q-card'), function (el) {
      var p = Q.list.filter(function (x) { return x.id === el.dataset.id; })[0];
      el.onclick = function (ev) {
        var b = ev.target.closest && ev.target.closest('button');
        if (b && b.dataset.act === 'see') { App.startConsultWith(p.profile); return; }
        selectPatient(p.id);
      };
    });
  }

  /* ---------- xu hướng 4 tuần ---------- */
  function selectPatient(id) {
    Q.sel = id; Q.week = 1;
    Array.prototype.forEach.call(document.querySelectorAll('#qList .q-card'), function (el) { el.classList.toggle('sel', el.dataset.id === id); });
    var f = $('#qFollow'); f.hidden = false;
    if (Q.follow[id]) { renderFollow(); return; }
    f.innerHTML = '<span class="muted">Đang phân tích 4 tuần và hỏi huấn luyện viên RL…</span>';
    var p = Q.list.filter(function (x) { return x.id === id; })[0];
    App.api('/api/followup', { profile: p.profile }).then(function (d) {
      Q.follow[id] = d; if (Q.sel === id) renderFollow();
    }).catch(function (e) { f.innerHTML = '<div class="banner">' + esc(e.message || 'Không phân tích được') + '</div>'; });
  }

  function chart(d, week) {
    var W = 420, H = 250, L = 38, R = 38, T = 14, B = 34, pw = W - L - R, ph = H - T - B;
    var g = d.series.glucose, w = d.series.weight, n = g.length;
    var gmin = Math.min(60, Math.min.apply(null, g) - 8), gmax = Math.max(140, Math.max.apply(null, g) + 8);
    var wmin = Math.min.apply(null, w) - 1.5, wmax = Math.max.apply(null, w) + 1.5;
    var x = function (i) { return L + pw * i / (n - 1); };
    var yg = function (v) { return T + ph * (1 - (v - gmin) / (gmax - gmin)); };
    var yw = function (v) { return T + ph * (1 - (v - wmin) / (wmax - wmin)); };
    var band = function (a, b, col) { var ya = yg(Math.min(b, gmax)), yb = yg(Math.max(a, gmin)); return b <= gmin || a >= gmax ? '' : '<rect x="' + L + '" y="' + ya + '" width="' + pw + '" height="' + (yb - ya) + '" fill="' + col + '"/>'; };
    var path = function (v, f) { return v.map(function (a, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + f(a).toFixed(1); }).join(''); };
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="q-svg" role="img" aria-label="Biểu đồ đường huyết đói và cân nặng 4 tuần (dữ liệu mẫu minh họa)">';
    s += band(gmin, 100, '#d9f7ee') + band(100, 126, '#fff1c9') + band(126, gmax, '#ffe0dc');
    var wx0 = x(Math.round(n * (week - 1) / 4)), wx1 = x(Math.min(n - 1, Math.round(n * week / 4) - 1));
    s += '<rect x="' + wx0 + '" y="' + T + '" width="' + Math.max(4, wx1 - wx0) + '" height="' + ph + '" fill="#2589e0" opacity=".13" stroke="#2589e0" stroke-dasharray="4 3"/>';
    [100, 126].forEach(function (t) { if (t > gmin && t < gmax) s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + yg(t) + '" y2="' + yg(t) + '" stroke="' + (t === 100 ? '#b57a00' : '#c0392b') + '" stroke-dasharray="5 4" stroke-width="1.3"/><text x="' + (L - 4) + '" y="' + (yg(t) + 4) + '" text-anchor="end" font-size="11" font-weight="800" fill="' + (t === 100 ? '#b57a00' : '#c0392b') + '">' + t + '</text>'; });
    s += '<path d="' + path(w, yw) + '" fill="none" stroke="#7060d8" stroke-width="2.4" stroke-dasharray="6 3"/>';
    s += '<path d="' + path(g, yg) + '" fill="none" stroke="#e8503a" stroke-width="2.8" stroke-linejoin="round"/>';
    g.forEach(function (v, i) { s += '<circle cx="' + x(i) + '" cy="' + yg(v) + '" r="2.3" fill="#e8503a"><title>Ngày ' + (i + 1) + ': ' + v + ' mg/dL</title></circle>'; });
    s += '<text x="' + L + '" y="' + (T - 3) + '" font-size="10.5" font-weight="800" fill="#e8503a">mg/dL</text><text x="' + (W - R) + '" y="' + (T - 3) + '" text-anchor="end" font-size="10.5" font-weight="800" fill="#7060d8">kg</text>';
    s += '<text x="' + (W - R + 4) + '" y="' + (yw(w[n - 1]) + 4) + '" font-size="11" font-weight="800" fill="#7060d8">' + w[n - 1].toFixed(1) + '</text>';
    s += '<text x="' + (L - 4) + '" y="' + (yg(g[n - 1]) - 6) + '" text-anchor="end" font-size="11" font-weight="800" fill="#e8503a">' + Math.round(g[n - 1]) + '</text>';
    for (var k = 1; k <= 4; k++) s += '<text x="' + x(Math.round(n * (k - 0.5) / 4)) + '" y="' + (H - 16) + '" text-anchor="middle" font-size="12" font-weight="800" fill="' + (k === week ? '#1b5fb0' : '#4a5a7c') + '">Tuần ' + k + '</text>';
    s += '<text x="' + (L + pw / 2) + '" y="' + (H - 3) + '" text-anchor="middle" font-size="10.5" fill="#4a5a7c">28 ngày liên tiếp (mẫu mô phỏng)</text></svg>';
    return s;
  }

  function planChips(plan) {
    if (!plan.length) return '<span class="muted">Không có hành động đặc biệt: giữ thói quen hiện tại.</span>';
    return plan.map(function (p) { return '<span class="q-chip">' + p.hour + 'h ' + p.emoji + ' ' + esc(p.label) + '</span>'; }).join('');
  }

  function renderFollow() {
    var d = Q.follow[Q.sel], f = $('#qFollow'), wk = d.weeks[Q.week - 1], st = d.stats;
    var changes = wk.changes.length ? '<ul class="q-chg">' + wk.changes.map(function (c) { return '<li class="' + (c.kind === 'đổi' ? 'chg' : c.kind === 'thêm' ? 'add' : 'del') + '"><b>' + esc(c.kind.toUpperCase()) + '</b> ' + esc(c.text) + '</li>'; }).join('') + '</ul>'
      : (Q.week === 1 ? '' : '<p class="muted"><b>Không có gợi ý nào đổi</b> so với tuần trước.</p>');
    f.innerHTML =
      '<h3>📈 Xu hướng 4 tuần của ' + esc(d.name) + '</h3>' +
      '<div class="q-sim">🧪 ' + esc(d.note) + '</div>' +
      chart(d, Q.week) +
      '<div class="q-legend"><span><i style="background:#e8503a"></i>Đường huyết đói (mg/dL)</span><span><i class="dash" style="background:#7060d8"></i>Cân nặng (kg)</span>' +
      '<span><i class="zone" style="background:#d9f7ee"></i>dưới 100 ổn</span><span><i class="zone" style="background:#fff1c9"></i>100–125 tiền tiểu đường</span><span><i class="zone" style="background:#ffe0dc"></i>từ 126 mức tiểu đường</span></div>' +
      '<div class="q-stats">' +
      stat('Độ dốc đường huyết', (st.glucose_slope_per_week > 0 ? '+' : '') + st.glucose_slope_per_week + ' mg/dL mỗi tuần', st.glucose_slope_per_week > 2 ? '#e8503a' : st.glucose_slope_per_week < -2 ? '#0d9f8f' : '#2589e0') +
      stat('Ngày từ 126', st.pct_ge_126 + '% số ngày', '#e8503a') + stat('Ngày từ 100', st.pct_ge_100 + '% số ngày', '#e8a317') +
      stat('Cân nặng sau 4 tuần', (st.weight_change_4w > 0 ? '+' : '') + st.weight_change_4w + ' kg', '#7060d8') + '</div>' +
      '<p class="q-comment">' + esc(d.comment) + ' <button class="btn sm" id="qSpeak">🔊 Đọc nhận xét</button></p>' +
      '<div class="q-slider"><label for="qWeek">Chọn tuần: <b id="qWeekLbl">Tuần ' + Q.week + '</b></label><input type="range" id="qWeek" min="1" max="4" step="1" value="' + Q.week + '"><div class="q-ticks"><span>1</span><span>2</span><span>3</span><span>4</span></div></div>' +
      '<div class="q-week"><div class="q-wk-h">Tuần ' + Q.week + ': đường huyết đói trung bình <b>' + wk.mean_glucose.toFixed(0) + ' mg/dL</b>, cân nặng <b>' + wk.mean_weight.toFixed(1) + ' kg</b></div>' +
      '<h4>🏃 Gợi ý của huấn luyện viên RL tuần này</h4><div class="q-chips">' + planChips(wk.plan) + '</div>' +
      '<p class="q-fine">Kế hoạch là hành động RL chọn nhiều nhất trong 15 ngày mô phỏng cho bệnh nhân ảo dựng từ số liệu tuần này. Mô phỏng đồ chơi, không phải lời khuyên y khoa.</p>' +
      (Q.week > 1 ? '<h4>🔄 Điều gì đổi so với tuần trước</h4>' + changes + '<ul class="q-why-l">' + wk.reasons.map(function (r) { return '<li>' + esc(r) + '</li>'; }).join('') + '</ul>' : '<p class="muted">Tuần đầu tiên: kéo thanh trượt sang tuần sau để xem gợi ý đổi thế nào.</p>') + '</div>' +
      '<p class="q-rln">' + esc(d.rl_note) + '</p>';
    $('#qWeek').oninput = function () { Q.week = +this.value; renderFollow(); var s = $('#qWeek'); if (s) s.focus(); };
    $('#qSpeak').onclick = function () { App.speakText(d.comment, 'coach'); };
  }
  function stat(label, val, col) { return '<div class="q-stat" style="--c:' + col + '"><span>' + esc(label) + '</span><b>' + esc(val) + '</b></div>'; }

  App.registerMode('queue', { onEnter: load });

  /* ---------- nút tải báo cáo PDF ---------- */
  App.onResult(function (data, el) {
    if (!el || !data || !data.summary) return;
    var old = el.querySelector('.q-pdf'); if (old) old.remove();
    var box = document.createElement('div');
    box.className = 'card q-pdf';
    box.innerHTML = '<h3>📄 Báo cáo cho bệnh nhân</h3><p class="muted">Tải một tệp PDF gọn gàng: mức nguy cơ, cờ đỏ, lời khuyên, hình, bảng và nguồn trích dẫn.</p>' +
      '<button class="btn primary" id="qPdfBtn">📄 Tải báo cáo PDF</button><span class="q-pdf-msg" id="qPdfMsg" role="status"></span>';
    el.appendChild(box);
    box.querySelector('#qPdfBtn').onclick = function () {
      var btn = this, msg = box.querySelector('#qPdfMsg');
      var name = '', m0 = /^Chào (.+?)!/.exec(((data.events || [])[0] || {}).text || ''); if (m0) name = m0[1];
      var rep = data.report || {};
      var body = { summary: data.summary, patient_name: name,
        report: { figures: (rep.figures || []).map(function (f) { return { title: f.title, caption: f.caption, png_base64: f.png_base64 }; }),
                  tables: (rep.tables || []).map(function (t) { return { title: t.title, columns: t.columns, rows: t.rows }; }) } };
      btn.disabled = true; msg.textContent = ' Đang lập PDF…';
      fetch('/api/report/pdf', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(function (r) {
        if (!r.ok) throw new Error('Máy chủ báo lỗi ' + r.status);
        return r.blob();
      }).then(function (b) {
        var a = document.createElement('a');
        a.href = URL.createObjectURL(b); a.download = 'bao-cao-kham.pdf';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
        msg.textContent = ' Đã tải xong.';
      }).catch(function (e) { msg.textContent = ' Không tải được: ' + (e.message || e); })
        .then(function () { btn.disabled = false; });
    };
  });
})();
