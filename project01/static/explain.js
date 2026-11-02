/* Giải thích kết quả: yếu tố đẩy nguy cơ lên/xuống và "đổi một chỉ số thì nguy cơ đổi bao nhiêu" (what-if tương tác, có giọng đọc). */
(function () {
  'use strict';
  var CA = window.ClinicApp; if (!CA) return;
  var esc = CA.esc, pct = CA.pct;
  var KNOBS = [['smoker', 'Hút thuốc'], ['phys_activity', 'Có vận động'], ['fruits_daily', 'Ăn trái cây hằng ngày'], ['veggies_daily', 'Ăn rau hằng ngày'], ['heavy_alcohol', 'Uống nhiều rượu bia']];
  var tmr = null;

  function comma(v, d) { return (Math.round(v * Math.pow(10, d)) / Math.pow(10, d)).toString().replace('.', ','); }
  function pp(v) { return (v > 0 ? '+' : v < 0 ? '−' : '') + comma(Math.abs(v) * 100, 1) + ' điểm %'; }

  function driversHtml(drivers) {
    var mx = Math.max.apply(null, drivers.map(function (d) { return Math.abs(d.effect); }).concat([0.01]));
    return '<div class="ex-drv">' + drivers.map(function (d) {
      var w = Math.round(Math.abs(d.effect) / mx * 100), up = d.effect > 0;
      return '<div class="ex-row"><span class="ex-lab">' + esc(d.vi) + '</span><div class="ex-bar"><div class="ex-half l">' + (up ? '' : '<i class="dn" style="width:' + w + '%"></i>') + '</div><div class="ex-half r">' + (up ? '<i class="up" style="width:' + w + '%"></i>' : '') + '</div></div><b class="ex-val ' + (up ? 'up' : 'dn') + '">' + pp(d.effect) + '</b></div>';
    }).join('') + '</div><div class="ex-legend"><span class="dn">◀ kéo nguy cơ xuống</span><span class="up">đẩy nguy cơ lên ▶</span></div>';
  }

  function curveSvg(curve, now, after) {
    var pts = curve.points, W = 360, H = 150, L = 36, B = 24, T = 10, R = 10, xs = function (b) { return L + (b - 18) / 22 * (W - L - R); };
    var mx = Math.max.apply(null, pts.map(function (p) { return p.risk_index; })) * 1.1, ys = function (v) { return T + (1 - v / mx) * (H - T - B); };
    var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + xs(p.bmi).toFixed(1) + ' ' + ys(p.risk_index).toFixed(1); }).join(' ');
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="ex-curve" role="img" aria-label="Nguy cơ theo BMI">';
    [18.5, 25, 30].forEach(function (b, i) { s += '<line x1="' + xs(b) + '" x2="' + xs(b) + '" y1="' + T + '" y2="' + (H - B) + '" stroke="#cfe0f5" stroke-dasharray="3 3"/><text x="' + xs(b) + '" y="' + (H - 8) + '" font-size="10" text-anchor="middle" fill="#4a5a7c">' + ['18,5', '25', '30'][i] + '</text>'; });
    s += '<path d="' + d + '" fill="none" stroke="#7060d8" stroke-width="3"/>';
    s += '<circle cx="' + xs(now.bmi) + '" cy="' + ys(now.risk_index) + '" r="6" fill="#e8503a"/><text x="' + (xs(now.bmi) + 9) + '" y="' + (ys(now.risk_index) - 6) + '" font-size="11" font-weight="700" fill="#e8503a">hiện tại</text>';
    if (after && Math.abs(after.bmi - now.bmi) > 0.2) s += '<circle cx="' + xs(after.bmi) + '" cy="' + ys(after.risk_index) + '" r="6" fill="#0d9f8f"/><text x="' + (xs(after.bmi) + 9) + '" y="' + (ys(after.risk_index) + 14) + '" font-size="11" font-weight="700" fill="#0d9f8f">sau khi đổi</text>';
    s += '<text x="6" y="' + (T + 8) + '" font-size="10" fill="#4a5a7c">nguy cơ</text><text x="' + (W - R) + '" y="' + (H - 8) + '" font-size="10" text-anchor="end" fill="#4a5a7c">BMI</text></svg>';
    return s;
  }

  function build(data, host) {
    var pe = data.events.filter(function (e) { return e.agent === 'predictor' && e.payload && e.payload.drivers; })[0];
    var profile = CA.S.lastProfile; if (!pe || !profile) return;
    var drivers = pe.payload.drivers, whatIf = pe.payload.what_if || [];
    var wNow = Number(profile.weight_kg || 62), h = Number(profile.height_cm || 165) / 100;
    var card = document.createElement('div'); card.className = 'card explain'; card.id = 'explainCard';
    var up = drivers.filter(function (d) { return d.effect > 0.005; }), dn = drivers.filter(function (d) { return d.effect < -0.005; });
    var say = 'Yếu tố đẩy nguy cơ lên: ' + (up.length ? up.slice(0, 3).map(function (d) { return d.vi + ', khoảng ' + comma(d.effect * 100, 0) + ' điểm phần trăm'; }).join('; ') : 'không có yếu tố nổi bật') + '. Yếu tố kéo nguy cơ xuống: ' + (dn.length ? dn.slice(0, 2).map(function (d) { return d.vi; }).join(', ') : 'không có yếu tố nổi bật') + '.';
    if (whatIf.length) say += ' Nếu ' + whatIf[0].action + ', nguy cơ giảm khoảng ' + comma(whatIf[0].gain * 100, 0) + ' điểm phần trăm.';
    var html = '<h3>🔍 Vì sao ra kết quả này?</h3><p class="muted small">Mỗi thanh cho biết nguy cơ của bạn sẽ đổi bao nhiêu <b>điểm phần trăm</b> nếu yếu tố đó bằng mức trung bình của dân số. Đỏ là yếu tố đang đẩy nguy cơ lên, xanh là yếu tố đang kéo xuống.</p>' + driversHtml(drivers);
    html += '<div class="btn-row"><button class="btn sm" id="exSpeak">🔊 Nghe giải thích</button></div>';
    html += '<h3 class="ex-h">🎚️ Thử đổi một chỉ số</h3><p class="muted small">Kéo thanh hoặc bật/tắt thói quen, mô hình sẽ chạy lại ngay. Đường huyết không nằm trong mô hình nên không đổi ở đây.</p>';
    html += '<div class="ex-ctl"><label class="ex-sl"><span>Cân nặng <b id="exW">' + comma(wNow, 0) + ' kg</b> (BMI <b id="exB"></b>)</span><input type="range" id="exWeight" min="35" max="150" step="1" value="' + Math.round(wNow) + '"></label>';
    html += KNOBS.map(function (k) { return '<label class="ex-tg"><input type="checkbox" data-k="' + k[0] + '"' + (profile[k[0]] ? ' checked' : '') + '><span>' + k[1] + '</span></label>'; }).join('') + '</div>';
    html += '<div class="ex-out" id="exOut"><div class="ex-delta" id="exDelta">Chưa đổi gì: nguy cơ giữ nguyên.</div><div class="ex-cmp" id="exCmp"></div></div><div id="exCurve"></div>';
    html += '<p class="muted small">Đây là thay đổi theo mô hình học từ khảo sát, không phải lời hứa về hiệu quả với riêng bạn.</p>';
    card.innerHTML = html;
    var first = host.querySelector('.card.critique') || host.querySelector('.card'); host.insertBefore(card, first ? first.nextSibling : null);
    var curve = null;
    function setBmi() { var v = card.querySelector('#exWeight').value; card.querySelector('#exB').textContent = comma(Number(v) / (h * h), 1); card.querySelector('#exW').textContent = v + ' kg'; }
    function cmpBar(label, v, col) { return '<div class="ex-b"><span>' + label + '</span><div class="ex-bt"><i style="width:' + Math.round(v * 100) + '%;background:' + col + '"></i></div><b>' + pct(v) + '%</b></div>'; }
    function run() {
      setBmi();
      var ch = { weight_kg: Number(card.querySelector('#exWeight').value) };
      card.querySelectorAll('input[data-k]').forEach(function (c) { ch[c.dataset.k] = c.checked; });
      CA.api('/api/whatif', { profile: profile, changes: ch }).then(function (r) {
        var d = r.delta_index;
        card.querySelector('#exDelta').innerHTML = d < -0.0005 ? '<b class="dn">▼ Giảm ' + comma(Math.abs(d) * 100, 1) + ' điểm %</b> ' + esc(r.text) : d > 0.0005 ? '<b class="up">▲ Tăng ' + comma(d * 100, 1) + ' điểm %</b> ' + esc(r.text) : esc(r.text);
        card.querySelector('#exCmp').innerHTML = cmpBar('Chỉ số nguy cơ, hiện tại', r.base.risk_index, '#7060d8') + cmpBar('Chỉ số nguy cơ, sau khi đổi', r.new.risk_index, d <= 0 ? '#0d9f8f' : '#e8503a') + cmpBar('Xác suất tiểu đường, hiện tại', r.base.risk.diabetes, '#e8a317') + cmpBar('Xác suất tiểu đường, sau khi đổi', r.new.risk.diabetes, '#f0603a');
        card.dataset.text = r.text;
        if (curve) card.querySelector('#exCurve').innerHTML = curveSvg(curve, { bmi: r.base.bmi, risk_index: r.base.risk_index }, { bmi: r.new.bmi, risk_index: r.new.risk_index });
      }, function (e) { card.querySelector('#exDelta').textContent = 'Không tính được: ' + e.message; });
    }
    function schedule() { clearTimeout(tmr); tmr = setTimeout(run, 160); }
    card.querySelector('#exWeight').oninput = schedule; card.querySelectorAll('input[data-k]').forEach(function (c) { c.onchange = schedule; });
    card.querySelector('#exSpeak').onclick = function () { var t = say; if (card.dataset.text && /Nếu thay đổi/.test(card.dataset.text)) t += ' ' + card.dataset.text; CA.speakText(t, 'predictor'); CA.say('predictor', 'Mình đang giải thích các yếu tố tác động.', { compact: true }); };
    CA.api('/api/whatif/curve', { profile: profile }).then(function (c) { curve = c; run(); }, run);
    setBmi();
  }
  CA.onResult(build);
})();
