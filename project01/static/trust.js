/* Chế độ "Độ tin cậy": dựng nội dung vào #view-trust từ /api/trust (số liệu tính thật trên tập test). */
(function () {
  'use strict';
  var A = window.ClinicApp, esc = A.esc, root = null, D = null, loading = false;
  var metric = 'recall_diabetes', groupKey = 'age', calibKey = 'diabetes';
  var METRICS = {
    recall_diabetes: ['Phát hiện tiểu đường (recall)', 'Trong số người thật sự bị tiểu đường, bao nhiêu % được agent đoán đúng.'],
    alert_rate: ['Tỉ lệ bị cảnh báo', 'Bao nhiêu % người trong nhóm bị gắn nhãn «nguy cơ cao».'],
    accuracy: ['Độ chính xác', 'Bao nhiêu % dự đoán đúng trong nhóm.'],
    alert_recall_diabetes: ['Cảnh báo bắt được tiểu đường', 'Trong số người bị tiểu đường, bao nhiêu % bị cảnh báo nguy cơ cao.']
  };

  function vn(x, d) { return (x * 100).toFixed(d == null ? 1 : d).replace('.', ',') + '%'; }
  function num(n) { return Number(n).toLocaleString('vi-VN'); }

  function onEnter() {
    if (!root) root = document.getElementById('view-trust');
    if (D || loading) return;
    loading = true;
    root.innerHTML = '<div class="card tr-note">Đang tải số liệu độ tin cậy…</div>';
    A.api('/api/trust').then(function (d) {
      loading = false;
      if (d.error) { root.innerHTML = '<div class="card tr-bad">' + esc(d.error) + '</div>'; return; }
      D = d; render();
    }, function (e) {
      loading = false;
      root.innerHTML = '<div class="card tr-bad">Không tải được số liệu. ' + esc(e.message || '') + ' <button class="btn" id="trRetry">Thử lại</button></div>';
      var b = document.getElementById('trRetry'); if (b) b.onclick = onEnter;
    });
  }

  function render() {
    root.innerHTML = '<div class="tr-wrap">' + hero() + classes() + calibration() + alerts() + fairness() + weak() + '</div>';
    bind();
    drawCalib();
  }

  function hero() {
    var b = D.baseline_always_none;
    return '<div class="card hero tr-hero"><h3>Mô hình đáng tin đến đâu?</h3>' +
      '<p class="muted">Mọi con số dưới đây được tính lại trên <b>' + num(D.test_rows) + '</b> hồ sơ mà mô hình chưa từng thấy khi học (tập kiểm tra).</p>' +
      '<div class="tr-kpis"><div class="tr-kpi" style="--k:#2589e0"><b>' + vn(D.accuracy) + '</b><span>độ chính xác tổng</span></div>' +
      '<div class="tr-kpi" style="--k:#e8a317"><b>' + vn(b.accuracy) + '</b><span>nếu cứ đoán «ai cũng không bị bệnh»</span></div>' +
      '<div class="tr-kpi" style="--k:#e5342d"><b>' + vn(D.per_class[1].recall) + '</b><span>người tiền tiểu đường được phát hiện</span></div></div>' +
      '<div class="tr-warn"><b>Đừng bị con số tổng đánh lừa.</b> Một «bác sĩ» chỉ biết nói «bạn khỏe» vẫn đúng ' + vn(b.accuracy) + ' số hồ sơ, <i>cao hơn</i> mô hình (' + vn(D.accuracy) +
      '), nhưng không phát hiện được ai. Vì vậy mô hình được chỉnh để ưu tiên không bỏ sót người bệnh, đổi lại độ chính xác tổng thấp hơn và có nhiều báo động nhầm.</div></div>';
  }

  function bar(v, color) {
    return '<div class="tr-bar"><i style="width:' + Math.max(1, v * 100) + '%;background:' + color + '"></i><span>' + vn(v) + '</span></div>';
  }

  function classes() {
    var rows = D.per_class.map(function (c) {
      var w = c.id === 'pre';
      return '<tr class="' + (w ? 'tr-weakrow' : '') + '"><td><b>' + esc(c.name) + '</b></td><td>' + num(c.support) + '</td>' +
        '<td>' + bar(c.recall, w ? '#e5342d' : '#16a34a') + '</td><td>' + bar(c.precision, '#2589e0') + '</td><td>' + vn(c.f1) + '</td></tr>';
    }).join('');
    var cm = D.confusion_matrix;
    var cmh = '<table class="tr-cm"><tr><th></th>' + D.confusion_labels.map(function (l) { return '<th>Agent đoán<br>' + esc(l) + '</th>'; }).join('') + '</tr>' +
      cm.map(function (r, i) {
        var tot = r.reduce(function (a, b) { return a + b; }, 0);
        return '<tr><th>Thực tế<br>' + esc(D.confusion_labels[i]) + '</th>' + r.map(function (v, j) {
          var a = Math.min(0.55, v / tot * 0.6 + 0.06);
          return '<td style="background:' + (i === j ? 'rgba(22,163,74,' : 'rgba(229,52,45,') + a + ')"><b>' + num(v) + '</b><small>' + vn(v / tot, 0) + ' của hàng</small></td>';
        }).join('') + '</tr>';
      }).join('') + '</table>';
    return '<div class="card"><h3>Đúng sai theo từng nhóm bệnh</h3>' +
      '<p class="help">«Recall» = trong số người thật sự thuộc nhóm đó, mô hình tìm ra được bao nhiêu %. «Precision» = trong số người mô hình gọi là nhóm đó, bao nhiêu % là đúng.</p>' +
      '<div class="tbl-wrap"><table class="tr-tbl"><tr><th>Nhóm</th><th>Số người</th><th>Recall</th><th>Precision</th><th>F1</th></tr>' + rows + '</table></div>' +
      '<div class="tr-warn red"><b>' + esc(D.pre_warning) + '</b> Nhóm tiền tiểu đường chỉ có ' + num(D.class_counts_test.pre) + ' người trong tập kiểm tra, quá ít để mô hình học tốt; precision cũng chỉ ' + vn(D.per_class[1].precision) + '.</div>' +
      '<h4>Bảng nhầm lẫn</h4><p class="help">Xanh = đoán đúng, đỏ = đoán sai. Mỗi hàng là người thật; mỗi cột là điều agent đoán.</p><div class="tbl-wrap">' + cmh + '</div></div>';
  }

  function calibration() {
    return '<div class="card"><h3>Xác suất có khớp thực tế không? (hiệu chuẩn)</h3>' +
      '<p class="help">Nếu mô hình nói «30% nguy cơ», thì trong 100 người như vậy khoảng 30 người nên thật sự bị bệnh. Điểm càng sát đường chéo xanh càng đáng tin. Điểm nằm trên đường chéo: mô hình nói thấp hơn thực tế; nằm dưới: nói cao hơn.</p>' +
      '<div class="seg tr-seg" id="trCalibSeg"><button data-k="diabetes" class="on">Tiểu đường</button><button data-k="pre">Tiền tiểu đường</button></div>' +
      '<div id="trCalib"></div><div id="trCalibNote" class="tr-note"></div></div>';
  }

  function drawCalib() {
    var c = D.calibration[calibKey], W = 360, H = 300, L = 42, B = 38, T = 12, R = 14, pw = W - L - R, ph = H - T - B;
    var mx = 0; c.bins.forEach(function (b) { if (b.n) mx = Math.max(mx, b.mean_pred, b.observed); });
    mx = Math.min(1, Math.ceil((mx + 0.05) * 10) / 10);
    function X(v) { return L + v / mx * pw; }
    function Y(v) { return T + ph - v / mx * ph; }
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="tr-svg" role="img" aria-label="Biểu đồ hiệu chuẩn">';
    for (var t = 0; t <= 5; t++) {
      var v = mx * t / 5;
      s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" class="g"/><line y1="' + T + '" y2="' + (T + ph) + '" x1="' + X(v) + '" x2="' + X(v) + '" class="g"/>' +
        '<text x="' + (L - 6) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + Math.round(v * 100) + '%</text><text x="' + X(v) + '" y="' + (H - B + 16) + '" text-anchor="middle">' + Math.round(v * 100) + '%</text>';
    }
    s += '<line x1="' + X(0) + '" y1="' + Y(0) + '" x2="' + X(mx) + '" y2="' + Y(mx) + '" class="diag"/>';
    var pts = c.bins.filter(function (b) { return b.n; });
    s += '<polyline fill="none" stroke="#7060d8" stroke-width="2.5" points="' + pts.map(function (b) { return X(b.mean_pred) + ',' + Y(b.observed); }).join(' ') + '"/>';
    pts.forEach(function (b) {
      var r = 4 + Math.min(9, Math.log10(b.n + 1) * 2.2);
      s += '<circle cx="' + X(b.mean_pred) + '" cy="' + Y(b.observed) + '" r="' + r + '" fill="' + (b.observed > b.mean_pred ? '#e8a317' : '#2589e0') + '" fill-opacity=".8" stroke="#14203d"><title>Mô hình nói ' + vn(b.mean_pred) + ', thực tế ' + vn(b.observed) + ' (' + num(b.n) + ' người)</title></circle>';
    });
    s += '<text x="' + (L + pw / 2) + '" y="' + (H - 6) + '" text-anchor="middle" class="ax">Xác suất mô hình đưa ra</text><text transform="translate(11 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle" class="ax">Tỉ lệ thật sự bị bệnh</text></svg>';
    document.getElementById('trCalib').innerHTML = s;
    var name = calibKey === 'pre' ? 'tiền tiểu đường' : 'tiểu đường';
    document.getElementById('trCalibNote').innerHTML = 'Sai số hiệu chuẩn <b>ECE = ' + vn(c.ece) + '</b> (càng gần 0 càng tốt) · điểm Brier = <b>' + String(c.brier).replace('.', ',') + '</b> · tỉ lệ ' + name + ' thật trong tập kiểm tra là ' + vn(c.base_rate) +
      '. Chấm vàng: thực tế cao hơn lời mô hình; chấm xanh: thực tế thấp hơn. Chấm to = nhiều người. ' +
      (c.ece > 0.05 ? '<b>Xác suất của mô hình chưa chuẩn</b>: vì cố ý ưu tiên không bỏ sót (trọng số lớp), nó thường nói cao hơn thực tế, nên hãy xem đó là «mức nguy cơ tương đối», không phải xác suất thật.' : 'Xác suất khá sát thực tế.');
  }

  function alerts() {
    var m = D.alerts.main, l = D.alerts.loose;
    function card(a, title, col) {
      return '<div class="tr-acard" style="--k:' + col + '"><h4>' + title + '</h4><div class="tr-big">' + vn(a.precision_diabetes.value) + '</div><div class="tr-cap">cảnh báo đúng (thật sự bị tiểu đường)</div>' +
        '<ul><li>Bắt được <b>' + vn(a.recall_diabetes.value, 0) + '</b> người tiểu đường (khoảng tin cậy ' + vn(a.recall_diabetes.ci[0], 0) + ' đến ' + vn(a.recall_diabetes.ci[1], 0) + ')</li>' +
        '<li>Báo nhầm cho <b>' + vn(a.false_alarm_rate.value, 0) + '</b> người hoàn toàn không bệnh</li>' +
        '<li>Cứ 100 lần cảnh báo, chỉ <b>' + Math.round(a.precision_diabetes.value * 100) + '</b> lần là tiểu đường; <b>' + Math.round(a.false_alerts_among_alerts.value * 100) + '</b> lần là người không bệnh</li>' +
        '<li>Tính cả tiền tiểu đường: đúng ' + vn(a.precision_any.value, 0) + ', bắt được ' + vn(a.recall_pre.value, 0) + ' người tiền tiểu đường</li>' +
        '<li>' + num(a.alerts) + ' / ' + num(D.test_rows) + ' người bị cảnh báo (' + vn(a.alert_rate, 0) + ')</li></ul></div>';
    }
    return '<div class="card"><h3>Khi agent báo «nguy cơ cao» thì đúng bao nhiêu?</h3>' +
      '<p class="help">«Nguy cơ cao» nghĩa là chỉ số nguy cơ (xác suất tiểu đường + nửa xác suất tiền tiểu đường) từ 0,25 trở lên. Đây là mức cảnh báo phòng khám đang dùng.</p>' +
      '<div class="tr-acards">' + card(m, 'Ngưỡng đang dùng: 0,25', '#e8a317') + card(l, 'Thử ngưỡng nhẹ hơn: 0,10', '#7060d8') + '</div>' +
      '<div class="tr-warn"><b>Đánh đổi:</b> hạ ngưỡng thì bắt thêm người bệnh (' + vn(m.recall_diabetes.value, 0) + ' lên ' + vn(l.recall_diabetes.value, 0) + ') nhưng cảnh báo đúng giảm (' + vn(m.precision_diabetes.value, 0) + ' xuống ' + vn(l.precision_diabetes.value, 0) +
      ') và nhiều người khỏe bị báo nhầm hơn. Cảnh báo chỉ là lời nhắc đi xét nghiệm, không phải kết luận bị bệnh.</div></div>';
  }

  function fairness() {
    return '<div class="card"><h3>Kiểm tra công bằng</h3><p class="help">Mô hình có đối xử như nhau với mọi nhóm người không? Mỗi thanh là một nhóm; vạch đen là khoảng tin cậy 95% (khoảng mà con số thật nhiều khả năng nằm trong đó; nhóm ít người thì khoảng rộng).</p>' +
      '<div class="seg tr-seg" id="trGroupSeg">' + ['age', 'sex', 'income'].map(function (k) { return '<button data-k="' + k + '"' + (k === groupKey ? ' class="on"' : '') + '>' + esc(D.groups[k].title) + '</button>'; }).join('') + '</div>' +
      '<div class="seg tr-seg" id="trMetricSeg">' + Object.keys(METRICS).map(function (k) { return '<button data-k="' + k + '"' + (k === metric ? ' class="on"' : '') + '>' + METRICS[k][0] + '</button>'; }).join('') + '</div>' +
      '<div id="trFair"></div>' +
      '<h4>Kết luận (tính từ số liệu thật)</h4><ul class="tr-list">' + D.fairness.conclusions.map(function (c) { return '<li>' + esc(c) + '</li>'; }).join('') + '</ul></div>';
  }

  function drawFair() {
    var g = D.groups[groupKey], gap = g.gaps[metric];
    var h = '<p class="help">' + esc(METRICS[metric][1]) + '</p>';
    g.rows.forEach(function (r) {
      var m = r[metric], v = m.value || 0, ci = m.ci || [0, 0];
      var low = r.group === gap.lowest && metric !== 'alert_rate';
      var col = metric === 'alert_rate' ? '#7060d8' : low ? '#e5342d' : '#0d9f8f';
      h += '<div class="tr-grow"><div class="tr-gname">' + esc(r.group) + '<small>' + num(r.n) + ' người, ' + num(r.diabetes_n) + ' bị tiểu đường</small></div>' +
        '<div class="tr-gtrack"><i style="width:' + v * 100 + '%;background:' + col + '"></i><u style="left:' + ci[0] * 100 + '%;width:' + Math.max(0.5, (ci[1] - ci[0]) * 100) + '%"></u></div>' +
        '<div class="tr-gval"><b>' + vn(v) + '</b><small>' + vn(ci[0], 0) + '–' + vn(ci[1], 0) + '</small></div></div>';
    });
    h += '<div class="tr-gap">Chênh lệch lớn nhất: <b>' + (gap.max_gap * 100).toFixed(1).replace('.', ',') + ' điểm phần trăm</b> (cao nhất «' + esc(gap.highest) + '», thấp nhất «' + esc(gap.lowest) + '»)</div>';
    document.getElementById('trFair').innerHTML = h;
  }

  function weak() {
    return '<div class="card tr-weakbox"><h3>Điểm yếu cần biết</h3><ul class="tr-list">' + D.limitations.map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('') + '</ul>' +
      '<p class="help">Kết luận: dùng agent như công cụ gợi ý đi khám sớm; chỉ xét nghiệm đường huyết mới xác nhận được bệnh.</p></div>';
  }

  function seg(id, fn) {
    var el = document.getElementById(id);
    el.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      Array.prototype.forEach.call(el.children, function (x) { x.classList.toggle('on', x === b); });
      fn(b.dataset.k);
    });
  }
  function bind() {
    seg('trCalibSeg', function (k) { calibKey = k; drawCalib(); });
    seg('trGroupSeg', function (k) { groupKey = k; drawFair(); });
    seg('trMetricSeg', function (k) { metric = k; drawFair(); });
    drawFair();
  }

  A.registerMode('trust', { onEnter: onEnter });
})();
