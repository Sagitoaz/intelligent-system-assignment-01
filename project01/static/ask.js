/* Chế độ "Hỏi bác sĩ": hỏi đáp tự do về bệnh tiểu đường, câu trả lời kèm trích dẫn từ kho NIDDK/CDC/MedlinePlus (POST /api/ask). */
(function () {
  'use strict';
  var A = window.ClinicApp, esc = A.esc, root = null, built = false, busy = false, seq = 0;
  var SAMPLES = [
    ['🩸', 'Tiểu đường là gì?'], ['😮‍💨', 'Triệu chứng của tiểu đường là gì?'], ['🧬', 'Nguyên nhân gây tiểu đường típ 2 là gì?'],
    ['🧪', 'Xét nghiệm A1C đo gì?'], ['🥗', 'Người tiểu đường nên chọn đồ uống nào?'], ['🏃', 'Mỗi tuần nên tập thể dục bao nhiêu phút?'],
    ['🦶', 'Tôi nên kiểm tra bàn chân bao lâu một lần?'], ['⚠️', 'Làm gì khi đường huyết xuống dưới 70 mg/dL?'], ['💊', 'Metformin hoạt động như thế nào?']
  ];

  function onEnter() {
    if (!root) root = document.getElementById('view-ask');
    if (!built) build();
    var inp = document.getElementById('akInput'); if (inp && !busy) setTimeout(function () { inp.focus(); }, 50);
  }

  function build() {
    built = true;
    root.innerHTML =
      '<div class="ak-wrap">' +
      '<div class="card hero ak-hero"><h2>💬 Hỏi bác sĩ về tiểu đường</h2>' +
      '<p>Gõ câu hỏi bằng tiếng Việt (hoặc tiếng Anh). Thủ thư tri thức tìm các đoạn tài liệu liên quan trong <b>492 đoạn</b> của NIDDK, CDC và MedlinePlus rồi trả lời <b>kèm trích dẫn [1], [2]</b> để bạn tự kiểm tra.</p>' +
      '<div class="ak-honest">⚕️ Đây là công cụ học tập: <b>không chẩn đoán</b>, không kê liều thuốc và không thay thế bác sĩ. Nếu không tìm thấy tài liệu đủ liên quan, agent sẽ từ chối thay vì đoán.</div></div>' +
      '<div class="card ak-samples"><h3>Thử một câu hỏi mẫu</h3><div class="ak-chips" id="akChips">' +
      SAMPLES.map(function (s, i) { return '<button class="ak-chip c' + (i % 6) + '" data-q="' + esc(s[1]) + '"><span>' + s[0] + '</span>' + esc(s[1]) + '</button>'; }).join('') + '</div></div>' +
      '<div class="ak-log" id="akLog" aria-live="polite"><div class="ak-empty" id="akEmpty">Chưa có câu hỏi nào. Chọn một câu mẫu ở trên hoặc tự gõ vào ô bên dưới.</div></div>' +
      '<form class="ak-form card" id="akForm" autocomplete="off">' +
      '<div class="ak-row"><input type="text" id="akInput" maxlength="300" placeholder="Ví dụ: Triệu chứng của tiểu đường là gì?" aria-label="Câu hỏi của bạn">' +
      '<button class="btn primary" id="akSend" type="submit">Hỏi ➤</button></div>' +
      '<div class="ak-opts"><label title="Một câu mở đầu dựa trên tuổi, BMI, huyết áp trong biểu mẫu Khám bệnh"><input type="checkbox" id="akProfile" checked> Nhắc đến hồ sơ của tôi</label>' +
      '<label title="Qwen2.5-1.5B chạy ngay trên máy này, lần đầu có thể mất vài chục giây; quá 60 giây sẽ tự chuyển về trích đoạn"><input type="checkbox" id="akQwen"> Cho Qwen viết lại câu trả lời (chậm hơn)</label></div></form>' +
      '</div>';
    document.getElementById('akChips').onclick = function (e) {
      var b = e.target.closest('.ak-chip'); if (!b) return; ask(b.getAttribute('data-q'));
    };
    document.getElementById('akForm').onsubmit = function (e) { e.preventDefault(); var i = document.getElementById('akInput'); ask(i.value); };
  }

  function log() { return document.getElementById('akLog'); }
  function scrollDown() { var l = log(); var p = document.getElementById('panel'); if (p) p.scrollTop = p.scrollHeight; if (l.lastChild && l.lastChild.scrollIntoView) l.lastChild.scrollIntoView({ block: 'nearest' }); }

  function ask(q) {
    q = String(q || '').trim();
    if (busy || q.length < 3) { if (!busy) flash('Hãy nhập câu hỏi dài hơn một chút.'); return; }
    busy = true; var my = ++seq;
    var empty = document.getElementById('akEmpty'); if (empty) empty.remove();
    var useQwen = document.getElementById('akQwen').checked, withProfile = document.getElementById('akProfile').checked;
    var L = log();
    L.insertAdjacentHTML('beforeend', '<div class="ak-msg me"><div class="ak-bub">' + esc(q) + '</div><span class="ak-av">🧑</span></div>');
    var pending = document.createElement('div'); pending.className = 'ak-msg bot';
    pending.innerHTML = '<span class="ak-av">📚</span><div class="ak-bub ak-wait"><span class="ak-dots"><i></i><i></i><i></i></span> <b>Đang tra cứu kho tri thức…</b>' +
      '<div class="ak-sub">' + (useQwen ? 'Qwen đang đọc các đoạn tài liệu và viết câu trả lời; lần đầu có thể mất tới khoảng 60 giây.' : 'Đang tìm các đoạn tài liệu liên quan nhất.') + '</div></div>';
    L.appendChild(pending); scrollDown();
    document.getElementById('akInput').value = ''; setBusy(true);
    A.say('knowledge', 'Để mình tra cứu tài liệu…', { compact: true });
    var body = { question: q, mode: useQwen ? 'auto' : 'retrieval' };
    if (withProfile) { try { body.profile = A.buildProfile(); } catch (e) {} }
    A.api('/api/ask', body).then(function (r) {
      if (my !== seq) return; pending.replaceWith(render(r)); finish(r);
    }, function (e) {
      pending.querySelector('.ak-bub').className = 'ak-bub ak-err';
      pending.querySelector('.ak-bub').innerHTML = '⚠️ Chưa hỏi được: ' + esc(e.message || 'lỗi không rõ') + ' <button class="btn sm" data-retry="1">Thử lại</button>';
      pending.querySelector('[data-retry]').onclick = function () { pending.remove(); busy = false; ask(q); };
      setBusy(false);
    });
  }

  function setBusy(v) {
    busy = v; var b = document.getElementById('akSend'); if (b) b.disabled = v;
    var i = document.getElementById('akInput'); if (i) i.disabled = v;
    if (!v && i) i.focus();
  }

  function flash(t) { A.banner && A.banner(t); }

  function finish(r) {
    setBusy(false); scrollDown();
    var s;
    if (r.refused) s = r.refused === 'emergency' ? 'Có thể là cấp cứu, hãy đến cơ sở y tế ngay.' : r.refused === 'personal' ? 'Câu này cần bác sĩ đánh giá riêng cho bạn.' : 'Mình chưa tìm thấy tài liệu đủ liên quan.';
    else s = 'Mình tìm thấy ' + r.citations.length + ' đoạn tài liệu liên quan, mời bạn xem trích dẫn.';
    A.say('knowledge', s, { compact: true, sticky: true });
  }

  function linkCites(text) {
    return esc(text).replace(/\[(\d)\]/g, function (m, n) { return '<a class="ak-ref" href="#" data-ref="' + n + '" title="Xem nguồn ' + n + '">' + n + '</a>'; });
  }

  function render(r) {
    var el = document.createElement('div'); el.className = 'ak-msg bot';
    var refused = !!r.refused;
    var modeTxt = r.mode === 'qwen' ? '🧠 Qwen viết lại' : '📑 Trích đoạn';
    var meta = '<div class="ak-meta"><span class="ak-badge ' + (r.mode === 'qwen' ? 'qw' : 'rt') + '">' + modeTxt + '</span><span class="ak-badge sec">⏱ ' + String(r.seconds).replace('.', ',') + ' giây</span>' +
      (refused ? '<span class="ak-badge no">Từ chối lịch sự</span>' : '') + '</div>';
    var cites = '';
    if (r.citations && r.citations.length) {
      cites = '<div class="ak-cites"><div class="ak-ct">Trích dẫn từ tài liệu gốc (tiếng Anh)</div>' + r.citations.map(function (c, i) {
        return '<details class="ak-cite" id="ak-c' + seq + '-' + (i + 1) + '"><summary><span class="ak-t"><b>[' + (i + 1) + ']</b> ' + esc(c.org || '') + ' · ' + esc(c.title || c.id) + '</span><span class="ak-sc">' + (c.score <= 1 ? 'độ giống ' + Math.round(c.score * 100) + '%' : 'điểm ' + c.score) + '</span></summary>' +
          '<blockquote>' + esc(c.quote) + '</blockquote><a href="' + esc(c.url) + '" target="_blank" rel="noopener">Mở trang nguồn ↗</a> <code>' + esc(c.id) + '</code></details>';
      }).join('') + '</div>';
    }
    var notes = (r.notes || []).map(function (n) { return '<div class="ak-note">ℹ️ ' + esc(n) + '</div>'; }).join('');
    el.innerHTML = '<span class="ak-av">📚</span><div class="ak-bub ' + (refused ? 'ak-refuse' : '') + '"><div class="ak-ans">' + linkCites(r.answer) + '</div>' + cites + meta + notes +
      '<div class="ak-act"><button class="btn sm" data-say="1" title="Đọc câu trả lời bằng giọng nói">🔊 Nghe</button></div></div>';
    el.querySelector('[data-say]').onclick = function () {
      var b = this; b.disabled = true; b.textContent = '🔊 Đang đọc…';
      Promise.resolve(A.speakText(r.speak || r.answer, 'knowledge')).then(function () { b.disabled = false; b.textContent = '🔊 Nghe'; }, function () { b.disabled = false; b.textContent = '🔊 Nghe'; });
    };
    var my = seq;
    el.querySelectorAll('.ak-ref').forEach(function (a) {
      a.onclick = function (e) { e.preventDefault(); var d = el.querySelector('#ak-c' + my + '-' + a.getAttribute('data-ref')); if (d) { d.open = true; d.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } };
    });
    return el;
  }

  A.registerMode('ask', { onEnter: onEnter });
})();
