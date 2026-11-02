/* Nhập hồ sơ bằng giọng nói: ghi âm trong trình duyệt, tự mã hóa WAV 16 kHz mono 16-bit, gửi /api/stt (Azure Speech-to-Text) rồi điền biểu mẫu.
   Có ô "đọc thử câu" (POST /api/stt/parse, chạy bộ phân tích luật, không cần micro hay Azure). */
(function () {
  'use strict';
  var A = window.ClinicApp, esc = A.esc;
  var MAX_SEC = 12, RATE = 16000;
  var card, st = { rec: null, timer: null, t0: 0, enabled: null };
  var EXAMPLES = [
    'Tôi là nam, 55 tuổi, cao một mét sáu lăm, nặng 70 ký, đường huyết đói 190, bị huyết áp cao và hay khát nước',
    'Chị 34 tuổi, cao 162 cm, nặng 54 ký, không hút thuốc, hay tập thể dục',
    'Ông sáu mươi bảy tuổi, mỡ máu cao, ít vận động, nhìn mờ, sụt cân, mệt mỏi'
  ];

  function build() {
    var form = document.getElementById('consultForm'); if (!form || document.getElementById('vcCard')) return;
    card = document.createElement('div'); card.className = 'card vc-card'; card.id = 'vcCard';
    card.innerHTML =
      '<div class="vc-head"><div class="vc-title"><span class="vc-ico">🎤</span><div><h3>Nói để điền hồ sơ</h3>' +
      '<div class="vc-sub">Bấm micro rồi nói tự nhiên, ví dụ: <i>“Tôi nam 55 tuổi, cao 1m65, nặng 70 ký, đường huyết đói 190”</i>.</div></div></div>' +
      '<button class="btn vc-mic" id="vcMic" type="button" aria-pressed="false"><span class="vc-dot"></span><span id="vcMicTxt">Bấm để nói</span></button></div>' +
      '<div class="vc-bar" id="vcBar" hidden><i id="vcFill"></i></div>' +
      '<div class="vc-status" id="vcStatus" role="status" aria-live="polite"></div>' +
      '<div class="vc-text" id="vcText" hidden></div>' +
      '<div class="vc-res" id="vcRes" hidden></div>' +
      '<details class="vc-try"><summary>⌨️ Không có micro? Gõ một câu để “đọc thử”</summary>' +
      '<div class="vc-row"><input type="text" id="vcIn" maxlength="300" placeholder="Gõ câu bạn muốn agent hiểu…" aria-label="Câu đọc thử">' +
      '<button class="btn sm primary" id="vcGo" type="button">Đọc thử</button></div>' +
      '<div class="vc-ex">' + EXAMPLES.map(function (e, i) { return '<button type="button" class="vc-exb" data-i="' + i + '">Mẫu ' + (i + 1) + '</button>'; }).join('') + '</div>' +
      '<div class="vc-fine">Câu gõ chỉ đi qua bộ luật phân tích trên máy chủ, không gọi Azure.</div></details>' +
      '<div class="vc-fine">🔒 Âm thanh chỉ gửi đến máy chủ của demo này để nhận dạng, không lưu lại. Kết quả có thể nghe nhầm: hãy kiểm tra lại biểu mẫu bên dưới.</div>';
    form.insertBefore(card, form.firstChild);
    document.getElementById('vcMic').onclick = toggle;
    document.getElementById('vcGo').onclick = function () { tryText(document.getElementById('vcIn').value); };
    document.getElementById('vcIn').onkeydown = function (e) { if (e.key === 'Enter') { e.preventDefault(); tryText(this.value); } };
    card.querySelector('.vc-ex').onclick = function (e) { var b = e.target.closest('.vc-exb'); if (!b) return; var t = EXAMPLES[+b.getAttribute('data-i')]; document.getElementById('vcIn').value = t; tryText(t); };
    A.api('/api/stt/status').then(function (s) {
      st.enabled = !!s.enabled;
      if (!st.enabled) { setStatus('ℹ️ Chưa cấu hình dịch vụ nhận giọng nói Azure, nên micro chưa dùng được. Bạn vẫn thử được bằng ô gõ câu bên dưới.', 'warn'); document.getElementById('vcMic').disabled = true; card.querySelector('.vc-try').open = true; }
    }, function () { st.enabled = null; });
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('ℹ️ Trình duyệt này không cho dùng micro (cần mở bằng localhost hoặc HTTPS). Hãy dùng ô gõ câu bên dưới.', 'warn'); document.getElementById('vcMic').disabled = true; card.querySelector('.vc-try').open = true;
    }
  }

  function setStatus(t, kind) { var s = document.getElementById('vcStatus'); s.className = 'vc-status ' + (kind || ''); s.textContent = t || ''; }

  /* ---------- ghi âm ---------- */
  function toggle() { if (st.rec) stop(); else start(); }

  function start() {
    setStatus('Đang xin quyền dùng micro…');
    navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }).then(function (stream) {
      var AC = window.AudioContext || window.webkitAudioContext, ctx = new AC();
      var src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(4096, 1, 1), chunks = [];
      var mute = ctx.createGain(); mute.gain.value = 0;
      proc.onaudioprocess = function (e) { chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
      src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
      st.rec = { stream: stream, ctx: ctx, src: src, proc: proc, chunks: chunks, rate: ctx.sampleRate };
      st.t0 = Date.now(); document.getElementById('vcBar').hidden = false;
      var mic = document.getElementById('vcMic'); mic.classList.add('on'); mic.setAttribute('aria-pressed', 'true'); document.getElementById('vcMicTxt').textContent = 'Đang nghe… bấm để dừng';
      A.say && A.say('doctor', 'Mình đang nghe bạn nói…', { compact: true });
      st.timer = setInterval(tick, 100);
    }, function (err) {
      var n = err && err.name;
      setStatus(n === 'NotAllowedError' || n === 'SecurityError' ? '🔇 Bạn chưa cho phép dùng micro. Hãy bấm vào biểu tượng ổ khóa cạnh địa chỉ trang, cho phép micro rồi thử lại. Hoặc dùng ô gõ câu bên dưới.'
        : n === 'NotFoundError' ? '🎙️ Không tìm thấy micro trên máy. Hãy cắm micro hoặc dùng ô gõ câu bên dưới.' : '⚠️ Không mở được micro (' + (n || 'lỗi không rõ') + '). Hãy dùng ô gõ câu bên dưới.', 'warn');
      card.querySelector('.vc-try').open = true;
    });
  }

  function tick() {
    var s = (Date.now() - st.t0) / 1000, f = document.getElementById('vcFill');
    f.style.width = Math.min(100, s / MAX_SEC * 100) + '%';
    setStatus('🔴 Đang ghi âm ' + s.toFixed(1).replace('.', ',') + ' / ' + MAX_SEC + ' giây');
    if (s >= MAX_SEC) stop();
  }

  function stop() {
    var r = st.rec; if (!r) return; st.rec = null; clearInterval(st.timer);
    try { r.proc.disconnect(); r.src.disconnect(); } catch (e) {}
    r.stream.getTracks().forEach(function (t) { t.stop(); });
    try { r.ctx.close(); } catch (e) {}
    var mic = document.getElementById('vcMic'); mic.classList.remove('on'); mic.setAttribute('aria-pressed', 'false'); document.getElementById('vcMicTxt').textContent = 'Bấm để nói';
    document.getElementById('vcBar').hidden = true;
    var pcm = resample(concat(r.chunks), r.rate, RATE);
    if (pcm.length < RATE * 0.4) { setStatus('Ghi âm quá ngắn, hãy bấm micro và nói lại.', 'warn'); return; }
    var wav = toWav(pcm, RATE);
    setStatus('⏳ Đang nhận dạng giọng nói bằng Azure…'); mic.disabled = true;
    A.api('/api/stt', { audio_base64: b64(wav), language: 'vi-VN' }).then(function (res) { mic.disabled = false; show(res, 'Đã nghe xong.'); },
      function (e) { mic.disabled = false; setStatus('⚠️ ' + (e.message || 'Chưa nhận dạng được') + ' Bạn có thể thử lại hoặc dùng ô gõ câu.', 'warn'); });
  }

  function concat(chunks) {
    var n = 0, i; for (i = 0; i < chunks.length; i++) n += chunks[i].length;
    var out = new Float32Array(n), o = 0; for (i = 0; i < chunks.length; i++) { out.set(chunks[i], o); o += chunks[i].length; } return out;
  }
  function resample(buf, from, to) {
    if (from === to) return buf;
    var ratio = from / to, n = Math.floor(buf.length / ratio), out = new Float32Array(n);
    for (var i = 0; i < n; i++) { // lấy trung bình trong cửa sổ để giảm răng cưa khi hạ tần số
      var a = Math.floor(i * ratio), b = Math.min(buf.length, Math.max(a + 1, Math.floor((i + 1) * ratio))), s = 0;
      for (var j = a; j < b; j++) s += buf[j]; out[i] = s / (b - a);
    } return out;
  }
  function toWav(f32, rate) {
    var n = f32.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf), i;
    function w(o, s) { for (var k = 0; k < s.length; k++) v.setUint8(o + k, s.charCodeAt(k)); }
    w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * 2, true);
    for (i = 0; i < n; i++) { var x = Math.max(-1, Math.min(1, f32[i])); v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true); }
    return new Uint8Array(buf);
  }
  function b64(u8) { var s = '', k = 0x8000; for (var i = 0; i < u8.length; i += k) s += String.fromCharCode.apply(null, u8.subarray(i, i + k)); return btoa(s); }

  /* ---------- gõ câu đọc thử ---------- */
  function tryText(t) {
    t = String(t || '').trim(); if (!t) { setStatus('Hãy gõ hoặc chọn một câu mẫu.', 'warn'); return; }
    setStatus('⏳ Đang phân tích câu…');
    A.api('/api/stt/parse', { text: t }).then(function (res) { show(res, 'Đã đọc xong câu bạn gõ.'); }, function (e) { setStatus('⚠️ ' + (e.message || 'Chưa phân tích được'), 'warn'); });
  }

  /* ---------- hiển thị kết quả + điền biểu mẫu ---------- */
  function show(res, msg) {
    var txt = document.getElementById('vcText'), box = document.getElementById('vcRes');
    txt.hidden = false; txt.innerHTML = '<span>Văn bản nghe được</span>“' + esc(res.text || '') + '”';
    var ok = res.understood || [], miss = res.missing || [];
    box.hidden = false;
    box.innerHTML = (ok.length ? '<div class="vc-lab">✅ Đã điền vào biểu mẫu (' + ok.length + ')</div><div class="vc-chips">' + ok.map(function (u, i) { return '<span class="vc-chip ok c' + (i % 6) + '"><b>' + esc(u.label) + '</b> ' + esc(u.text) + '</span>'; }).join('') + '</div>'
      : '<div class="vc-lab">Mình chưa nghe được chỉ số nào. Hãy nói rõ hơn, ví dụ “nam 55 tuổi, cao 1m65, nặng 70 ký”.</div>') +
      (miss.length ? '<div class="vc-lab">❔ Chưa nghe thấy (giữ nguyên giá trị cũ)</div><div class="vc-chips">' + miss.map(function (m) { return '<span class="vc-chip miss">' + esc(m.label) + '</span>'; }).join('') + '</div>' : '');
    if (ok.length) {
      A.setProfile(res.fields);
      setStatus('✅ ' + msg + ' Đã điền ' + ok.length + ' chỉ số, kéo xuống để kiểm tra lại biểu mẫu.', 'ok');
      A.say && A.say('doctor', 'Mình đã ghi nhận ' + ok.length + ' chỉ số từ lời bạn nói.', { compact: true });
    } else setStatus('ℹ️ ' + msg, 'warn');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
