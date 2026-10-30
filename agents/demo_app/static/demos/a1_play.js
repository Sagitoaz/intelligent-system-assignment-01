DEMOS['a1_play.js'] = {
  mount(root) {
    const cv = H.canvas(600, 400), msg = H.el('p', { class: 'note' }), stats = H.el('div');
    let st = null, seed = 1, steps = 0, busy = false;
    const map = H.select('Bản đồ', 'pmap', [['garden', 'Khu vườn'], ['maze', 'Mê cung'], ['city', 'Thành phố']], 'garden').querySelector('select');
    const pol = H.select('Mèo', 'ppol', [['q', 'Q-learning'], ['bfs', 'BFS'], ['random', 'Ngẫu nhiên']], 'q').querySelector('select');
    const trail = [];
    async function start() {
      seed = Math.floor(Math.random() * 1e6); steps = 0; trail.length = 0;
      st = await H.api('a1_play', { op: 'new', map: map.value, seed }); draw();
      msg.textContent = 'Dùng phím ← ↑ → ↓ (hoặc W A S D) để chạy trốn. Mèo đi sau mỗi bước của bạn.';
    }
    function draw() {
      H.drawMap(cv, st.map, st.cat, st.mouse, trail);
      stats.replaceChildren(H.stat(steps, 'bước bạn đã chạy'));
    }
    const keys = { ArrowUp: 0, w: 0, ArrowRight: 1, d: 1, ArrowDown: 2, s: 2, ArrowLeft: 3, a: 3 };
    async function onKey(e) {
      if (!root.isConnected) return document.removeEventListener('keydown', onKey);
      if (!(e.key in keys) || busy || !st || st.caught) return;
      e.preventDefault(); busy = true;
      try {
        const r = await H.api('a1_play', { map: map.value, policy: pol.value, cat: st.cat.join(','), mouse: st.mouse.join(','), dir: keys[e.key], seed: seed + steps });
        trail.push(st.cat); steps++; st = { ...st, cat: r.cat, mouse: r.mouse, caught: r.caught }; draw();
        if (r.caught) msg.textContent = `🐱 Mèo bắt được bạn sau ${steps} bước. Bấm "Ván mới" để chơi lại.`;
      } finally { busy = false; }
    }
    document.addEventListener('keydown', onKey);
    map.addEventListener('change', start);
    root.append(H.el('div', { class: 'row' }, map.parentNode, pol.parentNode, H.el('button', { class: 'btn primary', onclick: start }, 'Ván mới')), stats, cv, msg);
    start();
  },
};
