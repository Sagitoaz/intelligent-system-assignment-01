DEMOS['a1_chase.js'] = {
  mount(root) {
    const maps = [['garden', 'Khu vườn 18×12'], ['maze', 'Mê cung 26×16'], ['city', 'Thành phố 34×20']];
    const pol = [['q', 'Mèo Q-learning'], ['bfs', 'Mèo BFS (biết bản đồ)'], ['random', 'Mèo ngẫu nhiên']];
    const stats = H.el('div'), cv = H.canvas(600, 400), wrap = H.el('div', {}, cv), note = H.el('p', { class: 'note' });
    let timer, seed = 1, wins = 0, games = 0;
    const speed = H.el('input', { type: 'range', min: 20, max: 300, value: 90, id: 'speed' });
    const run = H.el('button', { class: 'btn primary', onclick: () => play(true) }, '▶ Ván mới');
    const auto = H.el('input', { type: 'checkbox', id: 'auto', checked: '' });
    async function play(fresh) {
      clearInterval(timer); run.disabled = true;
      if (fresh) seed = Math.floor(Math.random() * 1e6);
      let ep;
      try { ep = await H.api('a1_chase', { map: map.value, policy: policy.value, seed }); } catch (e) { note.textContent = e.message; run.disabled = false; return; }
      run.disabled = false; let i = 0;
      timer = setInterval(() => {
        const [cat, mouse] = ep.frames[i];
        H.drawMap(cv, ep.map, cat, mouse, ep.frames.slice(0, i).map(f => f[0]));
        stats.replaceChildren(H.stat(`${i}/${ep.steps}`, 'bước'), H.stat(`${wins}/${games}`, 'số ván mèo thắng'),
          H.stat(ep.map.open_cells, 'ô đi được'));
        if (++i >= ep.frames.length) {
          clearInterval(timer); games++; wins += ep.caught;
          note.textContent = ep.caught ? `🐱 bắt được chuột sau ${ep.steps} bước.` : `🐭 chuột thoát (hết ${ep.steps} bước).`;
          if (auto.checked) setTimeout(() => play(true), 900);
        }
      }, +speed.value);
    }
    const map = H.select('Bản đồ', 'map', maps, 'maze').querySelector('select'), policy = H.select('Chính sách', 'policy', pol, 'q').querySelector('select');
    [map, policy].forEach(s => s.addEventListener('change', () => { wins = games = 0; play(true); }));
    root.append(H.el('div', { class: 'row' }, map.parentNode, policy.parentNode, H.el('label', {}, 'Tốc độ (nhỏ = nhanh)', speed),
      H.el('label', {}, 'Tự chạy ván tiếp', auto), run), stats, wrap, note,
      H.el('p', { class: 'note' }, 'Chuột chỉ đi 3/4 số nhịp và chạy xa mèo theo khoảng cách BFS. Mèo Q-learning chỉ thấy tường kề, hướng/khoảng cách thô và hướng đánh hơi khi chuột trong tầm.'));
    play(true);
    new MutationObserver(() => { if (!root.isConnected) clearInterval(timer); }).observe(document.body, { childList: true, subtree: true });
  },
};
