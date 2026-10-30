DEMOS['a1_compare.js'] = {
  async mount(root) {
    root.append(H.el('p', { class: 'note' }, 'Đang đánh giá 3 chính sách × 3 bản đồ ...'));
    const res = await Promise.all(['garden', 'maze', 'city'].map(m => H.api('a1_compare', { map: m })));
    root.replaceChildren();
    const names = ['Ngẫu nhiên', 'Q-learning', 'BFS (oracle)'], colors = ['#7f95ad', '#f0603a', '#0d9f8f'];
    const rate = H.canvas(560, 300), steps = H.canvas(560, 300), curve = H.canvas(1140, 260);
    H.bars(rate, res.map(r => ({ label: r.map.title, values: r.rows.map(x => x.catch_rate * 100) })), names, colors, { fmt: v => v.toFixed(0) + '%', max: 118 });
    H.bars(steps, res.map(r => ({ label: r.map.title, values: r.rows.map(x => x.mean_steps_all) })), names, colors, { fmt: v => v.toFixed(0) });
    const palette = ['#f0603a', '#0d9f8f', '#7060d8'];
    H.line(curve, res.map((r, i) => ({ name: r.map.title.split(' ')[0] + ' / ' + r.episodes + ' ván', color: palette[i], data: r.curve })), { yLabel: 'tỉ lệ bắt được (cửa sổ 100 ván huấn luyện)', min: 0, max: 1 });
    root.append(H.el('div', { class: 'grid2' }, H.el('div', {}, H.el('b', {}, 'Tỉ lệ bắt được chuột (%)'), rate), H.el('div', {}, H.el('b', {}, 'Số bước trung bình mỗi ván (tối đa 250)'), steps)),
      H.el('p', {}, H.el('b', {}, 'Đường học của Q-learning')), curve,
      H.el('p', {}, H.el('b', {}, 'Bảng số liệu (150 ván đánh giá, seed cố định)')),
      H.table(['Bản đồ', 'Chính sách', 'Ván', 'Tỉ lệ bắt', 'Bước TB khi bắt', 'Bước TB tất cả'],
        res.flatMap(r => r.rows.map(x => [r.map.title, x.policy, x.episodes, (x.catch_rate * 100).toFixed(1) + '%', x.mean_steps_when_caught ?? '—', x.mean_steps_all]))),
      H.el('p', { class: 'note' }, 'BFS biết toàn bộ bản đồ nên là mốc tham chiếu; Q-learning chỉ dùng quan sát cục bộ nên chậm hơn nhưng vẫn bắt được hầu hết các ván.'));
  },
};
