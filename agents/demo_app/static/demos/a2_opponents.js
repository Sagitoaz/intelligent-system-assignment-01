DEMOS['a2_opponents.js'] = {
  async mount(root) {
    root.append(H.el('p', { class: 'note' }, 'Đang huấn luyện Q-learning với 3 kiểu đối thủ ...'));
    const r = await H.api('a2_opponents');
    root.replaceChildren();
    const cv = H.canvas(1140, 260), colors = ['#f0603a', '#0d9f8f', '#7060d8'], sm = c => { const o = []; for (let i = 0; i + 50 <= c.length; i += 50) o.push(c.slice(i, i + 50).reduce((a, b) => a + b, 0) / 50); return o; };
    H.line(cv, r.items.map((it, i) => ({ name: it.opponent, color: colors[i], data: sm(it.curve) })), { yLabel: 'lợi nhuận/bước của Q-learning' });
    root.append(H.el('b', {}, 'Đường học theo từng đối thủ'), cv);
    r.items.forEach(it => root.append(H.el('h3', {}, it.title),
      H.table(['Agent', 'Lợi nhuận/bước', 'Đối thủ/bước', 'Tỉ lệ giảm giá'], it.summary.map(s => [s.agent, s.profit_per_step, s.opponent_profit_per_step, (s.discount_rate * 100).toFixed(1) + '%'])),
      H.el('p', { class: 'note' }, 'Chính sách học được: ' + Object.entries(it.policy).map(([k, v]) => `${k} → ${v}`).join(' · '))));
  },
};
