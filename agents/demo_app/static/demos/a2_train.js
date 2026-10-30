DEMOS['a2_train.js'] = {
  mount(root) {
    const out = H.el('div'), run = H.el('button', { class: 'btn primary', onclick: go }, 'Huấn luyện');
    const f = {
      alpha: H.number('alpha', 'alpha', 0.1, 0.01, 0.001, 1), gamma: H.number('gamma', 'gamma', 0.9, 0.05, 0, 1),
      eps: H.number('epsilon đầu', 'eps', 0.2, 0.05, 0, 1), episodes: H.number('số episode', 'episodes', 3000, 500, 100, 10000),
      opponent: H.select('Đối thủ', 'opp', [['rule', 'Luật cố định'], ['always', 'Luôn giảm giá'], ['never', 'Không giảm giá']], 'rule'),
    };
    const val = k => f[k].querySelector('input,select').value;
    async function go() {
      run.disabled = true; out.replaceChildren(H.el('p', { class: 'note' }, 'Đang huấn luyện ... (vài giây)'));
      try {
        const r = await H.api('a2_train', { alpha: val('alpha'), gamma: val('gamma'), eps: val('eps'), episodes: val('episodes'), opponent: val('opponent') });
        const cv = H.canvas(560, 260), bars = H.canvas(560, 260), win = 50, sm = [];
        for (let i = 0; i + win <= r.curve.length; i += win) sm.push(r.curve.slice(i, i + win).reduce((a, b) => a + b, 0) / win);
        H.line(cv, [{ name: 'Q-learning', color: '#f0603a', data: sm }], { yLabel: 'lợi nhuận/bước (TB 50 episode)' });
        H.bars(bars, r.summary.map(s => ({ label: s.agent, values: [s.profit_per_step, s.opponent_profit_per_step] })), ['Agent', 'Đối thủ'], ['#0d9f8f', '#7f95ad']);
        const q = r.q; const qrows = [];
        for (let d = 0; d < 2; d++) for (let p = 0; p < 2; p++) qrows.push([d ? 'cao' : 'thấp', p ? 'giảm' : 'không', q[d][p][0].toFixed(1), q[d][p][1].toFixed(1), r.policy[`cầu=${d ? 'cao' : 'thấp'}, đối thủ trước=${p ? 'giảm' : 'không'}`]]);
        out.replaceChildren(H.el('p', {}, 'Đối thủ: ', H.el('b', {}, r.opponent_title)),
          H.el('div', { class: 'grid2' }, H.el('div', {}, H.el('b', {}, 'Đường học'), cv), H.el('div', {}, H.el('b', {}, 'Lợi nhuận/bước khi đánh giá (500 episode × 20 bước)'), bars)),
          H.el('p', {}, H.el('b', {}, 'Q-table & chính sách tham lam')),
          H.table(['Cầu', 'Đối thủ vừa giảm?', 'Q(giữ giá)', 'Q(giảm giá)', 'Chọn'], qrows),
          H.el('p', {}, H.el('b', {}, 'Kết quả chi tiết')),
          H.table(['Agent', 'Lợi nhuận/bước', 'Đối thủ/bước', 'Tỉ lệ giảm giá'], r.summary.map(s => [s.agent, s.profit_per_step, s.opponent_profit_per_step, (s.discount_rate * 100).toFixed(1) + '%'])));
      } catch (e) { out.replaceChildren(H.el('p', { class: 'err' }, e.message)); }
      run.disabled = false;
    }
    root.append(H.el('div', { class: 'row' }, Object.values(f), run), out); go();
  },
};
