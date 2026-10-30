DEMOS['a2_episode.js'] = {
  mount(root) {
    const out = H.el('div');
    const opp = H.select('Đối thủ', 'eopp', [['rule', 'Luật cố định'], ['always', 'Luôn giảm giá'], ['never', 'Không giảm giá']], 'rule');
    const seedIn = H.number('seed', 'eseed', 42, 1, 1, 9999);
    async function go() {
      const r = await H.api('a2_episode', { opponent: opp.querySelector('select').value, seed: seedIn.querySelector('input').value });
      const t = r.trace, sum = k => t.reduce((s, x) => s + x[k], 0), cv = H.canvas(1140, 240);
      let ca = 0, cb = 0; const A = [], B = [];
      t.forEach(x => { A.push(ca += x.rl_profit); B.push(cb += x.opp_profit); });
      H.line(cv, [{ name: 'Q-learning', color: '#f0603a', data: A }, { name: 'Đối thủ', color: '#0d9f8f', data: B }], { yLabel: 'lợi nhuận tích luỹ', min: 0 });
      out.replaceChildren(H.el('p', {}, 'Đối thủ: ', H.el('b', {}, r.opponent_title)),
        H.stat(sum('rl_profit').toFixed(0), 'lợi nhuận Q-learning'), H.stat(sum('opp_profit').toFixed(0), 'lợi nhuận đối thủ'), cv,
        H.table(['Bước', 'Cầu', 'Q-learning', 'Đối thủ', 'Bán (Q)', 'Bán (ĐT)', 'Lãi (Q)', 'Lãi (ĐT)'],
          t.map(x => [x.step, x.demand ? 'cao' : 'thấp', x.rl_action ? 'giảm giá' : 'giữ giá', x.opp_action ? 'giảm giá' : 'giữ giá', x.rl_sales, x.opp_sales, x.rl_profit, x.opp_profit])));
    }
    root.append(H.el('div', { class: 'row' }, opp, seedIn, H.el('button', { class: 'btn primary', onclick: go }, 'Chạy ván này')), out); go();
  },
};
