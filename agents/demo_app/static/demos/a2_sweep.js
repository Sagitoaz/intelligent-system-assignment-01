DEMOS['a2_sweep.js'] = {
  mount(root) {
    const out = H.el('div'), opp = H.select('Đối thủ', 'sopp', [['rule', 'Luật cố định'], ['always', 'Luôn giảm giá'], ['never', 'Không giảm giá']], 'rule');
    async function go() {
      out.replaceChildren(H.el('p', { class: 'note' }, 'Đang chạy 9 cấu hình (lần đầu mất khoảng 10 giây) ...'));
      const r = await H.api('a2_sweep', { opponent: opp.querySelector('select').value });
      const best = Math.max(...r.rows.map(x => x.profit)), cv = H.canvas(1140, 280);
      H.bars(cv, [0.01, 0.1, 0.5].map(a => ({ label: 'alpha ' + a, values: r.rows.filter(x => x.alpha === a).map(x => x.profit) })),
        ['ε=0.05', 'ε=0.2', 'ε=0.5'], ['#0d9f8f', '#f0603a', '#7060d8'], { max: Math.max(...r.rows.map(x => x.profit)) * 1.2 });
      out.replaceChildren(cv, H.el('p', { class: 'note' }, `Luật cố định đạt ${r.rows[0].rule_profit}/bước làm mốc tham chiếu.`),
        H.table(['alpha', 'epsilon đầu', 'Lợi nhuận/bước', 'Tỉ lệ giảm giá', 'Mốc luật'], r.rows.map(x => [x.alpha, x.epsilon,
          x.profit === best ? H.el('span', { class: 'good' }, x.profit + ' ★') : x.profit, (x.discount_rate * 100).toFixed(1) + '%', x.rule_profit])));
    }
    root.append(H.el('div', { class: 'row' }, opp, H.el('button', { class: 'btn primary', onclick: go }, 'Quét')), out); go();
  },
};
