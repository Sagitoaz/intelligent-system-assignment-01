DEMOS['a2_market.js'] = {
  mount(root) {
    const view = H.el('div'), log = [];
    let seed, rnd, demand, prevA, prevB, you = 0, bot = 0;
    async function start() {
      seed = Math.floor(Math.random() * 1e5); rnd = 0; prevA = prevB = you = bot = 0; log.length = 0;
      demand = (await H.api('a2_market', { op: 'new', seed })).demand; draw();
    }
    async function act(a) {
      if (rnd >= 20) return;
      const r = await H.api('a2_market', { seed, round: rnd, demand, prev_a: prevA, prev_b: prevB, action: a });
      log.unshift([r.round, r.demand ? 'cao' : 'thấp', r.you ? 'giảm giá' : 'giữ giá', r.bot ? 'giảm giá' : 'giữ giá', r.sales[0], r.sales[1], r.profit[0], r.profit[1]]);
      you += r.profit[0]; bot += r.profit[1]; prevA = r.you; prevB = r.bot; demand = r.next_demand; rnd = r.round; draw();
    }
    function draw() {
      const done = rnd >= 20;
      view.replaceChildren(H.el('div', {}, H.stat(`${rnd}/20`, 'vòng'), H.stat(you.toFixed(0), 'lợi nhuận của bạn'), H.stat(bot.toFixed(0), 'lợi nhuận đối thủ'),
        H.stat(done ? (you > bot ? '🏆 Bạn thắng' : you < bot ? 'Đối thủ thắng' : 'Hòa') : (demand ? 'cao' : 'thấp'), done ? 'kết quả' : 'nhu cầu vòng này')),
        H.el('div', { class: 'row' }, H.el('button', { class: 'btn primary', disabled: done ? '' : undefined, onclick: () => act(0) }, 'Giữ giá (10)'),
          H.el('button', { class: 'btn', disabled: done ? '' : undefined, onclick: () => act(1) }, 'Giảm giá 20% (8)'), H.el('button', { class: 'btn', onclick: start }, 'Ván mới')),
        log.length ? H.table(['Vòng', 'Cầu', 'Bạn', 'Đối thủ', 'Bán (bạn)', 'Bán (ĐT)', 'Lãi (bạn)', 'Lãi (ĐT)'], log) : null,
        H.el('p', { class: 'note' }, 'Giá gốc 10, chi phí 5. Giảm giá tăng xác suất mua nhưng giảm lãi mỗi sản phẩm. Đối thủ giảm giá khi cầu cao và bạn vừa giảm ở vòng trước.'));
    }
    root.append(view); start();
  },
};
