DEMOS['a1_orders.js'] = {
  mount(root) {
    const out = H.el('div'), text = H.el('input', { id: 'text', style: 'width:360px', value: 'hoàn tiền đơn DH100001' });
    const owner = H.el('input', { id: 'owner', style: 'width:90px', placeholder: 'vd U0042' });
    const examples = ['hoàn tiền đơn DH100001', 'hủy đơn DH100002', 'đơn DH100003 đang ở đâu', 'đơn DH100004 giao thiếu hàng', 'chính sách đổi trả thế nào', 'giúp mình với'];
    async function ask() {
      out.replaceChildren(H.el('p', { class: 'note' }, 'Đang xử lý ...'));
      const r = await H.api('a1_orders', { text: text.value, owner: owner.value });
      out.replaceChildren(
        H.el('p', {}, H.el('span', { class: 'badge' }, 'ý định: ' + r.intent)),
        H.el('ol', {}, r.steps.map(s => H.el('li', {}, s))),
        r.order ? H.table(Object.keys(r.order), [Object.values(r.order)]) : null,
        H.el('p', { class: 'good' }, '🤖 ' + r.answer));
    }
    root.append(H.el('div', { class: 'row' }, H.el('label', {}, 'Yêu cầu của khách', text), H.el('label', {}, 'Mã người dùng (tuỳ chọn)', owner),
      H.el('button', { class: 'btn primary', onclick: ask }, 'Gửi')),
      H.el('div', { class: 'row' }, examples.map(e => H.el('button', { class: 'btn', onclick: () => { text.value = e; ask(); } }, e))), out,
      H.el('p', { class: 'note' }, 'Dữ liệu: data/agent01/orders_large.csv (60.000 đơn tổng hợp). Ví dụ phân quyền: nhập một mã người dùng khác chủ đơn để thấy agent từ chối.'));
    ask();
  },
};
