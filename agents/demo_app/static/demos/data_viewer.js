DEMOS['data_viewer.js'] = {
  async mount(root, demo) {
    const id = demo.id, limit = 15; let file = null, offset = 0, query = '';
    const list = await H.api(id), body = H.el('div'), info = H.el('p', { class: 'note' });
    const q = H.el('input', { placeholder: 'tìm trong tệp ...', style: 'width:220px' });
    const total = list.files.reduce((s, f) => s + f.rows, 0);
    root.append(H.el('p', {}, `${list.files.length} tệp CSV · ${total.toLocaleString('vi-VN')} dòng dữ liệu`),
      H.table(['Tệp', 'Số dòng', 'KB', ''], list.files.map(f => [f.file, f.rows.toLocaleString('vi-VN'), f.kb.toLocaleString('vi-VN'),
        H.el('button', { class: 'btn', onclick: () => { file = f.file; offset = 0; query = ''; q.value = ''; load(); } }, 'Xem')])),
      H.el('div', { class: 'row', style: 'margin-top:14px' }, H.el('label', {}, 'Tìm kiếm', q),
        H.el('button', { class: 'btn', onclick: () => { query = q.value; offset = 0; load(); } }, 'Tìm'),
        H.el('button', { class: 'btn', onclick: () => { offset = Math.max(0, offset - limit); load(); } }, '‹ Trước'),
        H.el('button', { class: 'btn', onclick: () => { offset += limit; load(); } }, 'Sau ›')), info, body);
    async function load() {
      if (!file) return;
      const r = await H.api(id, { file, offset, limit, q: query });
      info.textContent = `${file}: dòng ${r.total ? offset + 1 : 0}–${offset + r.rows.length} / ${r.total.toLocaleString('vi-VN')}`;
      body.replaceChildren(H.table(r.header, r.rows));
    }
    file = list.files.sort((a, b) => b.rows - a.rows)[0]?.file; load();
  },
};
