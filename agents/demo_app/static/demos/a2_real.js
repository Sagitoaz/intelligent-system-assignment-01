DEMOS['a2_real.js'] = {
  async mount(root) {
    root.append(H.el('p', { class: 'note' }, 'Đang đọc 336.724 dòng dữ liệu thật ...'));
    const r = await H.api('a2_real');
    root.replaceChildren();
    const cv = H.canvas(1140, 300);
    H.bars(cv, r.rows.slice(0, 12).map(x => ({ label: x.code, values: [x.normal_units, x.low_units] })), ['Ngày giá thường', 'Ngày giá thấp hơn 5%+'], ['#7f95ad', '#f0603a'], { fmt: v => v.toFixed(0) });
    root.append(
      H.el('div', {}, H.stat(r.products.toLocaleString('vi-VN'), 'mặt hàng trong dữ liệu'), H.stat(r.compared, 'mặt hàng đủ ngày để so'), H.stat(`${r.higher}/${r.compared}`, 'mặt hàng bán nhiều hơn khi giá thấp')),
      H.el('div', { class: 'box blue' }, 'Mỗi mặt hàng được chia thành ngày giá thường và ngày giá thấp hơn mức thường ít nhất 5%, rồi so số lượng bán trung bình mỗi ngày. Đây là dữ liệu quan sát thật của một cửa hàng online ở Anh (2010–2011), chưa tách được tác động của mùa vụ hay khuyến mãi theo lô, nên chỉ minh họa xu hướng, không chứng minh nhân quả.'),
      H.el('b', {}, 'Số lượng bán trung bình mỗi ngày: giá thường và giá thấp hơn'), cv,
      H.table(['Mã hàng', 'Giá thường', 'Số ngày giá thường', 'Số ngày giá thấp', 'Bán/ngày (thường)', 'Bán/ngày (giá thấp)'],
        r.rows.map(x => [x.code, x.typical_price, x.normal_days, x.low_days, x.normal_units, x.low_units])),
      H.el('p', { class: 'note' }, 'Nguồn: UCI Machine Learning Repository, Online Retail (Chen, 2015), CC BY 4.0. Tệp: data/agent02/external/.'));
  },
};
