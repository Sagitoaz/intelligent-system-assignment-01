DEMOS['a1_real.js'] = {
  async mount(root) {
    const r = await H.api('a1_real'), cv = H.canvas(560, 280), ln = H.canvas(560, 280);
    H.bars(cv, r.countries.map(c => ({ label: c.country.slice(0, 9), values: [c.cancelled / c.invoices * 100] })), ['% hóa đơn bị hủy/hoàn'], ['#7060d8'], { fmt: v => v.toFixed(1) + '%' });
    H.line(ln, [{ name: 'Hóa đơn', color: '#2589e0', data: r.months.map(m => m.invoices) }, { name: 'Đơn hủy', color: '#f0603a', data: r.months.map(m => m.cancelled) }], { yLabel: 'theo tháng 12/2010 – 12/2011', min: 0 });
    root.append(H.el('div', {}, H.stat(r.total.toLocaleString('vi-VN'), 'hóa đơn'), H.stat(r.cancelled.toLocaleString('vi-VN'), 'hóa đơn hủy/hoàn (mã bắt đầu bằng C)'), H.stat((r.cancelled / r.total * 100).toFixed(1) + '%', 'tỉ lệ hủy/hoàn')),
      H.el('div', { class: 'grid2' }, H.el('div', {}, H.el('b', {}, 'Tỉ lệ hủy/hoàn theo quốc gia (8 nước nhiều hóa đơn nhất)'), cv), H.el('div', {}, H.el('b', {}, 'Số hóa đơn và đơn hủy theo tháng'), ln)),
      H.table(['Quốc gia', 'Hóa đơn', 'Hủy/hoàn', 'Doanh thu (GBP)'], r.countries.map(c => [c.country, c.invoices.toLocaleString('vi-VN'), c.cancelled, c.revenue.toLocaleString('vi-VN')])),
      H.el('p', { class: 'note' }, 'Nguồn: UCI Online Retail (Chen, 2015), CC BY 4.0. Đây là dữ liệu thật, tách biệt với 60.000 đơn tổng hợp dùng cho "Trợ lý đơn hàng". Tệp: data/agent01/external/online_retail_invoices.csv.'));
  },
};
