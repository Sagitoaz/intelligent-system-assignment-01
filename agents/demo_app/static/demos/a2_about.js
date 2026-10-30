DEMOS['a2_about.js'] = {
  mount(root) {
    const box = (cls, title, ...body) => H.el('div', { class: 'box ' + cls }, H.el('b', {}, title), ...body.map(t => H.el('p', { style: 'margin:6px 0 0' }, t)));
    root.append(
      box('blue', '1. Bài toán một câu', 'Hai cửa hàng bán cùng một món hàng. Mỗi ngày mỗi bên chọn giữ giá (10) hoặc giảm giá 20% (còn 8). Cả hai chọn cùng lúc, không biết đối thủ chọn gì hôm nay. Ai kiếm được nhiều lãi hơn?'),
      box('', '2. Tiền lãi tính thế nào', 'Chi phí mỗi món là 5. Giữ giá lãi 5 mỗi món, giảm giá chỉ lãi 3 mỗi món. Nhưng giảm giá làm khách dễ mua hơn (+15% xác suất mua), còn đối thủ giảm giá thì khách bị hút đi (−10%). Ngày đông khách (cầu cao, 20 người) bán nhiều hơn ngày vắng (cầu thấp, 10 người). Nên giảm giá không phải lúc nào cũng lời.'),
      box('green', '3. Hai người bán là ai', '🤖 Agent A học bằng Q-learning: thử, xem lãi, rồi nhớ. 📏 Agent B làm theo luật cố định: chỉ giảm giá khi cầu cao và hôm trước đối thủ vừa giảm, còn lại giữ giá. B không học gì cả.'),
      box('violet', '4. Q-learning học gì', 'Agent chỉ nhìn 2 thông tin: cầu hôm nay (thấp/cao) và hôm trước đối thủ có giảm giá không, tức 4 tình huống. Với mỗi tình huống nó giữ một con số Q cho "giữ giá" và một cho "giảm giá", là lãi mong đợi lâu dài của lựa chọn đó. Sau mỗi ngày nó chỉnh con số: Q mới = Q cũ + alpha × (lãi thực tế + gamma × Q tốt nhất của ngày sau − Q cũ). Cuối cùng chọn hành động có Q lớn hơn: đó là Q-table trong demo.'),
      H.el('h3', {}, 'Ba núm chỉnh trong demo "Huấn luyện & so sánh"'),
      H.table(['Tham số', 'Ý nghĩa đời thường', 'Quá nhỏ', 'Quá lớn'], [
        ['alpha', 'Tin kinh nghiệm mới bao nhiêu', 'Học rất chậm', 'Bị ngày may rủi làm xao động'],
        ['gamma', 'Coi trọng lãi ngày mai bao nhiêu', 'Chỉ nghĩ ngắn hạn', 'Khó phân biệt, học chậm'],
        ['epsilon', 'Tỉ lệ thử bừa để khám phá', 'Dễ kẹt ở lựa chọn tạm ổn', 'Mất lãi vì thử quá nhiều']]),
      box('', '5. Cách đọc kết quả', 'Biểu đồ cột so lãi mỗi bước của Q-learning, luật cố định, luôn giảm giá và không giảm giá. Kết quả thật thú vị: Q-learning thường không thắng luật cố định, vì luật đã khá tốt và thị trường nhiều may rủi. Điều đó không có nghĩa là sai; thí nghiệm cho thấy học máy không tự động hơn luật tốt, nhất là khi ít dữ liệu hoặc ít trạng thái.'),
      box('blue', '6. Thử ngay', 'Mở "Bạn là người bán" để tự chơi 20 vòng với đối thủ theo luật. Mở "Ván 20 bước" để xem từng ngày. Mở "Dữ liệu thật" để xem dữ liệu bán lẻ có thật khớp với giả định "giảm giá thì bán nhiều hơn" không.'));
  },
};
