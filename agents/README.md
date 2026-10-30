# Bài tập Agent 01 và Agent 02

```
agents/
├── agent01/      Discovering AI Agent: báo cáo Word, notebook, bản mở rộng (expanded/)
├── agent02/      RL vs Rule-based (hai người bán): báo cáo Word, notebook, mã thí nghiệm
└── demo_app/     Web demo chạy localhost cho cả hai agent
data/agent01/     CSV: 60.000 đơn hàng, 12.000 câu hỏi, bản đồ, 13.500 ván mèo đuổi chuột
data/agent02/     CSV: 240.000 bước thị trường, 250.000 dòng nhật ký huấn luyện, kết quả thí nghiệm
```

## Chạy demo trên localhost

```powershell
python agents/demo_app/server.py --open     # mở http://localhost:8100
```

Chỉ dùng thư viện chuẩn của Python 3.10+, không cần cài thêm. Q-table của mèo được lưu sẵn
trong `agents/demo_app/cache/`, nên máy chủ khởi động ngay mà không huấn luyện lại.

| Agent | Demo |
|---|---|
| Agent 01 | Mèo đuổi chuột (3 bản đồ tới 34×20) · Bạn là chuột · So sánh chính sách · Trợ lý đơn hàng · Kho dữ liệu CSV |
| Agent 02 | Huấn luyện & so sánh · Ba kiểu đối thủ · Ván 20 bước · Quét siêu tham số · Bạn là người bán · Kho dữ liệu CSV |

## Thêm demo mới

1. Tạo `demos/<tên>.py` có `META = dict(agent="agent01"|"agent02", id=..., title=..., js="<tên>.js", description=...)` và `handle(params) -> dict`.
2. Tạo `static/demos/<tên>.js` đăng ký `DEMOS['<tên>.js'] = { mount(root, meta) {...} }` (có sẵn `H.api`, `H.line`, `H.bars`, `H.table`, `H.drawMap`).
3. Thêm id vào `ORDER` trong `server.py` nếu muốn đặt vị trí trong menu. Khởi động lại máy chủ.

## Dữ liệu và ảnh

- `python agents/demo_app/build_datasets.py` tạo lại các CSV (seed cố định, không huấn luyện mô hình).
- `agents/demo_app/screenshots/` là ảnh chụp từng demo.
