import shared

META = dict(agent="agent01", id="a1_data", title="Kho dữ liệu CSV", js="data_viewer.js", data_agent="agent01",
            description="Duyệt và tìm kiếm các tệp CSV của Agent 01: đơn hàng, câu hỏi, bản đồ, nhật ký đuổi bắt.")


def handle(p):
    if "file" in p:
        return shared.read_csv_page("agent01", p["file"], int(p.get("offset", 0)), int(p.get("limit", 15)), p.get("q", ""))
    return dict(files=shared.list_csv("agent01"))
