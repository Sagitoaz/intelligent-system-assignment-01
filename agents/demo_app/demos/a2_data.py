import shared

META = dict(agent="agent02", id="a2_data", title="Kho dữ liệu CSV", js="data_viewer.js", data_agent="agent02",
            description="Duyệt và tìm kiếm các tệp CSV của Agent 02: nhật ký thị trường, đường học, kết quả thí nghiệm.")


def handle(p):
    if "file" in p:
        return shared.read_csv_page("agent02", p["file"], int(p.get("offset", 0)), int(p.get("limit", 15)), p.get("q", ""))
    return dict(files=shared.list_csv("agent02"))
