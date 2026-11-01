"""Demo tiếng Việt cho Chương 2–4 của tiểu luận: chỉ suy luận bằng trọng số đã lưu, không huấn luyện lại.

Tham số URL (hub gộp nhúng bằng iframe): ?embed=true&chapter=2|3|4&dataset=<id>&kind=<mô hình>
  - chapter: chọn chương (không có thì hiện cả ba, chọn bằng nút radio)
  - dataset: thị trường mặc định ở Chương 3, 4 (sp500 | vnindex | btc); Chương 2 chỉ có một bộ dữ liệu vay
    (các mã cũ credit_default, taiwan_bankruptcy được chấp nhận và quy về bộ vay Lending Club)
  - kind: giữ để tương thích; mô hình hiển thị cố định theo chương (MLP, CNN4, LSTM)
  - embed=true: ẩn thanh công cụ Streamlit cho gọn
"""
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import common  # noqa: E402
import ch2  # noqa: E402
import market  # noqa: E402

TITLES = {'2': 'Chương 2 · Học máy: chấm điểm rủi ro một khoản vay',
          '3': 'Chương 3 · CNN: máy “nhìn” biểu đồ giá',
          '4': 'Chương 4 · RNN, LSTM: máy “đọc” chuỗi tăng giảm giá'}


def run_chapter(chapter, records, analysis):
    try:
        if chapter == '2':
            ch2.render(records, analysis)
        else:
            market.render(chapter, records, analysis)
    except Exception as error:  # noqa: BLE001 - mọi lỗi đều phải thành thông báo thân thiện, không để trang trắng
        st.warning('Mục này tạm thời chưa hiển thị được vì thiếu dữ liệu hoặc tệp mô hình. '
                   'Hãy kiểm tra thư mục dữ liệu và kết quả rồi tải lại trang.')
        st.caption(f'Chi tiết cho người quản trị: {type(error).__name__}: {str(error)[:140]}')


def main():
    common.configure()
    embed = str(st.query_params.get('embed', '')).lower() in ('1', 'true', 'yes')
    st.set_page_config(page_title='Hiểu AI qua dữ liệu tài chính', layout='wide')
    common.inject_css(embed)
    records = common.read_records(str(common.RESULTS))
    analysis = common.safe_json(common.RESULTS / 'analysis.json')
    chapter = str(st.query_params.get('chapter', ''))
    if chapter not in TITLES:
        st.title('Hiểu AI qua dữ liệu tài chính')
        st.write('Chọn một chương rồi thử trực tiếp: mỗi chương có phần giải thích “Demo này làm gì?” và “Cách đọc kết quả”.')
        chapter = st.radio('Chương', list(TITLES), format_func=TITLES.get, horizontal=True, key='chapter_pick')
    elif not embed:
        st.title(TITLES[chapter])
    run_chapter(chapter, records, analysis)


if __name__ == '__main__':
    main()
