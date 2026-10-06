"""Thử thủ công tự động bằng Edge, có mạng thật nếu --live (không thuộc pytest).

Chạy sau khi khởi động Streamlit với --client.toolbarMode minimal:
 .venv\\Scripts\\python.exe -m scripts.tieuluan.check_app_browser --url http://localhost:8517 --live --screenshots
Ảnh được cắt khoảng trắng đầu bằng Pillow; kết quả JSON lưu trong logs (không commit).
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8517')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--screenshots', action='store_true')
    args = ap.parse_args()
    checks = []; live_sources = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1280, 'height':1180},device_scale_factor=1.5,locale='vi-VN')
        page.set_default_timeout(60000)
        page.goto(args.url)
        page.get_by_role('tab',name='Giới thiệu',exact=True).wait_for()

        def ready():
            page.wait_for_timeout(600)
            page.locator('[data-testid="stStatusWidget"]').wait_for(state='hidden',timeout=60000)
            assert page.locator('[data-testid="stException"]').count() == 0
            text = page.locator('body').inner_text()
            assert 'Traceback' not in text and 'Error' not in text, text
            assert 'Chưa có đủ dữ liệu hoặc trọng số' not in text, text

        def tab(name):
            page.get_by_role('tab',name=name,exact=True).click(); ready(); checks.append(name)

        def choose(label, value):
            page.get_by_role('combobox',name=label,exact=True).click()
            page.get_by_role('option',name=value,exact=True).click()
            ready(); checks.append(label+' → '+value)

        def capture(name):
            if not args.screenshots: return
            from PIL import Image
            path = ROOT / f'figures/tieuluan/appendix_{name}.png'
            page.evaluate('window.scrollTo(0,0)')
            page.locator('[data-testid="stMain"]').evaluate('(e)=>e.scrollTop=0')
            page.screenshot(path=str(path))
            with Image.open(path) as im:
                # Tiêu đề nằm dưới thanh công cụ; giữ một lề nhỏ phía trên.
                y = max(0,int(page.get_by_role('heading',name='Hiểu AI qua dữ liệu tài chính',exact=True).bounding_box()['y']*1.5)-15)
                im.crop((0,y,im.width,im.height)).save(path)

        ready()
        for name in ['Giới thiệu','Thử trên dữ liệu kiểm tra','Dự báo phiên tới','Tự chấm điểm tín dụng','So sánh mô hình','Backtest','Dữ liệu và giới hạn']:
            tab(name)
        tab('Thử trên dữ liệu kiểm tra')
        for market in ['S&P 500','VN-Index','Bitcoin']:
            choose('Bộ dữ liệu kiểm tra',market)
            for kind in ['CNN4','LSTM']:
                choose('Mô hình kiểm tra',kind)
                box = page.get_by_role('combobox',name='Ngày cuối cửa sổ quan sát',exact=True)
                box.click(); box.press('End'); box.press('Enter'); ready()
                assert 'Đã đối chiếu với dự báo đã lưu' in page.locator('body').inner_text()
            if market == 'VN-Index': capture('demo_app')
        for dataset in ['Phá sản doanh nghiệp Đài Loan','Vỡ nợ thẻ tín dụng Đài Loan']:
            choose('Bộ dữ liệu kiểm tra',dataset)
            page.get_by_role('button',name='Chọn ngẫu nhiên',exact=True).click(); ready()
        tab('Tự chấm điểm tín dụng')
        for preset in ['Ít rủi ro hơn · phân vị 10%','Ở giữa · phân vị 50%','Nhiều rủi ro hơn · phân vị 90%']:
            choose('Hồ sơ mẫu theo xác suất trong tập kiểm tra',preset)
            page.get_by_role('button',name='Chấm điểm hồ sơ',exact=True).click(); ready()
            page.get_by_text('Xác suất vỡ nợ tháng tới',exact=True).wait_for()
            assert 'Xác suất vỡ nợ tháng tới' in page.locator('body').inner_text()
        capture('credit')
        tab('So sánh mô hình')
        for dataset in ['S&P 500','VN-Index','Bitcoin','Phá sản doanh nghiệp Đài Loan','Vỡ nợ thẻ tín dụng Đài Loan']:
            choose('Bộ dữ liệu so sánh',dataset)
        tab('Backtest')
        for market in ['S&P 500','VN-Index','Bitcoin']:
            choose('Thị trường mô phỏng',market)
            if market == 'VN-Index': capture('demo_backtest')
        if args.live:
            tab('Dự báo phiên tới')
            for market in ['S&P 500','Bitcoin','VN-Index']:
                choose('Thị trường dự báo',market)
                page.get_by_role('button',name='Tải lại dữ liệu mới nhất',exact=True).click()
                ready()
                page.get_by_text('Nguồn thực tế:',exact=False).wait_for(timeout=60000)
                ready()
                live_sources.append(page.get_by_text('Nguồn thực tế:',exact=False).inner_text())
                assert page.get_by_text('Xác suất tăng',exact=True).count() == 2
            capture('live')
        browser.close()
    output = {'checks':checks,'live_sources':live_sources,'passed':True}
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'logs/tieuluan_browser_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))


if __name__ == '__main__': main()
