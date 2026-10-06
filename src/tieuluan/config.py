"""Cấu hình dùng chung: đường dẫn, mốc thời gian, danh mục tài sản, hạt giống ngẫu nhiên."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tieuluan"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tieuluan"
FIGURES_DIR = PROJECT_ROOT / "figures" / "tieuluan"
MODELS_DIR = PROJECT_ROOT / "models" / "tieuluan"
DOCS_DIR = PROJECT_ROOT / "docs" / "tieuluan"

# Ảnh chụp dữ liệu (data snapshot): mọi chuỗi giá được cắt tại cùng một ngày
# để kết quả tái lập được dù sau này nguồn dữ liệu cập nhật thêm phiên mới.
SNAPSHOT_START = "2000-01-01"
SNAPSHOT_END = "2026-09-30"  # bao gồm ngày này

# Mốc chia dữ liệu theo thời gian, dùng chung cho mọi thị trường ở Chương 2–4.
# Train ≤ 2019 | Validation 2020–2022 | Test 2023-01-01 → 2026-09-30.
TRAIN_END = "2019-12-31"
VAL_END = "2022-12-31"

SEEDS = (11, 22, 33)

# Ngày bắt đầu phân tích của từng chỉ số trong thực nghiệm.
# - S&P 500: từ 2000 (snapshot bắt đầu từ đây).
# - VN-Index: từ 2010. Trước 07/2009 chỉ DNSE có dữ liệu và giá được lưu làm tròn tới số
#   nguyên (2003–06/2009), sinh ra hàng trăm phiên "đứng giá" giả và OHLC phẳng; SSI có giá
#   hai chữ số thập phân liên tục từ 07/2009, nên 2010 là năm đầy đủ đầu tiên đạt chất lượng.
# - Bitcoin: từ phiên đầu tiên có trên Yahoo Finance (17/09/2014).
ANALYSIS_START = {"sp500": "2000-01-01", "vnindex": "2010-01-01", "btc": "2014-09-17"}

# Ba thị trường đại diện cho ba "kiểu" thị trường khác nhau.
MARKET_INDICES = {
    "sp500": {"source": "yahoo", "symbol": "^GSPC", "name": "S&P 500", "country": "Mỹ"},
    "vnindex": {"source": "vn", "symbol": "VNINDEX", "name": "VN-Index", "country": "Việt Nam"},
    "btc": {"source": "yahoo", "symbol": "BTC-USD", "name": "Bitcoin (BTC-USD)", "country": "Toàn cầu"},
}

# 30 cổ phiếu thành phần chỉ số Dow Jones Industrial Average, theo danh sách
# hiện hành sau khi Alphabet thay Verizon (hiệu lực 29/06/2026, thông cáo S&P DJI 23/06/2026).
US_STOCKS = (
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "WMT",
)

# 30 cổ phiếu rổ VN30 tại ngày 05/10/2026 (API công khai của VPS: getlistckindex/VN30).
VN_STOCKS = (
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MCH", "MSN", "MWG", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TCX", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
)

# 20 tiền mã hóa vốn hóa lớn (không gồm stablecoin), có lịch sử trên Yahoo Finance.
CRYPTO_ASSETS = (
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD", "DOGE-USD", "ADA-USD",
    "TRX-USD", "LINK-USD", "XLM-USD", "LTC-USD", "BCH-USD", "AVAX-USD", "DOT-USD",
    "XMR-USD", "ETC-USD", "HBAR-USD", "ATOM-USD", "ALGO-USD", "VET-USD",
)

UCI_DATASETS = {
    "taiwan_bankruptcy": {
        "id": 572,
        "url": "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip",
        "page": "https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction",
    },
    "credit_default": {
        "id": 350,
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "page": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    },
}


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
