"""Phân tích bổ sung từ dự báo đã lưu: khoảng tin cậy AUC và backtest minh họa có phí giao dịch.

Chạy sau run_experiments:  .venv\\Scripts\\python.exe -m scripts.tieuluan.analyze_results
Kết quả: results/tieuluan/full/analysis.json (đọc bởi build_report).

Quy ước "dự báo đại diện" của một kiến trúc: trung bình xác suất của 3 seed bản PyTorch (riêng
hồi quy logistic/rừng ngẫu nhiên dùng scikit-learn). Ngưỡng của chiến lược được chọn trên
VALIDATION như ở bài toán phân loại, nên tập test không tham gia bất kỳ lựa chọn nào.
"""

import json

import numpy as np

from scripts.tieuluan.run_experiments import select_threshold
from src.tieuluan.analysis import backtest, bootstrap_auc, bootstrap_auc_difference
from src.tieuluan.experiment_data import OUT, ROOT

RESULTS = ROOT / "results/tieuluan/full"
SEEDS = (11, 22, 33)
MARKETS = ("sp500", "vnindex", "btc")
# Phí mỗi chiều giao dịch (mua hoặc bán), giả định thận trọng cho nhà đầu tư cá nhân:
# - S&P 500: quỹ ETF mô phỏng chỉ số, môi giới 0 đồng, chênh lệch giá mua–bán ≈ 0,01–0,05% → 0,05%.
# - VN-Index: phí môi giới ≈ 0,15% mỗi chiều + thuế 0,1% trên giá trị bán → bình quân 0,20%/chiều.
# - Bitcoin: phí taker phổ biến trên sàn giao ngay ≈ 0,10%.
COST = {"sp500": 0.0005, "vnindex": 0.0020, "btc": 0.0010}
PERIODS = {"sp500": 252, "vnindex": 252, "btc": 365}
MARKET_KINDS = ("cnn4", "cnn8", "cnndeep", "rnn", "lstm", "gru")


def load_prediction(dataset, kind, framework, seed):
    with np.load(RESULTS / f"{dataset}__{kind}__{framework}__{seed}_predictions.npz", allow_pickle=False) as d:
        return {key: d[key] for key in d.files}


def ensemble(dataset, kind, framework):
    runs = [load_prediction(dataset, kind, framework, s) for s in SEEDS]
    return (runs[0]["y"], np.mean([r["probability"] for r in runs], axis=0),
            runs[0]["val_y"], np.mean([r["val_probability"] for r in runs], axis=0))


def strip(result):
    return {k: v for k, v in result.items() if k != "equity"}


def run_backtest(position, returns, dataset):
    """Backtest có phí; kèm lợi suất và Sharpe khi bỏ phí để thấy phần lợi thế bị chi phí lấy đi."""
    result = backtest(position, returns, cost_per_side=COST[dataset], periods_per_year=PERIODS[dataset])
    gross = backtest(position, returns, cost_per_side=0.0, periods_per_year=PERIODS[dataset])
    result["annual_return_gross"], result["sharpe_gross"] = gross["annual_return"], gross["sharpe"]
    return result


def main():
    analysis = {"auc_ci": {}, "backtest": {}, "equity": {}, "uci": {},
                "assumptions": {"cost_per_side": COST, "periods_per_year": PERIODS, "block": 20, "n_boot": 2000,
                                "representative": "mean probability of 3 PyTorch seeds"}}
    for dataset in MARKETS:
        with np.load(OUT / f"{dataset}.npz", allow_pickle=False) as d:
            returns, target_dates = d["test_return"], d["test_target_date"]
        analysis["auc_ci"][dataset], analysis["backtest"][dataset] = {}, {}
        persistence = load_prediction(dataset, "persistence", "baseline", 0)
        y = persistence["y"]
        analysis["auc_ci"][dataset]["persistence"] = bootstrap_auc(y, persistence["probability"], block=20)
        curves = {"buy_hold": run_backtest(np.ones_like(returns), returns, dataset),
                  "persistence": run_backtest(persistence["probability"], returns, dataset)}
        for kind in MARKET_KINDS:
            y, p, yv, pv = ensemble(dataset, kind, "pytorch")
            analysis["auc_ci"][dataset][kind] = bootstrap_auc(y, p, block=20)
            threshold = select_threshold(yv, pv)
            curves[kind] = run_backtest((p >= threshold).astype(float), returns, dataset)
            curves[kind]["threshold"] = threshold
        analysis["backtest"][dataset] = {name: strip(r) for name, r in curves.items()}
        analysis["equity"][dataset] = {"dates": [str(t) for t in target_dates],
                                       **{name: r["equity"].round(6).tolist() for name, r in curves.items()}}
    # Hai bộ dữ liệu bảng: khoảng tin cậy AUC và so sánh có ghép cặp giữa các mô hình.
    for dataset in ("taiwan_bankruptcy", "credit_default"):
        y, p_mlp, _, _ = ensemble(dataset, "mlp", "pytorch")
        _, p_rf, _, _ = ensemble(dataset, "random_forest", "sklearn")
        p_lr = load_prediction(dataset, "logistic", "sklearn", 11)["probability"]
        analysis["uci"][dataset] = {
            "logistic": bootstrap_auc(y, p_lr), "random_forest": bootstrap_auc(y, p_rf), "mlp": bootstrap_auc(y, p_mlp),
            "rf_minus_mlp": bootstrap_auc_difference(y, p_rf, p_mlp),
            "mlp_minus_logistic": bootstrap_auc_difference(y, p_mlp, p_lr)}
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=1, ensure_ascii=False), encoding="utf8")
    for dataset in MARKETS:
        print(dataset, {k: (round(v["auc"], 3), round(v["low"], 3), round(v["high"], 3)) for k, v in analysis["auc_ci"][dataset].items()})
        print("   ", {k: (round(v["annual_return"], 3), round(v["sharpe"], 2), round(v["max_drawdown"], 3), round(v["exposure"], 2))
                    for k, v in analysis["backtest"][dataset].items()})
    for dataset, v in analysis["uci"].items():
        print(dataset, {k: {kk: round(vv, 3) for kk, vv in val.items() if isinstance(vv, float)} for k, val in v.items()})


if __name__ == "__main__":
    main()
