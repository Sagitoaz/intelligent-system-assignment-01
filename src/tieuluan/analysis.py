"""Đánh giá bổ sung tính từ dự báo đã lưu (không huấn luyện lại).

1. Khoảng tin cậy bootstrap cho ROC-AUC.
   - Dữ liệu bảng (mỗi dòng một doanh nghiệp/khách hàng, coi như độc lập): bootstrap phân tầng.
   - Chuỗi thị trường (các ngày liền kề phụ thuộc nhau): bootstrap theo khối trượt
     (moving block bootstrap, Künsch 1989) với khối 20 phiên ≈ một tháng giao dịch.
2. Backtest minh họa chiến lược "nắm giữ hoặc đứng ngoài" (long/flat) có tính phí giao dịch.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(np.flatnonzero(y == c), size=int((y == c).sum()), replace=True) for c in (0, 1)]
    return np.concatenate(parts)


def bootstrap_auc(y, p, *, block: int | None = None, n_boot: int = 2000, seed: int = 2026) -> dict:
    """AUC, khoảng tin cậy 95% (phân vị 2,5–97,5) và p-value một phía cho giả thuyết AUC ≤ 0,5."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _block_indices(len(y), block, rng) if block else _stratified_indices(y, rng)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    stats = np.asarray(stats)
    auc = float(roc_auc_score(y, p))
    return {"auc": auc, "low": float(np.quantile(stats, .025)), "high": float(np.quantile(stats, .975)),
            "p_value_auc_le_half": float((1 + np.sum(stats <= .5)) / (1 + len(stats))), "n_boot": int(len(stats)),
            "method": f"moving block bootstrap, block={block}" if block else "stratified bootstrap"}


def bootstrap_auc_difference(y, p_a, p_b, *, n_boot: int = 2000, seed: int = 2026) -> dict:
    """Bootstrap phân tầng có ghép cặp cho hiệu AUC(a) − AUC(b) trên cùng các mẫu test."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _stratified_indices(y, rng)
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    return {"difference": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "low": float(np.quantile(diffs, .025)), "high": float(np.quantile(diffs, .975))}


def backtest(position, returns, *, cost_per_side: float, periods_per_year: int) -> dict:
    """Mô phỏng nắm giữ (position = 1) hoặc đứng ngoài (0) cho từng kỳ t → t+1.

    Quyết định đưa ra sau khi biết giá đóng cửa ngày t và được giả định khớp đúng giá đó
    (giả định lạc quan). Mỗi lần đổi trạng thái trả phí ``cost_per_side`` trên giá trị giao dịch;
    lần mua đầu tiên cũng tính phí. Lãi suất phi rủi ro bằng 0 khi tính Sharpe.
    """
    position = np.asarray(position, dtype=float)
    returns = np.asarray(returns, dtype=float)
    previous = np.r_[0.0, position[:-1]]
    trades = np.abs(position - previous)
    net = position * returns - cost_per_side * trades
    equity = np.cumprod(1 + net)
    years = len(net) / periods_per_year
    sd = net.std(ddof=1)
    drawdown = 1 - equity / np.maximum.accumulate(equity)
    return {"annual_return": float(equity[-1] ** (1 / years) - 1), "annual_volatility": float(sd * np.sqrt(periods_per_year)),
            "sharpe": float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
            "max_drawdown": float(drawdown.max()), "exposure": float(position.mean()),
            "entries": int(((position == 1) & (previous == 0)).sum()), "total_cost": float((cost_per_side * trades).sum()),
            "final_wealth": float(equity[-1]), "equity": equity}
