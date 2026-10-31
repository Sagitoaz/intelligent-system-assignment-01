"""Chương 6 (Deep Graph): GNN trên đồ thị cổ phiếu VN30.

Bài toán: mỗi ngày giao dịch, mỗi cổ phiếu là một nút; dự đoán cổ phiếu có lợi suất 5 phiên tới cao hơn trung vị của rổ hay không.
So sánh: Logistic Regression, MLP (không dùng đồ thị), GCN trên đồ thị tương quan, GCN trên đồ thị ngành, GCN trên đồ thị ngẫu nhiên (đối chứng).
Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.complete_gnn   ->  data/tieuluan/gnn/results.json (+ hình, đồ thị)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "tieuluan", "raw", "vn_stocks")
OUT = os.path.join(ROOT, "data", "tieuluan", "gnn")
os.makedirs(OUT, exist_ok=True)

SECTORS = {  # phân ngành gán thủ công theo hoạt động chính (không lấy từ nguồn dữ liệu)
    "Ngân hàng": ["ACB", "BID", "CTG", "HDB", "LPB", "MBB", "SHB", "STB", "TCB", "VCB", "VIB", "VPB"],
    "Bất động sản": ["VHM", "VIC", "VRE"],
    "Tiêu dùng": ["MCH", "MSN", "MWG", "SAB", "VNM"],
    "Dầu khí - năng lượng": ["BSR", "GAS"],
    "Chứng khoán": ["SSI"],
    "Công nghệ": ["FPT"],
    "Vật liệu": ["HPG", "GVR"],
    "Hàng không": ["VJC"],
}
HORIZON = 5
LAGS = (1, 2, 3, 5, 10, 20)
SPLIT = dict(train_end="2022-12-31", val_end="2023-12-31")
SEEDS = (0, 1, 2, 3, 4)


def load_panel():
    """Giá đóng cửa và khối lượng của các mã có dữ liệu liên tục từ 2018-06."""
    close, vol = {}, {}
    for sec, ts in SECTORS.items():
        for t in ts:
            d = pd.read_csv(os.path.join(RAW, t + ".csv"), parse_dates=["date"]).drop_duplicates("date").set_index("date").sort_index()
            close[t], vol[t] = d["close"], d["volume"]
    close, vol = pd.DataFrame(close), pd.DataFrame(vol)
    start = max(c.first_valid_index() for _, c in close.items())
    close, vol = close.loc[start:].ffill(limit=3), vol.loc[start:].fillna(0)
    close = close.dropna(axis=0, how="any")
    return close, vol.loc[close.index]


def build_features(close, vol):
    ret = np.log(close).diff()
    feats = []
    for k in LAGS:
        r = np.log(close).diff(k)
        feats.append(r.sub(r.mean(axis=1), axis=0))  # trừ trung bình thị trường: chỉ giữ phần chênh lệch giữa các mã
    feats.append(ret.rolling(20).std())
    feats.append((vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9))
    X = np.stack([f.values for f in feats], axis=-1)  # (ngày, nút, đặc trưng)
    fwd = np.log(close).shift(-HORIZON) - np.log(close)
    y = (fwd.sub(fwd.median(axis=1), axis=0) > 0).astype(float).values
    ok = ~np.isnan(X).any(axis=(1, 2)) & ~fwd.isna().any(axis=1).values
    dates = close.index
    return X[ok], y[ok], fwd.values[ok], dates[ok]


def norm_adj(a):
    a = a + np.eye(len(a))
    d = a.sum(1)
    return (a / np.sqrt(d[:, None] * d[None, :])).astype(np.float32)


def graphs(close, train_mask, tickers, seed=0):
    ret = np.log(close).diff().loc[train_mask].dropna()
    corr = np.corrcoef(ret.values.T)
    np.fill_diagonal(corr, -1)
    n, k = len(tickers), 4
    a_corr = np.zeros((n, n))
    for i in range(n):
        for j in np.argsort(-corr[i])[:k]:
            a_corr[i, j] = a_corr[j, i] = 1
    a_sec = np.zeros((n, n))
    idx = {t: i for i, t in enumerate(tickers)}
    for ts in SECTORS.values():
        for u in ts:
            for v in ts:
                if u != v:
                    a_sec[idx[u], idx[v]] = 1
    rng = np.random.default_rng(seed)
    a_rand = np.zeros((n, n))
    m = int(a_corr.sum() / 2)
    while a_rand.sum() / 2 < m:
        i, j = rng.integers(0, n, 2)
        if i != j:
            a_rand[i, j] = a_rand[j, i] = 1
    return dict(corr=a_corr, sector=a_sec, random=a_rand), corr


class GCN(nn.Module):
    """Hai lớp GCN dạng ma trận kề chuẩn hóa A_hat: H' = ReLU(A_hat H W), rồi lớp tuyến tính ra logit cho từng nút."""

    def __init__(self, a_hat, f_in, hid=32, drop=0.3):
        super().__init__()
        self.register_buffer("A", torch.tensor(a_hat))
        self.w1, self.w2 = nn.Linear(f_in, hid), nn.Linear(hid, hid)
        self.out, self.drop = nn.Linear(hid, 1), nn.Dropout(drop)

    def forward(self, x):  # x: (lô ngày, nút, đặc trưng)
        h = torch.relu(self.A @ self.w1(x))
        h = self.drop(h)
        h = torch.relu(self.A @ self.w2(h))
        return self.out(self.drop(h)).squeeze(-1)


class NodeMLP(nn.Module):
    """Cùng cỡ với GCN nhưng không trộn thông tin giữa các nút (A_hat = I)."""

    def __init__(self, f_in, hid=32, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f_in, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y.ravel(), p.ravel()))


def fit(model, Xtr, ytr, Xva, yva, seed, epochs=80, lr=2e-3, wd=1e-3, bs=64):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, hist = -1, None, []
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).numpy()
        a = auc(yva, pv)
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(xt), val_auc=a))
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return hist, best


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def metrics(y, p, fwd):
    pred = p > 0.5
    acc = float((pred == (y > 0.5)).mean())
    # IC: tương quan hạng giữa điểm dự đoán và lợi suất 5 phiên, trung bình theo ngày
    from scipy.stats import spearmanr
    ic = float(np.nanmean([spearmanr(p[i], fwd[i])[0] for i in range(len(p))]))
    # chọn 5 mã điểm cao nhất mỗi ngày: lợi suất 5 phiên trung bình so với trung bình rổ
    top = np.argsort(-p, axis=1)[:, :5]
    spread = float(np.mean([fwd[i, top[i]].mean() - fwd[i].mean() for i in range(len(p))]))
    return dict(accuracy=acc, auc=auc(y, p), ic=ic, top5_excess=spread)


def run():
    t0 = time.time()
    close, vol = load_panel()
    tickers = list(close.columns)
    X, y, fwd, dates = build_features(close, vol)
    tr = dates <= pd.Timestamp(SPLIT["train_end"])
    va = (dates > pd.Timestamp(SPLIT["train_end"])) & (dates <= pd.Timestamp(SPLIT["val_end"]))
    te = dates > pd.Timestamp(SPLIT["val_end"])
    # bỏ HORIZON ngày ở ranh giới để nhãn không nhìn sang tập sau
    idx = np.arange(len(dates))
    tr_i, va_i, te_i = idx[tr][:-HORIZON], idx[va][:-HORIZON], idx[te]
    mu, sd = X[tr_i].reshape(-1, X.shape[-1]).mean(0), X[tr_i].reshape(-1, X.shape[-1]).std(0) + 1e-9
    Xn = ((X - mu) / sd).astype(np.float32)
    A, corr = graphs(close, close.index <= pd.Timestamp(SPLIT["train_end"]), tickers)
    info = dict(tickers=tickers, n_days=int(len(dates)), n_features=int(X.shape[-1]), split={k: int(v) for k, v in dict(train=len(tr_i), val=len(va_i), test=len(te_i)).items()},
                first=str(dates[0].date()), last=str(dates[-1].date()), edges={k: int(v.sum() / 2) for k, v in A.items()}, horizon=HORIZON,
                positive_rate_test=float(y[te_i].mean()), splits_dates=dict(train_end=SPLIT["train_end"], val_end=SPLIT["val_end"]))
    models = {
        "Logistic Regression": None,
        "MLP (không đồ thị)": lambda s: NodeMLP(X.shape[-1]),
        "GCN, đồ thị ngẫu nhiên": lambda s: GCN(norm_adj(graphs(close, close.index <= pd.Timestamp(SPLIT["train_end"]), tickers, seed=s)[0]["random"]), X.shape[-1]),
        "GCN, đồ thị ngành": lambda s: GCN(norm_adj(A["sector"]), X.shape[-1]),
        "GCN, đồ thị tương quan": lambda s: GCN(norm_adj(A["corr"]), X.shape[-1]),
    }
    results, curves = {}, {}
    from sklearn.linear_model import LogisticRegression
    for name, make in models.items():
        runs = []
        for s in SEEDS:
            if make is None:
                lr = LogisticRegression(max_iter=500, C=0.5).fit(Xn[tr_i].reshape(-1, X.shape[-1]), y[tr_i].ravel())
                p = lr.predict_proba(Xn[te_i].reshape(-1, X.shape[-1]))[:, 1].reshape(len(te_i), -1)
                runs.append(metrics(y[te_i], p, fwd[te_i]))
                continue
            model = make(s)
            hist, best = fit(model, Xn[tr_i], y[tr_i], Xn[va_i], y[va_i], seed=s)
            if s == 0:
                curves[name] = hist
            runs.append(metrics(y[te_i], predict(model, Xn[te_i]), fwd[te_i]) | dict(val_auc=best))
            print(name, "seed", s, {k: round(v, 4) for k, v in runs[-1].items()}, flush=True)
        agg = {k: dict(mean=float(np.mean([r[k] for r in runs])), std=float(np.std([r[k] for r in runs]))) for k in runs[0]}
        results[name] = agg
    # đỉnh của đồ thị tương quan: nút có bậc cao nhất, dùng cho hình
    deg = A["corr"].sum(1)
    info["corr_degree"] = {t: int(d) for t, d in zip(tickers, deg)}
    info["seconds"] = round(time.time() - t0, 1)
    out = dict(info=info, results=results, curves=curves, seeds=list(SEEDS), config=dict(hidden=32, layers=2, dropout=0.3, lr=2e-3, weight_decay=1e-3, epochs=80, batch=64, knn=4))
    json.dump(out, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez(os.path.join(OUT, "graphs.npz"), corr=A["corr"], sector=A["sector"], random=A["random"], corr_matrix=corr, tickers=np.array(tickers))
    out["sector_task"] = run_sector()
    json.dump(out, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("xong", info["seconds"], "giây")


def sector_data():
    """Dữ liệu bài toán B: đặc trưng riêng của từng mã (chuẩn hóa), nhãn Ngân hàng hay không, và ma trận tương quan lúc huấn luyện."""
    close, vol = load_panel()
    tickers = list(close.columns)
    ret = np.log(close).diff().dropna()
    tr = ret.loc[: SPLIT["train_end"]]
    mkt = tr.mean(axis=1)
    f = np.stack([tr.std().values, [np.cov(tr[t], mkt)[0, 1] / mkt.var() for t in tickers], tr.skew().values, tr.kurt().values,
                  np.log1p(vol.loc[tr.index].mean().values), [tr[t].autocorr(1) for t in tickers]], axis=1)
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    label = np.array([int(t in SECTORS["Ngân hàng"]) for t in tickers])
    return close, tickers, f, label, np.corrcoef(tr.values.T)


def run_sector():
    """Bài toán B (phân loại nút, bán giám sát): chỉ biết ngành của 1/3 số mã, đoán ngành (Ngân hàng hay không) cho phần còn lại.
    Đặc trưng nút chỉ là thống kê riêng của từng mã (độ biến động, beta, khối lượng...), không chứa thông tin về hàng xóm."""
    from sklearn.linear_model import LogisticRegression
    close, tickers, f, label, _ = sector_data()
    A, _ = graphs(close, close.index <= pd.Timestamp(SPLIT["train_end"]), tickers)
    n = len(tickers)
    rows = {k: [] for k in ("Logistic Regression", "MLP (không đồ thị)", "GCN, đồ thị ngẫu nhiên", "GCN, đồ thị tương quan", "GCN, đồ thị ngành (tham khảo)")}
    homophily = float(np.mean([label[i] == label[j] for i in range(n) for j in range(n) if A["corr"][i, j]]))
    for rep in range(40):
        rng = np.random.default_rng(rep)
        tr_idx = np.concatenate([rng.choice(np.where(label == c)[0], size=max(2, int(round((label == c).sum() / 3))), replace=False) for c in (0, 1)])
        te_idx = np.setdiff1d(np.arange(n), tr_idx)
        x = torch.tensor(f, dtype=torch.float32)[None]
        yt = torch.tensor(label, dtype=torch.float32)

        def train_eval(model):
            torch.manual_seed(rep)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
            for _ in range(150):
                model.train()
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(x)[0][tr_idx], yt[tr_idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                return float(((torch.sigmoid(model(x)[0][te_idx]) > 0.5).float() == yt[te_idx]).float().mean())

        rows["Logistic Regression"].append(float((LogisticRegression(max_iter=300).fit(f[tr_idx], label[tr_idx]).predict(f[te_idx]) == label[te_idx]).mean()))
        rows["MLP (không đồ thị)"].append(train_eval(NodeMLP(f.shape[1], hid=16, drop=0.2)))
        ar = graphs(close, close.index <= pd.Timestamp(SPLIT["train_end"]), tickers, seed=rep)[0]["random"]
        rows["GCN, đồ thị ngẫu nhiên"].append(train_eval(GCN(norm_adj(ar), f.shape[1], hid=16, drop=0.2)))
        rows["GCN, đồ thị tương quan"].append(train_eval(GCN(norm_adj(A["corr"]), f.shape[1], hid=16, drop=0.2)))
        rows["GCN, đồ thị ngành (tham khảo)"].append(train_eval(GCN(norm_adj(A["sector"]), f.shape[1], hid=16, drop=0.2)))
    res = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in rows.items()}
    res["_per_split"] = {k: [round(x, 4) for x in v] for k, v in rows.items()}
    res["_info"] = dict(nodes=n, labelled=int(len(tr_idx)), repeats=40, majority_rate=float(max(label.mean(), 1 - label.mean())), corr_edge_homophily=homophily, features=["độ biến động", "beta", "độ lệch", "độ nhọn", "log khối lượng", "tự tương quan bậc 1"])
    return res


if __name__ == "__main__":
    run()
