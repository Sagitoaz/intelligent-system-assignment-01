"""Đánh giá độ tin cậy của agent Deep Learning trên đúng tập test (50.736 dòng) và ghi artifacts/trust_metrics.json.

Chạy: .venv\\Scripts\\python.exe project01/train/eval_trust.py   (chỉ cần numpy, pandas, scikit-learn; không cần torch)
Phép chia train/test giống hệt train_dl.py (test_size=0,2, random_state=42, stratify theo nhãn).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from project01.mas.dl_agent import DLAgent  # noqa: E402
from project01.mas.data import FEATURES  # noqa: E402

CSV = os.path.join(ROOT, "data", "diabetes", "diabetes_012_health_indicators_BRFSS2015.csv")
OUT = os.path.join(ROOT, "project01", "artifacts", "trust_metrics.json")
SEED = 42
CLS = ["none", "pre", "diabetes"]
CLS_VI = ["Không tiểu đường", "Tiền tiểu đường", "Tiểu đường"]
HIGH = 0.25  # ngưỡng "nguy cơ cao" của orchestrator.level


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k, n):
    return dict(k=int(k), n=int(n), value=round(k / n, 4) if n else None, ci=wilson(k, n))


def confusion(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def per_class(cm):
    out = []
    for k in range(3):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k].sum() if cm[k].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append(dict(id=CLS[k], name=CLS_VI[k], support=int(cm[k].sum()), predicted=int(cm[:, k].sum()), precision=round(float(prec), 4),
                        recall=round(float(rec), 4), f1=round(float(f1), 4)))
    return out


def reliability(p, truth, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        n = int(m.sum())
        if n:
            mp, fr = float(p[m].mean()), float(truth[m].mean())
            ece += n / len(p) * abs(mp - fr)
        else:
            mp = fr = None
        rows.append(dict(lo=round(float(edges[i]), 2), hi=round(float(edges[i + 1]), 2), n=n,
                         mean_pred=None if mp is None else round(mp, 4), observed=None if fr is None else round(fr, 4)))
    return dict(bins=rows, ece=round(float(ece), 4), brier=round(float(((p - truth) ** 2).mean()), 4),
                base_rate=round(float(truth.mean()), 4))


def alert_stats(score, y, thr):
    a = score >= thr
    n_alert = int(a.sum())
    tp_dia = int((a & (y == 2)).sum())
    tp_any = int((a & (y >= 1)).sum())
    return dict(threshold=thr, alerts=n_alert, alert_rate=round(n_alert / len(y), 4),
                precision_diabetes=rate(tp_dia, n_alert), recall_diabetes=rate(tp_dia, int((y == 2).sum())),
                false_alarm_rate=rate(int((a & (y == 0)).sum()), int((y == 0).sum())),
                false_alerts_among_alerts=rate(int((a & (y == 0)).sum()), n_alert),
                precision_any=rate(tp_any, n_alert), recall_any=rate(tp_any, int((y >= 1).sum())),
                recall_pre=rate(int((a & (y == 1)).sum()), int((y == 1).sum())))


def group_table(mask_list, y, pred, score):
    rows = []
    for name, m in mask_list:
        n = int(m.sum())
        yy, pp, ss = y[m], pred[m], score[m]
        dia = yy == 2
        pre = yy == 1
        rows.append(dict(
            group=name, n=n, diabetes_n=int(dia.sum()), pre_n=int(pre.sum()),
            accuracy=rate(int((yy == pp).sum()), n),
            recall_diabetes=rate(int((dia & (pp == 2)).sum()), int(dia.sum())),
            recall_pre=rate(int((pre & (pp == 1)).sum()), int(pre.sum())),
            alert_rate=rate(int((ss >= HIGH).sum()), n),
            alert_recall_diabetes=rate(int((dia & (ss >= HIGH)).sum()), int(dia.sum())),
            alert_precision=rate(int(((ss >= HIGH) & dia).sum()), int((ss >= HIGH).sum())),
            diabetes_prevalence=rate(int(dia.sum()), n)))
    return rows


def spread(rows, key):
    vals = [(r[key]["value"], r["group"]) for r in rows if r[key]["value"] is not None and r[key]["n"] >= 30]
    hi, lo = max(vals), min(vals)
    return dict(max_gap=round(hi[0] - lo[0], 4), highest=hi[1], lowest=lo[1], highest_value=hi[0], lowest_value=lo[0])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(CSV)
    y_all = df["Diabetes_012"].astype(int).values
    X = df[FEATURES].astype("float32").values
    _, Xte, _, yte = train_test_split(X, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
    agent = DLAgent()
    probs = agent._forward(Xte)
    pred = (probs * agent.scale).argmax(1)
    score = DLAgent.index(probs)
    n = len(yte)
    cm = confusion(yte, pred)
    acc = float((pred == yte).mean())
    base_acc = float((yte == 0).mean())
    pc = per_class(cm)
    recall_pre = pc[1]["recall"]

    # hiệu chuẩn
    calib = dict(diabetes=reliability(probs[:, 2], (yte == 2).astype(float)), pre=reliability(probs[:, 1], (yte == 1).astype(float)),
                 index=reliability(np.clip(score, 0, 1), ((yte == 2) | (yte == 1)).astype(float)))

    alerts = dict(main=alert_stats(score, yte, HIGH), loose=alert_stats(score, yte, 0.10))

    # nhóm
    age = X_age = Xte[:, FEATURES.index("Age")].astype(int)
    sex = Xte[:, FEATURES.index("Sex")].astype(int)
    inc = Xte[:, FEATURES.index("Income")].astype(int)
    age_groups = [("18–34 tuổi", (age >= 1) & (age <= 3)), ("35–49 tuổi", (age >= 4) & (age <= 6)), ("50–59 tuổi", (age >= 7) & (age <= 8)),
                  ("60–69 tuổi", (age >= 9) & (age <= 10)), ("70 tuổi trở lên", age >= 11)]
    sex_groups = [("Nữ", sex == 0), ("Nam", sex == 1)]
    inc_groups = [("Thu nhập thấp (dưới 25 nghìn USD/năm)", inc <= 4), ("Thu nhập trung bình (25–50 nghìn)", (inc >= 5) & (inc <= 6)),
                  ("Thu nhập cao (từ 50 nghìn)", inc >= 7)]
    groups = {}
    for key, title, lst in (("age", "Nhóm tuổi", age_groups), ("sex", "Giới tính", sex_groups), ("income", "Thu nhập", inc_groups)):
        rows = group_table(lst, yte, pred, score)
        assert sum(r["n"] for r in rows) == n, key
        groups[key] = dict(title=title, rows=rows,
                           gaps={k: spread(rows, k) for k in ("accuracy", "recall_diabetes", "alert_rate", "alert_recall_diabetes")})

    # kết luận bằng lời, sinh từ số thật
    def pc_(v):
        return f"{v * 100:.1f}".replace(".", ",") + "%"
    concl = []
    for key in ("age", "sex", "income"):
        g = groups[key]
        r = g["gaps"]["recall_diabetes"]
        concl.append(f"{g['title']}: nhóm bị bỏ sót nhiều nhất là «{r['lowest']}» (recall tiểu đường {pc_(r['lowest_value'])}), "
                     f"nhóm được phát hiện tốt nhất là «{r['highest']}» ({pc_(r['highest_value'])}); chênh lệch {str(round(r['max_gap'] * 100, 1)).replace(".", ",")} điểm phần trăm.")
        a = g["gaps"]["alert_rate"]
        concl.append(f"{g['title']}: tỉ lệ bị cảnh báo «nguy cơ cao» dao động từ {pc_(a['lowest_value'])} («{a['lowest']}») đến {pc_(a['highest_value'])} («{a['highest']}»).")
    concl.append("Khoảng tin cậy Wilson 95% cho biết độ chắc chắn của từng con số; nhóm có khoảng rộng (ít mẫu) thì chênh lệch có thể chỉ là ngẫu nhiên.")
    concl.append("Chênh lệch giữa các nhóm một phần phản ánh tỉ lệ bệnh khác nhau thật sự giữa các nhóm, không hoàn toàn do mô hình thiên vị; "
                 "nhưng nhóm nào recall thấp hơn thì người thuộc nhóm đó dễ bị bỏ sót hơn.")

    cnt = np.bincount(y_all, minlength=3)
    limits = [
        f"Tiền tiểu đường chỉ chiếm {pc_(cnt[1] / len(y_all))} dữ liệu ({format(int(cnt[1]), ",").replace(",", ".")} hồ sơ) nên mô hình gần như không học được nhóm này; recall chỉ {pc_(recall_pre)}.",
        f"Độ chính xác tổng {pc_(acc)} dễ gây hiểu lầm: luôn đoán «không tiểu đường» cũng đúng {pc_(base_acc)} số hồ sơ.",
        "Dữ liệu BRFSS là khảo sát qua điện thoại do người dân tự khai (cân nặng, thói quen, bệnh nền), không có xét nghiệm, nên nhãn và đặc trưng đều có sai số.",
        "Mô hình KHÔNG dùng đường huyết, A1C hay bất kỳ kết quả xét nghiệm nào; nó chỉ ước lượng nguy cơ từ lối sống và bệnh nền, không thể chẩn đoán.",
        "Chỉ gồm dân số Mỹ năm 2015; chưa kiểm chứng trên người Việt hay các nhóm dân tộc khác nhau, nên chưa nên dùng để quyết định lâm sàng.",
        "Nhãn là «từng được bác sĩ chẩn đoán»: người bệnh chưa được phát hiện bị tính là «không bệnh», làm số liệu độ chính xác bị lạc quan hoặc lệch.",
        "Hệ số điều chỉnh ngưỡng (decision_scale) được chọn để cân bằng macro-F1 trên tập validation; nó tăng recall nhưng làm tăng cảnh báo sai.",
        "Tập test cùng nguồn với tập huấn luyện (chia ngẫu nhiên), nên số liệu thực tế ở nơi khác có thể thấp hơn.",
    ]

    result = dict(
        test_rows=int(n), class_counts_test={CLS[k]: int((yte == k).sum()) for k in range(3)}, decision_scale=[float(v) for v in agent.scale],
        accuracy=round(acc, 4), accuracy_ci=wilson(int((pred == yte).sum()), n), baseline_always_none=dict(accuracy=round(base_acc, 4), recall_pre=0.0, recall_diabetes=0.0),
        per_class=pc, confusion_matrix=cm.tolist(), confusion_labels=CLS_VI,
        macro_f1=round(float(np.mean([c["f1"] for c in pc])), 4),
        pre_warning=f"Recall nhóm tiền tiểu đường chỉ {pc_(recall_pre)}: hơn {int(round((1 - recall_pre) * 100))}% người tiền tiểu đường bị bỏ sót.",
        calibration=calib, alerts=alerts, groups=groups,
        fairness=dict(max_gaps={k: {gk: groups[k]["gaps"][gk] for gk in groups[k]["gaps"]} for k in groups}, conclusions=concl,
                      note="Chỉ xét nhóm từ 30 mẫu trở lên."),
        limitations=limits)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("accuracy", result["accuracy"], "baseline", result["baseline_always_none"]["accuracy"])
    print("recall", [c["recall"] for c in pc], "precision", [c["precision"] for c in pc])
    print("ECE dia/pre", calib["diabetes"]["ece"], calib["pre"]["ece"], "Brier", calib["diabetes"]["brier"], calib["pre"]["brier"])
    for k in ("main", "loose"):
        a = alerts[k]
        print(k, a["threshold"], "alerts", a["alerts"], "prec_dia", a["precision_diabetes"]["value"], "recall_dia", a["recall_diabetes"]["value"],
              "false_alarm", a["false_alarm_rate"]["value"], "prec_any", a["precision_any"]["value"], "recall_pre", a["recall_pre"]["value"])
    for k in groups:
        print(k, {g: v for g, v in groups[k]["gaps"].items()})
    print("\n".join(concl))


if __name__ == "__main__":
    main()
