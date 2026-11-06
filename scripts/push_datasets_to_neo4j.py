"""Đẩy một mẫu nhỏ (đều nhau) của từng bộ dữ liệu trong data/ lên Neo4j dưới dạng đồ thị.

Mỗi bộ -> nút nhãn DSNode + nhãn theo loại, thuộc tính `ds` = tên bộ, `uid` = ds|key (duy nhất).
Mẫu ngẫu nhiên cố định seed=42, ~N bản ghi/bộ (mặc định 3000) để tổng nằm gọn trong Aura Free (200k nút/400k cạnh).
Idempotent (MERGE). Kết nối dùng chung load_config() của push_to_neo4j.py (không in mật khẩu).

Chạy: python scripts/push_datasets_to_neo4j.py [--n 3000] [--only ten_bo] [--reset]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_to_neo4j import ROOT, load_config  # noqa: E402

D = ROOT / "data"
BATCH = 1000
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SEED = 42


class Sink:
    """Gom nút/cạnh của một bộ rồi nạp theo lô."""

    def __init__(self, ds: str):
        self.ds, self.nodes, self.edges = ds, {}, []

    def node(self, label: str, key, **props):
        uid = f"{self.ds}|{label}|{key}"
        props = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)) for k, v in props.items()}
        self.nodes.setdefault((label, uid), {"uid": uid, "key": str(key), "props": {k: v for k, v in props.items() if v is not None}})
        return uid

    def edge(self, a: str, rel: str, b: str, **props):
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in props.items()}
        self.edges.append((rel, a, b, props))

    def flush(self, session):
        by_label: dict[str, list] = {}
        for (label, _), r in self.nodes.items():
            by_label.setdefault(label, []).append(r)
        for label, rows in by_label.items():
            assert IDENT.match(label), label
            for i in range(0, len(rows), BATCH):
                session.run(
                    f"UNWIND $rows AS r MERGE (n:DSNode {{uid:r.uid}}) SET n:{label}, n.key=r.key, n.ds=$ds, n += r.props",
                    rows=rows[i : i + BATCH], ds=self.ds).consume()
        by_rel: dict[str, list] = {}
        for rel, a, b, p in self.edges:
            assert IDENT.match(rel), rel
            by_rel.setdefault(rel, []).append({"a": a, "b": b, "p": p})
        for rel, rows in by_rel.items():
            for i in range(0, len(rows), BATCH):
                session.run(
                    f"UNWIND $rows AS r MATCH (a:DSNode {{uid:r.a}}), (b:DSNode {{uid:r.b}}) MERGE (a)-[e:{rel}]->(b) SET e += r.p",
                    rows=rows[i : i + BATCH]).consume()
        return len(self.nodes), len(self.edges)


def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)


def rd(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


# ---------------------------------------------------------------- từng bộ
def agent01_orders(n):
    s = Sink("agent01_orders")
    for r in sample(rd(D / "agent01/orders_large.csv"), n).itertuples():
        o = s.node("Order", r.id, status=r.status, paid=r.paid, age_days=r.age_days, channel=r.channel)
        s.edge(o, "OWNED_BY", s.node("User", r.owner))
        s.edge(o, "IN_TENANT", s.node("Tenant", r.tenant))
    return s


def agent01_intents(n):
    s = Sink("agent01_intents")
    for r in sample(rd(D / "agent01/intent_utterances.csv"), n).itertuples():
        s.edge(s.node("Utterance", r.id, text=r.text), "HAS_INTENT", s.node("Intent", r.intent))
    return s


def agent01_catmouse(n):
    s = Sink("agent01_catmouse")
    for r in sample(rd(D / "agent01/cat_mouse_episodes.csv"), n).itertuples():
        e = s.node("Episode", f"{r.map}-{r.policy}-{r.episode}", caught=r.caught, steps=r.steps)
        s.edge(e, "ON_MAP", s.node("Map", r.map))
        s.edge(e, "USED_POLICY", s.node("Policy", r.policy))
    return s


def agent01_retail_invoices(n):
    s = Sink("agent01_retail_invoices")
    for r in sample(rd(D / "agent01/external/online_retail_invoices.csv"), n).itertuples():
        i = s.node("Invoice", r.invoice, date=r.date, lines=r.lines, items=r.items, total=r.total, cancelled=r.cancelled)
        c = s.node("Customer", r.customer_id)
        s.edge(i, "BY_CUSTOMER", c)
        s.edge(c, "IN_COUNTRY", s.node("Country", r.country))
    return s


def agent02_retail_lines(n):
    s = Sink("agent02_retail_lines")
    df = rd(D / "agent02/external/online_retail_lines.csv", dtype={"invoice": str, "stock_code": str})
    inv = sample(pd.DataFrame({"invoice": df["invoice"].unique()}), max(1, n // 8))["invoice"]
    for r in df[df["invoice"].isin(set(inv))].itertuples():
        i = s.node("Invoice", r.invoice, date=r.invoice_date)
        p = s.node("Product", r.stock_code, description=r.description)
        s.edge(i, "CONTAINS", p, quantity=r.quantity, unit_price=r.unit_price)
        if not pd.isna(r.customer_id):
            c = s.node("Customer", int(r.customer_id))
            s.edge(i, "BY_CUSTOMER", c)
            s.edge(c, "IN_COUNTRY", s.node("Country", r.country))
    return s


def agent02_market(n):
    s = Sink("agent02_market")
    df = rd(D / "agent02/market_steps.csv")
    g = df.groupby(["scenario", "episode"]).agg(steps=("step", "count"), profit_a=("profit_a", "sum"), profit_b=("profit_b", "sum"),
                                                 sales_a=("sales_a", "sum"), sales_b=("sales_b", "sum")).reset_index()
    for scen, part in g.groupby("scenario"):
        for r in sample(part, n // g["scenario"].nunique() + 1).itertuples():
            e = s.node("MarketEpisode", f"{scen}-{r.episode}", episode=r.episode, steps=r.steps, profit_a=r.profit_a,
                       profit_b=r.profit_b, sales_a=r.sales_a, sales_b=r.sales_b)
            s.edge(e, "IN_SCENARIO", s.node("Scenario", scen))
    return s


def agent02_qlearning(n):
    s = Sink("agent02_qlearning")
    df = rd(D / "agent02/q_learning_training_all.csv")
    for r in sample(df, n).itertuples():
        e = s.node("TrainingEpisode", f"{r.config}-{r.seed}-{r.episode}", episode=r.episode, rl_reward=r.rl_episode_reward,
                   rule_reward=r.rule_episode_reward, epsilon=r.epsilon, alpha=r.alpha)
        run = s.node("Run", f"{r.config}-{r.seed}", seed=r.seed)
        s.edge(e, "IN_RUN", run)
        s.edge(run, "USES_CONFIG", s.node("Config", r.config))
    return s


def assignment06_prices(n):
    s = Sink("assignment06_prices")
    for f, name in [("stock_aapl.csv", "AAPL"), ("commodity_gold.csv", "GOLD")]:
        df = rd(D / "assignment06" / f).tail(n // 2)
        a = s.node("Asset", name)
        for r in df.itertuples():
            s.edge(a, "HAS_PRICE", s.node("PriceDay", f"{name}-{r.Date}", date=r.Date, close=r.Close, high=r.High, low=r.Low, volume=r.Volume))
    return s


def chapter05_brfss2023(n):
    s = Sink("chapter05_brfss2023")
    for i, r in enumerate(sample(rd(D / "chapter05/brfss2023_diabetes_150k.csv"), n).itertuples()):
        p = s.node("Respondent", i, bmi=r.bmi, general_health=r.general_health, heart_disease=r.heart_disease, stroke=r.stroke,
                   high_blood_pressure=r.high_blood_pressure, high_cholesterol=r.high_cholesterol, sex=r.sex,
                   income_group=r.income_group, smoker_status=r.smoker_status, education=r.education)
        s.edge(p, "HAS_DIABETES_STATUS", s.node("DiabetesStatus", r.diabetes_status))
        s.edge(p, "IN_AGE_GROUP", s.node("AgeGroup", r.age_group))
        s.edge(p, "HAS_BMI_CATEGORY", s.node("BMICategory", r.bmi_category))
        s.edge(p, "LIVES_IN", s.node("State", r.state))
    return s


def diabetes_brfss2015(n):
    s = Sink("diabetes_brfss2015")
    df = rd(D / "diabetes/diabetes_binary_health_indicators_BRFSS2015.csv")
    parts = [sample(df[df["Diabetes_binary"] == c], n // 2) for c in (0.0, 1.0)]  # cân bằng hai lớp
    for i, r in enumerate(pd.concat(parts).itertuples()):
        p = s.node("Respondent", i, bmi=r.BMI, high_bp=r.HighBP, high_chol=r.HighChol, smoker=r.Smoker, phys_activity=r.PhysActivity,
                   gen_health=r.GenHlth, education=int(r.Education), income=int(r.Income))
        s.edge(p, "HAS_DIABETES_CLASS", s.node("DiabetesClass", "diabetes" if r.Diabetes_binary == 1 else "no_diabetes"))
        s.edge(p, "IN_AGE_BRACKET", s.node("AgeBracket", int(r.Age)))
        s.edge(p, "HAS_SEX", s.node("Sex", "male" if r.Sex == 1 else "female"))
        s.edge(p, "HAS_INCOME", s.node("IncomeLevel", int(r.Income)))
    return s


def diabetes_encounters(n):
    """Bệnh án 130 bệnh viện Mỹ (UCI 296): Patient -HAS_ENCOUNTER-> Encounter -HAS_RECORDED_CODE-> DiagnosisCode."""
    s = Sink("diabetes_encounters")
    df = rd(D / "chapter04/diabetic_data.csv", dtype={"diag_1": str, "diag_2": str, "diag_3": str})
    for r in sample(df, n).itertuples():
        e = s.node("Encounter", r.encounter_id, age=r.age, gender=r.gender, race=r.race, readmitted=r.readmitted,
                   time_in_hospital=r.time_in_hospital, num_medications=r.num_medications, number_diagnoses=r.number_diagnoses,
                   a1c_result=r.A1Cresult, insulin=r.insulin, diabetes_med=r.diabetesMed)
        s.edge(s.node("Patient", r.patient_nbr), "HAS_ENCOUNTER", e)
        for code in dict.fromkeys(c for c in (r.diag_1, r.diag_2, r.diag_3) if isinstance(c, str) and c != "?"):
            s.edge(e, "HAS_RECORDED_CODE", s.node("DiagnosisCode", code))
    return s


def ecommerce_reviews(n):
    s = Sink("ecommerce_reviews")
    df = pd.read_csv(D / "ecommerce/train.csv", nrows=200_000)
    for i, r in enumerate(sample(df, n).itertuples()):
        s.edge(s.node("Review", i, text=str(r.Review)[:400]), "HAS_SENTIMENT", s.node("Sentiment", "positive" if r.Label == 2 else "negative"))
    return s


def house_price_listings(n):
    s = Sink("house_price_listings")
    parts = []
    for ch in pd.read_csv(D / "house_price/realtor-data.zip.csv", chunksize=300_000, dtype={"zip_code": str}):
        parts.append(ch.sample(frac=n / 2_226_382 * 1.1, random_state=SEED))
    for i, r in enumerate(sample(pd.concat(parts), n).itertuples()):
        l = s.node("Listing", i, price=r.price, bed=r.bed, bath=r.bath, house_size=r.house_size, acre_lot=r.acre_lot, status=r.status)
        if isinstance(r.city, str) and isinstance(r.state, str):
            c = s.node("City", f"{r.city}, {r.state}")
            s.edge(l, "IN_CITY", c)
            s.edge(c, "IN_STATE", s.node("State", r.state))
    return s


def tieuluan_markets(n):
    s = Sink("tieuluan_markets")
    raw = D / "tieuluan/raw"
    days = max(20, n // 83)  # 83 tài sản
    for market, folder in [("US_STOCK", "us_stocks"), ("VN_STOCK", "vn_stocks"), ("CRYPTO", "crypto"), ("INDEX", "indices")]:
        m = s.node("Market", market)
        for f in sorted((raw / folder).glob("*.csv")):
            a = s.node("Asset", f.stem, market=market)
            s.edge(a, "TRADED_IN", m)
            for r in rd(f).tail(days).itertuples():
                s.edge(a, "HAS_PRICE", s.node("PriceDay", f"{f.stem}-{r.date}", date=r.date, close=r.close, volume=r.volume))
    return s


def tieuluan_lending(n):
    s = Sink("tieuluan_lending")
    for i, r in enumerate(sample(rd(D / "tieuluan/raw/lending_club/loans_36m_2012_2015_150k.csv"), n).itertuples()):
        l = s.node("Loan", i, amount=r.loan_amnt, int_rate=r.int_rate, annual_inc=r.annual_inc, dti=r.dti, fico=r.fico, issue_date=r.issue_d, defaulted=r.default)
        s.edge(l, "HAS_GRADE", s.node("Grade", r.grade))
        s.edge(l, "HAS_PURPOSE", s.node("Purpose", r.purpose))
        s.edge(l, "HAS_STATUS", s.node("LoanStatus", r.loan_status))
        s.edge(l, "HOME_OWNERSHIP", s.node("HomeOwnership", r.home_ownership))
    return s


def tieuluan_bankruptcy(n):
    s = Sink("tieuluan_bankruptcy")
    df = rd(D / "tieuluan/raw/uci/taiwan_bankruptcy/data.csv")
    df.columns = [c.strip() for c in df.columns]
    parts = [sample(df[df["Bankrupt?"] == c], n // 2 if c == 0 else min(n // 2, int((df["Bankrupt?"] == 1).sum()))) for c in (0, 1)]
    keep = ["ROA(C) before interest and depreciation before interest", "Debt ratio %", "Net Income to Total Assets", "Current Ratio"]
    for i, r in pd.concat(parts).reset_index(drop=True).iterrows():
        c = s.node("Company", i, **{re.sub(r"\W+", "_", k).strip("_").lower(): r[k] for k in keep if k in df.columns})
        s.edge(c, "HAS_OUTCOME", s.node("Outcome", "bankrupt" if r["Bankrupt?"] == 1 else "healthy"))
    return s


def image_set(ds, path, names, n):
    s = Sink(ds)
    d = np.load(path)
    x, y = d["x_train"], d["y_train"]
    idx = np.random.default_rng(SEED).choice(len(y), min(n, len(y)), replace=False)
    for i in idx:
        img = x[i]
        s.edge(s.node("Image", int(i), mean_intensity=round(float(img.mean()), 2), ink_pixels=int((img > 127).sum())),
               "HAS_CLASS", s.node("Class", int(y[i]), name=names[int(y[i])]))
    return s


def mnist(n):
    return image_set("mnist", D / "mnist/mnist.npz", [str(i) for i in range(10)], n)


def fashion(n):
    names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return image_set("fashion_mnist", D / "fashion_mnist/fashion_mnist.npz", names, n)


DATASETS = {f.__name__: f for f in [chapter05_brfss2023, diabetes_brfss2015, diabetes_encounters]}  # chỉ giữ dữ liệu tiểu đường
# cỡ mẫu mặc định (~60% Aura Free: giới hạn cạnh 400k chạm trước)
DEFAULT_N = {"chapter05_brfss2023": 10000, "diabetes_brfss2015": 16000, "diabetes_encounters": 34000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="ghi đè cỡ mẫu mọi bộ")
    ap.add_argument("--only")
    ap.add_argument("--reset", action="store_true", help="xóa các nút DSNode của bộ được chọn trước khi nạp")
    a = ap.parse_args()
    cfg = load_config()
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    driver.verify_connectivity()
    with driver.session(database=cfg["database"]) as ses:
        ses.run("CREATE CONSTRAINT ds_uid IF NOT EXISTS FOR (n:DSNode) REQUIRE n.uid IS UNIQUE").consume()
        for name, fn in DATASETS.items():
            if a.only and a.only != name:
                continue
            sink = fn(a.n or DEFAULT_N.get(name, 3000))
            if a.reset:
                ses.run("MATCH (n:DSNode {ds:$d}) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS", d=sink.ds).consume()
            nn, ne = sink.flush(ses)
            print(f"  {sink.ds}: {nn} nút, {ne} cạnh", flush=True)
        print("\n== Tổng trên Neo4j ==")
        for r in ses.run("MATCH (n:DSNode) RETURN n.ds AS ds, count(n) AS c ORDER BY ds"):
            print(f"  {r['ds']}: {r['c']} nút")
        print("  TỔNG nút:", ses.run("MATCH (n) RETURN count(n) AS c").single()["c"],
              "| cạnh:", ses.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"], "(giới hạn Aura Free 200k/400k)")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
