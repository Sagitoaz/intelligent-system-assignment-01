"""Bác sĩ trưởng (agent điều phối): chạy lần lượt các agent chuyên môn, gom kết quả thành kịch bản hội chẩn và kết luận."""
from __future__ import annotations

from . import rl, rules
from .data import SYMPTOMS, bmi, normalize
from .dl_agent import DLAgent
from .kb import REL_VI, KnowledgeBase
from .report import build_report

AGENTS = [
    dict(id="doctor", name="Bác sĩ trưởng", role="Điều phối", color="#2589e0", blurb="Nhận bệnh nhân, mời từng chuyên gia, gom ý kiến và đưa ra tư vấn cuối cùng."),
    dict(id="predictor", name="Bác sĩ dữ liệu", role="Deep Learning", color="#7060d8", blurb="Mạng nơ-ron MLP học từ 253.680 hồ sơ khảo sát sức khỏe BRFSS để ước lượng nguy cơ."),
    dict(id="knowledge", name="Thủ thư tri thức", role="KB + RAG", color="#0d9f8f", blurb="Tra đồ thị tri thức tiểu đường và trích dẫn nguyên văn tài liệu của NIDDK."),
    dict(id="coach", name="Huấn luyện viên", role="Reinforcement Learning", color="#f0603a", blurb="Học bằng thử và sai cách ăn uống, vận động giữ đường huyết trong ngưỡng an toàn."),
    dict(id="guardian", name="Hộ lý an toàn", role="Luật (rule-based)", color="#e8a317", blurb="Kiểm tra các dấu hiệu nguy hiểm theo luật cố định và luôn được ưu tiên."),
    dict(id="reporter", name="Thư ký báo cáo", role="Sinh hình và bảng", color="#e8508a", blurb="Tự vẽ biểu đồ và lập bảng tóm tắt cho hồ sơ vừa khám."),
]
LEVELS = {"thấp": "#0d9f8f", "trung bình": "#e8a317", "cao": "#e8503a"}
SYM_NODE = {"thirst": "thirst", "frequent_urination": "urination", "fatigue": "fatigue", "blurred_vision": "blurred", "slow_healing": "slow_heal", "weight_loss": "weight_loss"}
SYM_VI = dict(SYMPTOMS)


class Clinic:
    def __init__(self):
        self.dl, self.kb, self.coach = DLAgent(), KnowledgeBase(), rl.QCoach.load()

    # ---- thông tin chung
    def meta(self):
        from .data import PERSONAS, fields
        m = self.dl.metrics
        return dict(agents=AGENTS, personas=PERSONAS, fields=fields(),
                    model=dict(name="MLP 3 lớp ẩn (21→128→64→32→3)", dataset=m["dataset"], rows=m["rows"], train_rows=m["train_rows"], test_rows=m["test_rows"],
                               accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4), auc_ovr=round(m["auc_ovr"], 4)),
                    disclaimer="Chỉ để học tập: hệ thống không thay thế chẩn đoán hay tư vấn của bác sĩ. Mô phỏng 24 giờ là mô hình đồ chơi, không phải y khoa thật.")

    def patient(self, profile: dict) -> tuple[dict, dict, dict]:
        prof = normalize(profile)
        dl = self.dl.predict(prof)
        pat = rl.patient_from_risk(dl["risk_index"], dl["bmi"], prof["fasting_glucose"], prof["name"])
        return prof, dl, pat

    # ---- hội chẩn
    def kb_nodes(self, prof: dict, dl: dict) -> list[str]:
        ids = []
        if prof["high_bp"]: ids.append("high_bp")
        if prof["high_chol"]: ids.append("high_chol")
        if dl["bmi"] >= 25: ids.append("overweight")
        if not prof["phys_activity"]: ids.append("inactivity")
        if prof["smoker"]: ids.append("smoking")
        if prof["age_years"] >= 35: ids.append("age")
        ids += [SYM_NODE[s] for s in prof["symptoms"]]
        if prof["fasting_glucose"] is not None:
            ids.append("fpg")
        top = dl["risk"]
        ids += ["diabetes"] + (["prediabetes"] if top["pre"] >= 0.04 else []) + (["type2"] if top["diabetes"] >= 0.2 else []) + ["a1c", "screening"]
        if not prof["phys_activity"] or dl["bmi"] >= 25:
            ids += ["activity", "healthy_eating"]
        seen = []
        for i in ids:
            if i in self.kb.nodes and i not in seen:
                seen.append(i)
        return seen

    def level(self, dl: dict, rule_list: list[dict]) -> str:
        r = dl["risk_index"]
        lv = "thấp" if r < 0.10 else "trung bình" if r < 0.25 else "cao"
        top = rules.top_level(rule_list)
        if top == "danger":
            return "cao"
        if top == "warn" and lv == "thấp":
            return "trung bình"
        return lv

    def consult(self, profile: dict) -> dict:
        prof, dl, pat = self.patient(profile)
        name = prof["name"]
        pct = lambda v: f"{v * 100:.0f}%" if v >= 0.1 else f"{v * 100:.1f}%"
        risk = dl["risk"]
        rule_list = rules.evaluate(prof, dl["risk_index"], risk["diabetes"])
        level = self.level(dl, rule_list)
        nodes = self.kb_nodes(prof, dl)
        passages = self.kb.passages_for(nodes[:10], 3)
        facts = []
        for e in self.kb.edges:
            if e["source"] in nodes and e["target"] in nodes and e["rel"] in ("raises_risk", "symptom_of", "diagnosed_by"):
                facts.append(f"{self.kb.nodes[e['source']]['vi']} — {REL_VI[e['rel']]} — {self.kb.nodes[e['target']]['vi']}")
        plan = rl.plan_for(pat, self.coach, seed=3)
        cmp = rl.compare(pat, self.coach, days=20, seed=7)
        by = {c["id"]: c for c in cmp["coaches"]}
        drivers_up = [d for d in dl["drivers"] if d["effect"] > 0.005][:3]
        gains = [w for w in dl["what_if"] if w["gain"] >= 0.005][:3]

        ev = []
        add = lambda agent, typ, text, ms=None, **kw: ev.append(dict(i=len(ev), agent=agent, type=typ, text=text, ms=ms or max(1800, 38 * len(text)), **kw))
        add("doctor", "speak", f"Chào {name}! Hồ sơ của bạn: {prof['age_years']} tuổi, {'nam' if prof['sex']=='male' else 'nữ'}, BMI {dl['bmi']:.1f}"
            f"{', đường huyết đói ' + format(prof['fasting_glucose'], '.0f') + ' mg/dL' if prof['fasting_glucose'] else ''}. Mình sẽ mời từng chuyên gia xem hồ sơ nhé.", focus="doctor")
        add("doctor", "handoff", "Mời Bác sĩ dữ liệu ước lượng nguy cơ trước.", 1500, to="predictor")
        sentence = f"Mô hình học từ 253.680 hồ sơ ước lượng: không tiểu đường {pct(risk['none'])}, tiền tiểu đường {pct(risk['pre'])}, tiểu đường {pct(risk['diabetes'])}."
        if drivers_up:
            sentence += " Yếu tố đẩy nguy cơ lên nhiều nhất: " + ", ".join(f"{d['vi']} (+{d['effect']*100:.0f} điểm %)" for d in drivers_up) + "."
        add("predictor", "tool", sentence, focus="predictor", payload=dict(risk=risk, drivers=dl["drivers"], what_if=dl["what_if"], label=dl["label"]))
        add("doctor", "handoff", "Cảm ơn! Thủ thư tri thức hãy tra cứu giúp tài liệu liên quan.", 1500, to="knowledge")
        top_p = passages[0] if passages else None
        know = f"Mình đối chiếu {len(nodes)} khái niệm trong đồ thị tri thức." + (f" Tài liệu NIDDK nói: “{top_p['quote'][:170].rstrip()}…”" if top_p else "")
        add("knowledge", "cite", know, 3600, focus="knowledge", payload=dict(kg_nodes=nodes, facts=facts[:6], passages=[dict(id=p["id"], quote=p["quote"], url=p["url"]) for p in passages]))
        add("doctor", "handoff", "Hộ lý an toàn hãy kiểm tra các dấu hiệu nguy hiểm.", 1500, to="guardian")
        if rule_list:
            for r in rule_list:
                add("guardian", "alert", f"[{r['id']}] {r['text']}", 3400, level=r["level"], focus="guardian", payload=dict(basis=r["basis"]))
        else:
            add("guardian", "speak", "Không có luật cảnh báo nào được kích hoạt với hồ sơ này. Mình vẫn nhắc: đây không phải chẩn đoán.", focus="guardian")
        add("doctor", "handoff", "Huấn luyện viên hãy gợi ý một ngày sinh hoạt hợp lý.", 1500, to="coach")
        if plan:
            plan_txt = ", ".join(f"{p['hour']}h {p['label'].lower()}" for p in plan[:6])
            coach_text = (f"Mình đã luyện bằng thử và sai trên hàng chục nghìn ngày mô phỏng. Với bạn, mình gợi ý: {plan_txt}. "
                          f"Trong mô phỏng, đường huyết nằm trong ngưỡng 70–180 mg/dL khoảng {by['rl']['time_in_range']*100:.0f}% thời gian, "
                          f"so với {by['none']['time_in_range']*100:.0f}% nếu không có huấn luyện viên.")
        else:
            coach_text = "Hôm nay mình chưa có gợi ý đặc biệt, hãy duy trì thói quen hiện tại."
        add("coach", "speak", coach_text, 4200, focus="coach", payload=dict(plan=plan, compare=[{k: c[k] for k in ("id", "name", "time_in_range", "mean_glucose")} for c in cmp["coaches"]]))
        add("doctor", "handoff", "Thư ký báo cáo, nhờ bạn tổng hợp hình và bảng.", 1400, to="reporter")
        add("reporter", "tool", "Mình vẽ biểu đồ nguy cơ, các yếu tố tác động, một ngày mô phỏng và lập các bảng tóm tắt.", 2200, focus="reporter")

        headline = {"cao": "Nguy cơ cao: nên đi xét nghiệm xác nhận sớm.", "trung bình": "Nguy cơ trung bình: nên điều chỉnh lối sống và tầm soát định kỳ.",
                    "thấp": "Nguy cơ thấp: hãy duy trì lối sống lành mạnh."}[level]
        advice = []
        advice += [r["text"] for r in rule_list if r["level"] in ("danger", "warn")][:2]
        for w in gains:
            advice.append(f"{w['action'][0].upper() + w['action'][1:]}: mô hình cho thấy có thể giảm khoảng {w['gain']*100:.0f} điểm phần trăm chỉ số nguy cơ.")
        if plan:
            advice.append("Theo huấn luyện viên RL: " + ", ".join(f"{p['hour']}h {p['emoji']} {p['label'].lower()}" for p in plan[:5]) + ".")
        advice.append("Nên làm xét nghiệm A1C hoặc đường huyết đói tại cơ sở y tế để có kết luận chính xác." if level != "thấp" else "Tầm soát định kỳ theo khuyến cáo của bác sĩ, kể cả khi kết quả bình thường.")
        final = f"{headline} " + (advice[0] if advice else "")
        add("doctor", "final", f"Tóm tắt cho {name}: {headline} Mình đã ghi lại lời khuyên và báo cáo bên dưới. Đây chỉ là công cụ học tập, không thay bác sĩ.", 3600, focus="doctor")

        citations = [dict(id=p["id"], quote=p["quote"], url=p["url"]) for p in passages]
        summary = dict(risk=risk, level=level, level_color=LEVELS[level], headline=headline, advice=advice, red_flags=[dict(r) for r in rule_list if r["level"] != "info"],
                       citations=citations, kg_nodes=nodes, label=dl["label"], risk_index=round(dl["risk_index"], 4), bmi=round(dl["bmi"], 1))
        report = build_report(prof, dl, rule_list, plan, cmp, passages, self.dl.population_risk_index(), self.dl.metrics)
        return dict(events=ev, summary=summary, report=report)
