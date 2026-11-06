"""Bác sĩ trưởng (agent điều phối): chạy lần lượt các agent chuyên môn, gom kết quả thành kịch bản hội chẩn và kết luận.

================================ BẢN ĐỒ FILE (đọc phần này trước) ================================
Đây là TRUNG TÂM của hệ đa agent (Multi-Agent System) Project 01: class Clinic nối tất cả agent chuyên môn lại với nhau.
Endpoint: POST /api/consult -> Clinic.consult(profile)  (server.py dòng đăng ký "/api/consult"); GET /api/meta -> Clinic.meta().
Giao diện: static/app.js (phòng khám 3D static/scene.js) phát lại danh sách "events" theo từng lượt nói của từng agent.
Bảy agent (mô tả trong AGENTS bên dưới) và file cài đặt:
    predictor (Deep Learning MLP)    -> dl_agent.py      (DLAgent.predict: xác suất 3 lớp, yếu tố tác động, what-if)
    knowledge (KB + RAG)             -> kb.py            (KnowledgeBase: đồ thị tri thức + trích đoạn NIDDK)
    coach     (Reinforcement Learning)-> rl.py           (QCoach: Q-learning gợi ý ăn uống/vận động trong 24 giờ mô phỏng)
    guardian  (luật rule-based)      -> rules.py         (rules.evaluate: cờ đỏ, luôn ưu tiên)
    critic    (kiểm tra chéo/tranh luận) -> critic.py    (critic.review: đối chiếu MLP với luật, số đo, tài liệu; tối đa 2 vòng)
    reporter  (hình + bảng)          -> report.py        (build_report)
    doctor    (điều phối)            -> chính file này
Dữ liệu: không đọc file trực tiếp; các agent con tự nạp artifacts/ (dl_model.npz, kg.json, rl_qtable.json...). Không ghi file.
Các hàm chính của Clinic:
    __init__()    : nạp DLAgent, KnowledgeBase, QCoach (làm một lần khi khởi động server)
    meta()        : thông tin chung cho giao diện (danh sách agent, hồ sơ mẫu, độ chính xác mô hình, cảnh báo)
    patient()     : chuẩn hóa hồ sơ -> dự đoán MLP -> dựng "bệnh nhân ảo" cho RL (trả prof, dl, pat)
    kb_nodes()    : chọn các nút đồ thị tri thức liên quan tới hồ sơ này (để tra cứu và trích dẫn)
    level()       : mức nguy cơ thấp/trung bình/cao từ risk_index, có luật an toàn kéo mức lên
    consult()     : LUỒNG HỘI CHẨN đầy đủ (xem chú thích từng bước trong hàm)
Thầy hỏi "điều phối nhiều agent ở đâu?" -> Clinic.consult() (chuỗi add("doctor","handoff",...) mời từng agent theo thứ tự).
Thầy hỏi "mức nguy cơ cuối cùng quyết định thế nào?" -> Clinic.level() rồi critic.review()["final_level"] trong consult().
================================================================================================
"""
from __future__ import annotations

from . import rl, rules
from . import critic as critic_mod
from .data import SYMPTOMS, bmi, normalize
from .dl_agent import DLAgent
from .kb import REL_VI, KnowledgeBase
from .report import build_report

# Danh sách 7 agent hiển thị trên giao diện (id dùng làm khóa ở events; color là màu bong bóng thoại; role cho biết kỹ thuật AI).
AGENTS = [
    dict(id="doctor", name="Bác sĩ trưởng", role="Điều phối", color="#2589e0", blurb="Nhận bệnh nhân, mời từng chuyên gia, gom ý kiến và đưa ra tư vấn cuối cùng."),
    dict(id="predictor", name="Bác sĩ dữ liệu", role="Deep Learning", color="#7060d8", blurb="Mạng nơ-ron MLP học từ 253.680 hồ sơ khảo sát sức khỏe BRFSS để ước lượng nguy cơ."),
    dict(id="knowledge", name="Thủ thư tri thức", role="KB + RAG", color="#0d9f8f", blurb="Tra đồ thị tri thức tiểu đường và trích dẫn nguyên văn tài liệu của NIDDK."),
    dict(id="coach", name="Huấn luyện viên", role="Reinforcement Learning", color="#f0603a", blurb="Học bằng thử và sai cách ăn uống, vận động giữ đường huyết trong ngưỡng an toàn."),
    dict(id="guardian", name="Hộ lý an toàn", role="Luật (rule-based)", color="#e8a317", blurb="Kiểm tra các dấu hiệu nguy hiểm theo luật cố định và luôn được ưu tiên."),
    dict(id="critic", name="Bác sĩ phản biện", role="Kiểm tra chéo", color="#8a5cf0", blurb="Đối chiếu kết luận của Bác sĩ dữ liệu với luật an toàn, số đo thật và tài liệu; nếu mâu thuẫn thì tranh luận tối đa 2 vòng."),
    dict(id="reporter", name="Thư ký báo cáo", role="Sinh hình và bảng", color="#e8508a", blurb="Tự vẽ biểu đồ và lập bảng tóm tắt cho hồ sơ vừa khám."),
]
# Màu theo mức nguy cơ (thấp/trung bình/cao) cho thẻ kết quả.
LEVELS = {"thấp": "#0d9f8f", "trung bình": "#e8a317", "cao": "#e8503a"}
# Ánh xạ mã triệu chứng của hồ sơ -> id nút trong đồ thị tri thức (kg.json) để tra KB.
SYM_NODE = {"thirst": "thirst", "frequent_urination": "urination", "fatigue": "fatigue", "blurred_vision": "blurred", "slow_healing": "slow_heal", "weight_loss": "weight_loss"}
SYM_VI = dict(SYMPTOMS)


class Clinic:
    """Phòng khám AI: giữ sẵn các agent (dl, kb, coach) và thực hiện hội chẩn. Một thể hiện duy nhất (CLINIC trong server.py)."""
    def __init__(self):
        """Nạp mô hình MLP (artifacts/dl_model.npz), đồ thị tri thức (kg.json + corpus) và bảng Q của huấn luyện viên RL (rl_qtable.json)."""
        self.dl, self.kb, self.coach = DLAgent(), KnowledgeBase(), rl.QCoach.load()

    # ---- thông tin chung
    def meta(self):
        """Trả về thông tin chung cho giao diện: agents, personas (hồ sơ mẫu), fields (ô nhập), model (kiến trúc + độ chính xác trên tập kiểm tra), disclaimer."""
        from .data import PERSONAS, fields
        m = self.dl.metrics
        return dict(agents=AGENTS, personas=PERSONAS, fields=fields(),
                    model=dict(name="MLP 3 lớp ẩn (21→128→64→32→3)", dataset=m["dataset"], rows=m["rows"], train_rows=m["train_rows"], test_rows=m["test_rows"],
                               accuracy=round(m["accuracy"], 4), macro_f1=round(m["macro_f1"], 4), auc_ovr=round(m["auc_ovr"], 4)),
                    disclaimer="Chỉ để học tập: hệ thống không thay thế chẩn đoán hay tư vấn của bác sĩ. Mô phỏng 24 giờ là mô hình đồ chơi, không phải y khoa thật.")

    def patient(self, profile: dict) -> tuple[dict, dict, dict]:
        """Từ hồ sơ thô: (prof đã chuẩn hóa, dl = kết quả MLP gồm risk/risk_index/bmi/drivers/what_if, pat = bệnh nhân ảo cho môi trường RL).
        Đây là cầu nối Deep Learning -> Reinforcement Learning: chỉ số nguy cơ do MLP tính được dùng để chọn độ nhạy insulin của bệnh nhân ảo."""
        prof = normalize(profile)
        dl = self.dl.predict(prof)
        pat = rl.patient_from_risk(dl["risk_index"], dl["bmi"], prof["fasting_glucose"], prof["name"])
        return prof, dl, pat

    # ---- hội chẩn
    def kb_nodes(self, prof: dict, dl: dict) -> list[str]:
        """Chọn id các nút đồ thị tri thức liên quan hồ sơ (huyết áp, mỡ máu, thừa cân, ít vận động, hút thuốc, tuổi, triệu chứng, nhóm bệnh...), bỏ trùng và nút không có trong KG.
        Danh sách này là "truy vấn" cho KB agent: dùng để lấy sự kiện (facts) và đoạn trích NIDDK (passages_for)."""
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
        """Mức nguy cơ tiếng Việt: risk_index <0,10 thấp; <0,25 trung bình; còn lại cao. Luật an toàn có quyền ưu tiên:
        cờ đỏ "danger" -> luôn "cao"; cờ "warn" mà mô hình bảo "thấp" -> nâng lên "trung bình" (an toàn hơn là bỏ sót)."""
        r = dl["risk_index"]
        lv = "thấp" if r < 0.10 else "trung bình" if r < 0.25 else "cao"
        top = rules.top_level(rule_list)
        if top == "danger":
            return "cao"
        if top == "warn" and lv == "thấp":
            return "trung bình"
        return lv

    def consult(self, profile: dict) -> dict:
        """LUỒNG HỘI CHẨN. Đầu vào: hồ sơ bệnh nhân (dict). Đầu ra: dict(events=kịch bản lượt nói, summary=kết luận, report=hình + bảng).
        Thứ tự: bác sĩ trưởng chào -> MLP ước lượng -> KB tra cứu -> luật an toàn -> critic phản biện (tối đa 2 vòng) -> RL gợi ý một ngày -> reporter -> kết luận.
        Mỗi bước ghi một "event" {agent, type: speak|handoff|tool|cite|alert|debate|final, text, ms thời lượng hiển thị, payload}, giao diện chỉ việc phát lại."""
        prof, dl, pat = self.patient(profile)
        name = prof["name"]
        pct = lambda v: f"{v * 100:.0f}%" if v >= 0.1 else f"{v * 100:.1f}%"
        risk = dl["risk"]
        rule_list = rules.evaluate(prof, dl["risk_index"], risk["diabetes"])
        # Mức nguy cơ cơ sở (MLP + luật) -> critic kiểm tra chéo; nếu mâu thuẫn với luật/số đo thì tranh luận và có thể NÂNG mức (final_level).
        base_level = self.level(dl, rule_list)
        review = critic_mod.review(self, prof, dl, rule_list, base_level)
        level = review["final_level"]
        # KB agent: chọn nút liên quan -> lấy tối đa 3 đoạn tài liệu NIDDK gắn với các nút đó (Graph-RAG) -> lấy các cạnh "nguyên nhân/triệu chứng/chẩn đoán" làm sự kiện.
        nodes = self.kb_nodes(prof, dl)
        passages = self.kb.passages_for(nodes[:10], 3)
        facts = []
        for e in self.kb.edges:
            if e["source"] in nodes and e["target"] in nodes and e["rel"] in ("raises_risk", "symptom_of", "diagnosed_by"):
                facts.append(f"{self.kb.nodes[e['source']]['vi']} — {REL_VI[e['rel']]} — {self.kb.nodes[e['target']]['vi']}")
        # RL coach: kế hoạch ngày (seed cố định để tái lập) + so sánh 3 kiểu (RL, luật, không huấn luyện viên) trên 20 ngày mô phỏng.
        plan = rl.plan_for(pat, self.coach, seed=3)
        cmp = rl.compare(pat, self.coach, days=20, seed=7)
        by = {c["id"]: c for c in cmp["coaches"]}
        drivers_up = [d for d in dl["drivers"] if d["effect"] > 0.005][:3]
        gains = [w for w in dl["what_if"] if w["gain"] >= 0.005][:3]

        # Từ đây dựng kịch bản: ev là danh sách sự kiện; add() thêm một sự kiện, thời lượng mặc định tỉ lệ độ dài câu (>=1,8 giây).
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
        add("doctor", "handoff", "Bác sĩ phản biện, nhờ bạn kiểm tra chéo kết luận của Bác sĩ dữ liệu.", 1500, to="critic")
        # Phần phản biện: phát lại các lượt tranh luận mà critic.review() đã sinh (kiểm tra chéo, trích tài liệu, đồng ý/phản đối, kết luận vòng).
        for e in review["events"]:
            extra = {k: e[k] for k in ("round", "stance", "reply_to") if k in e}
            payload = dict(checks=review["checks"], outcome=review["outcome"]) if e is review["events"][0] else None
            if e["type"] == "cite" and review["quote"]:
                payload = dict(passages=[review["quote"]])
            add(e["agent"], e["type"], e["text"], focus=e["agent"], payload=payload, **extra)
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

        # Kết luận cuối: tiêu đề theo mức nguy cơ + danh sách lời khuyên (luật nguy hiểm/lưu ý, what-if giảm nguy cơ, kế hoạch RL, nhắc xét nghiệm A1C).
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
        critique = dict(checks=review["checks"], outcome=review["outcome"], rounds=review["rounds"], mlp_level=review["mlp_level"], base_level=base_level,
                        final_level=level, escalated=review["escalated"], quote=review["quote"])
        summary = dict(risk=risk, level=level, critique=critique, level_color=LEVELS[level], headline=headline, advice=advice, red_flags=[dict(r) for r in rule_list if r["level"] != "info"],
                       citations=citations, kg_nodes=nodes, label=dl["label"], risk_index=round(dl["risk_index"], 4), bmi=round(dl["bmi"], 1))
        # Thư ký báo cáo: sinh 3 hình matplotlib (base64 PNG) + 7 bảng; giao diện hiển thị và /api/report/pdf (pdf_report.py) dùng lại để xuất PDF.
        report = build_report(prof, dl, rule_list, plan, cmp, passages, self.dl.population_risk_index(), self.dl.metrics)
        return dict(events=ev, summary=summary, report=report)
