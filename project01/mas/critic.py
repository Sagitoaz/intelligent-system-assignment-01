"""Bác sĩ phản biện: kiểm tra chéo kết luận của Bác sĩ dữ liệu (MLP) bằng bằng chứng chạy được.

Ba nguồn bằng chứng độc lập với nhau:
  1. luật an toàn (rules.py) và các ngưỡng xét nghiệm của ADA/NIDDK mà luật dựa vào,
  2. số liệu thật của hồ sơ (đường huyết, triệu chứng) và kiểm tra độ nhạy của chính mô hình (đổi cân nặng ±5 kg, đo xem nhãn có đổi không),
  3. đoạn trích nguyên văn từ kho tri thức (kb.py) có nguồn.
Mô hình MLP không dùng đường huyết làm đầu vào, nên khi đường huyết đo trực tiếp mâu thuẫn với nguy cơ ước lượng thì đây là bất đồng thật,
không phải mô hình tự đồng ý với chính nó. Tranh luận tối đa 2 vòng; kết luận cuối ưu tiên bằng chứng đo trực tiếp.

=====================  BẢN ĐỒ FILE (thầy hỏi "bác sĩ phản biện / critic agent / tranh luận giữa các agent ở đâu" thì mở file này)  =====================
Thuộc agent: "Bác sĩ phản biện" (id="critic", vai trò Kiểm tra chéo). Không học máy: là bộ kiểm tra dựa trên bằng chứng chạy được.

Nó làm 4 phép kiểm tra (checks) trên kết luận của MLP, mỗi phép trả status:
  C1 đường huyết đo trực tiếp  ↔ mức nguy cơ MLP : "conflict" nếu đường huyết ≥ 126 mà MLP không báo "cao", hoặc 100–125 mà MLP báo "thấp";
                                                    "soft" nếu MLP báo "cao" nhưng đường huyết < 100.
  C2 triệu chứng điển hình     ↔ mức nguy cơ MLP : "conflict" nếu ≥ 2 triệu chứng điển hình mà MLP báo "thấp" và đường huyết chưa ≥ 126.
  C3 độ chắc chắn của MLP                         : "soft" nếu hai lớp xác suất cao nhất chỉ cách nhau < 10 điểm phần trăm.
  C4 độ nhạy với sai số đo                        : "soft" nếu đổi cân nặng ±5 kg làm mức nguy cơ đổi (kết luận sát ranh giới).
  Quy ước: "conflict" = bất đồng THẬT (có sàn nâng mức, floor); "soft" = nghi vấn nhẹ (chỉ ghi chú); "ok"; "na" = không đủ dữ liệu.
  Mức nguy cơ từ risk_index: < 0,10 thấp; < 0,25 trung bình; còn lại cao (hàm _lv).
  Kết luận cuối: final_level = mức cao nhất trong {mức cơ sở (MLP + luật), sàn (floor) của mọi "conflict"} → nếu có xung đột thật thì mức bị NÂNG, ưu tiên số đo trực tiếp.

Thành phần chính (tìm theo tên):
  - _lv(), _pct(), _trim()  : đổi risk_index → thấp/trung bình/cao, định dạng phần trăm kiểu Việt (dấu phẩy), cắt chuỗi gọn.
  - _stability()            : chạy lại MLP với cân nặng ±5 kg (phép thử C4).
  - _corpus_bm25(), _quote(): nạp BM25 trên corpus 492 đoạn (data/chapter04/diabetes_corpus.jsonl), lấy đoạn trích nguyên văn làm bằng chứng.
  - review()                : HÀM CHÍNH — chạy C1–C4, tính final_level, chọn đoạn trích, rồi gọi _debate().
  - _debate()               : dựng kịch bản tranh luận giữa critic / predictor / guardian / knowledge, tối đa MAX_ROUNDS = 2 vòng.
                              Kết quả (outcome): "agree" (không mâu thuẫn), "noted" (chỉ nghi vấn nhẹ, 1 vòng), "escalated" (nâng mức), "kept" (giữ mức).

Dữ liệu đọc: data/chapter04/diabetes_corpus.jsonl (corpus 492 đoạn; khác corpus 183 đoạn của kb.py). Không ghi file.
Được gọi từ: mas/orchestrator.py (Clinic.consult gọi critic_mod.review(...) sau khi có kết quả MLP và luật; events của _debate được phát lại thành hội thoại trên giao diện).
=====================================================================================================================================================
"""
from __future__ import annotations

import re

from .data import SYMPTOMS, normalize

SYM = dict(SYMPTOMS)
LEVEL_ORDER = {"thấp": 0, "trung bình": 1, "cao": 2}  # để so sánh/chọn "mức cao nhất" giữa các mức bằng chữ
MAX_ROUNDS = 2  # số vòng tranh luận tối đa giữa critic và predictor


def _lv(risk_index: float) -> str:
    """Đổi risk_index (P(tiểu đường) + 0,5·P(tiền tiểu đường)) thành mức: < 0,10 "thấp"; < 0,25 "trung bình"; còn lại "cao"."""
    return "thấp" if risk_index < 0.10 else "trung bình" if risk_index < 0.25 else "cao"


def _pct(v: float) -> str:
    """Định dạng xác suất v∈[0,1] thành phần trăm kiểu Việt: ≥ 10% làm tròn nguyên ("35%"), nhỏ hơn thì 1 chữ số thập phân dùng dấu phẩy ("4,5%")."""
    return f"{v * 100:.0f}%" if v >= 0.1 else f"{v * 100:.1f}%".replace(".", ",")


def _stability(dl_agent, prof: dict) -> dict:
    """Phép thử C4: đổi cân nặng ±5 kg (sai số đo thực tế), xem mức nguy cơ của mô hình có đổi không.
    Trả về dict(base = mức gốc, levels = {−5: mức, +5: mức}, flips = danh sách độ lệch làm mức đổi)."""
    base = _lv(dl_agent.predict(prof)["risk_index"])
    levels = {}
    for d in (-5, 5):
        p = dict(prof, weight_kg=max(35.0, prof["weight_kg"] + d))
        levels[d] = _lv(dl_agent.predict(p)["risk_index"])
    return dict(base=base, levels=levels, flips=[d for d, lv in levels.items() if lv != base])


_CORPUS: dict = {}  # bộ nhớ đệm để chỉ dựng chỉ mục BM25 một lần


def _corpus_bm25():
    """BM25 trên kho 492 đoạn NIDDK/CDC/MedlinePlus (data/chapter04/diabetes_corpus.jsonl), nạp một lần."""
    if "bm25" not in _CORPUS:
        import json
        import os
        from .kb import BM25, ROOT
        rows = []
        for line in open(os.path.join(ROOT, "data", "chapter04", "diabetes_corpus.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r.get("language") == "en" and len(r["text"]) >= 80:
                rows.append(dict(id=r["id"], text=r["text"].strip(), url=r["source_url"]))
        _CORPUS["bm25"] = BM25(rows)
    return _CORPUS["bm25"]


def _quote(query: str, must: tuple[str, ...]) -> dict | None:
    """Đoạn trích gần nhất với truy vấn trong kho 492 đoạn; ưu tiên đoạn có chứa từ khóa cần chứng minh. Bỏ tiêu đề mục ở đầu đoạn."""
    res = _corpus_bm25().search(query, 5)
    pick = next((p for p in res if any(m in p["quote"].lower() for m in must)), res[0] if res else None)
    if not pick:
        return None
    body = pick["quote"].rsplit(" — ", 1)[-1]
    return dict(id=pick["id"], quote=body, url=pick["url"])


def _trim(text: str, n: int = 150) -> str:
    """Gộp khoảng trắng và cắt văn bản còn khoảng n ký tự, cắt ở ranh giới từ và thêm "…" để câu trích gọn trong khung hội thoại."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + "…"


def review(clinic, prof: dict, dl: dict, rule_list: list[dict], base_level: str) -> dict:
    """HÀM CHÍNH của Bác sĩ phản biện. Đầu vào: clinic (để chạy lại MLP), hồ sơ, kết quả MLP `dl` (risk, risk_index), danh sách luật đã kích hoạt,
    base_level (mức cơ sở từ MLP + luật). Đầu ra dict: checks (C1–C4), conflicts/hard (id xung đột nhẹ/thật), mlp_level, base_level, final_level,
    escalated (có bị nâng mức không), stability, quote (đoạn trích bằng chứng), events/rounds/outcome (kịch bản tranh luận từ _debate)."""
    prof = normalize(prof)
    g = prof["fasting_glucose"]
    sym = [s for s in ("thirst", "frequent_urination", "weight_loss", "blurred_vision") if s in prof["symptoms"]]
    ri, risk = dl["risk_index"], dl["risk"]
    mlp_level = _lv(ri)
    ordered = sorted(risk.items(), key=lambda kv: -kv[1])
    margin = ordered[0][1] - ordered[1][1]  # khoảng cách giữa hai lớp xác suất cao nhất (dùng cho C3)
    stab = _stability(clinic.dl, prof)
    recall_pre = clinic.dl.metrics["recall_per_class"][1]
    checks, conflicts = [], []  # checks: mọi phép thử; conflicts: các phép thử có status "conflict" hoặc "soft"

    def add(cid, title, status, evidence, source, **kw):
        c = dict(id=cid, title=title, status=status, evidence=evidence, source=source, **kw)
        checks.append(c)
        if status in ("conflict", "soft"):
            conflicts.append(c)

    # --- C1: đường huyết đo trực tiếp so với nguy cơ của mô hình
    if g is None:
        add("C1", "Đường huyết so với mô hình", "na", "Chưa có đường huyết đói để đối chiếu; mô hình cũng không dùng chỉ số này.", "Hồ sơ")
    elif g >= 126 and mlp_level != "cao":
        add("C1", "Đường huyết so với mô hình", "conflict",
            f"Đường huyết đói {g:.0f} mg/dL ≥ 126 (ngưỡng tiểu đường của ADA) nhưng mô hình ước lượng nguy cơ {mlp_level} ({_pct(ri)}).",
            "Luật " + "/".join(r["id"] for r in rule_list if r["id"] in ("R1", "R2", "R3")) + " + ADA", floor="cao", kind="glucose_high")
    elif 100 <= g < 126 and mlp_level == "thấp":
        add("C1", "Đường huyết so với mô hình", "conflict",
            f"Đường huyết đói {g:.0f} mg/dL thuộc vùng tiền tiểu đường (100–125) nhưng mô hình ước lượng nguy cơ thấp ({_pct(ri)}). "
            f"Mô hình cũng chỉ bắt được {recall_pre * 100:.0f}% ca tiền tiểu đường trong tập kiểm thử.", "Luật R4 + dl_metrics", floor="trung bình", kind="glucose_pre")
    elif g < 100 and mlp_level == "cao":
        add("C1", "Đường huyết so với mô hình", "soft",
            f"Mô hình ước lượng nguy cơ cao ({_pct(ri)}) nhưng đường huyết đói {g:.0f} mg/dL đang bình thường.", "Hồ sơ", floor=None, kind="glucose_ok")
    else:
        add("C1", "Đường huyết so với mô hình", "ok", f"Đường huyết đói {g:.0f} mg/dL cùng chiều với nguy cơ {mlp_level} của mô hình.", "Luật + mô hình")

    # --- C2: triệu chứng điển hình so với mô hình
    if len(sym) >= 2 and mlp_level == "thấp" and (g is None or g < 126):
        names = ", ".join(SYM[s].lower() for s in sym)
        add("C2", "Triệu chứng so với mô hình", "conflict", f"Có {len(sym)} triệu chứng điển hình ({names}) nhưng mô hình báo nguy cơ thấp; khảo sát BRFSS không ghi triệu chứng.",
            "NIDDK: triệu chứng điển hình nên xét nghiệm", floor="trung bình", kind="symptoms")
    else:
        add("C2", "Triệu chứng so với mô hình", "ok", "Triệu chứng điển hình không mâu thuẫn với mô hình." if sym else "Không có triệu chứng điển hình.", "Hồ sơ")

    # --- C3: độ chắc chắn của mô hình
    if margin < 0.10:
        add("C3", "Độ chắc chắn của mô hình", "soft", f"Hai nhóm đứng đầu chỉ cách nhau {margin * 100:.0f} điểm phần trăm: mô hình không chắc chắn.", "Xác suất của mô hình", floor=None, kind="margin")
    else:
        add("C3", "Độ chắc chắn của mô hình", "ok", f"Nhóm đứng đầu hơn nhóm thứ hai {margin * 100:.0f} điểm phần trăm.", "Xác suất của mô hình")

    # --- C4: độ nhạy với sai số đo (kiểm tra chạy thật trên mô hình)
    if stab["flips"]:
        add("C4", "Độ nhạy với sai số đo", "soft",
            f"Chỉ cần cân nặng lệch ±5 kg là mức nguy cơ đổi ({stab['base']} → " + ", ".join(f"{stab['levels'][d]} khi {d:+d} kg" for d in stab["flips"]) + "): kết luận nằm sát ranh giới.",
            "Chạy lại mô hình", floor=None, kind="stability")
    else:
        add("C4", "Độ nhạy với sai số đo", "ok", f"Đổi cân nặng ±5 kg vẫn giữ mức nguy cơ {stab['base']}.", "Chạy lại mô hình")

    # Xung đột THẬT (hard) → nâng mức: final_level = mức cao nhất trong {base_level, floor của từng xung đột thật}. Xung đột nhẹ (soft) không nâng mức.
    hard = [c for c in conflicts if c["status"] == "conflict"]
    floor = max([c["floor"] for c in hard if c.get("floor")] + [base_level], key=lambda x: LEVEL_ORDER[x])
    quote = None  # đoạn tài liệu nguyên văn làm bằng chứng; chỉ tìm khi có xung đột thật. must = từ khóa phải có trong đoạn được chọn.
    if hard:
        kind = hard[0]["kind"]
        q, must = {"glucose_high": ("fasting plasma glucose test diagnose diabetes", ("fasting plasma glucose",)),
                   "glucose_pre": ("diagnose prediabetes blood tests fasting glucose", ("prediabetes",)),
                   "symptoms": ("symptoms of diabetes increased thirst frequent urination blurred vision", ("thirst", "urinat"))}[kind]
        quote = _quote(q, must)

    out = dict(checks=checks, conflicts=[c["id"] for c in conflicts], hard=[c["id"] for c in hard], mlp_level=mlp_level, base_level=base_level,
               final_level=floor, escalated=LEVEL_ORDER[floor] > LEVEL_ORDER[base_level], stability=stab,
               quote=dict(id=quote["id"], quote=quote["quote"], url=quote["url"]) if quote else None)
    out["events"], out["rounds"], out["outcome"] = _debate(prof, dl, rule_list, out, quote)
    return out


def _debate(prof, dl, rule_list, rv, quote):
    """Dựng kịch bản tranh luận (tối đa MAX_ROUNDS vòng) từ các bằng chứng đã tính ở review().
    Mỗi lượt nói là một event dict(agent, type, text, round, stance, reply_to). Trả về (events, số vòng, outcome).
    Ba nhánh: không có nghi vấn → 0 vòng "agree"; chỉ nghi vấn nhẹ → 1 vòng "noted"; có xung đột thật → 2 vòng
    (vòng 1: critic phản biện – predictor bảo vệ; vòng 2: guardian đưa luật, knowledge trích tài liệu, critic kết luận) → "escalated" hoặc "kept".
    Lưu ý: đây là kịch bản dựng từ bằng chứng đã tính (template), không phải hai mô hình ngôn ngữ nói chuyện với nhau."""
    ev = []

    def say(agent, text, typ="debate", **kw):
        ev.append(dict(agent=agent, type=typ, text=text, **kw))

    risk, mlp = dl["risk"], rv["mlp_level"]  # xác suất 3 lớp và mức nguy cơ của MLP
    hard = [c for c in rv["checks"] if c["id"] in rv["hard"]]
    soft = [c for c in rv["checks"] if c["status"] == "soft"]
    top_rule = next((r for r in rule_list if r["level"] in ("danger", "warn")), None)

    if not hard and not soft:  # nhánh 1: mọi phép thử đều ok → đồng thuận
        say("critic", f"Mình đã kiểm tra chéo kết luận của Bác sĩ dữ liệu bằng 4 phép thử (đường huyết, triệu chứng, độ chắc chắn, độ nhạy). "
            f"Không có mâu thuẫn: nguy cơ {mlp} được các luật và số liệu ủng hộ.", "verdict", round=0, stance="agree")
        return ev, 0, "agree"

    if not hard:  # chỉ có nghi vấn nhẹ: một vòng
        c = soft[0]
        say("critic", f"Mình có một điểm cần hỏi lại. {c['evidence']}", round=1, stance="challenge", reply_to="predictor")
        reply = ("Mình đồng ý đây là điểm yếu: mình chỉ ước lượng khả năng theo nhóm người có hồ sơ giống bạn, không phải kết quả xét nghiệm. "
                 f"Mức hiện tại của mình là {mlp} ({_pct(dl['risk_index'])}).")
        say("predictor", reply, round=1, stance="concede", reply_to="critic")
        say("critic", "Vậy mình ghi chú độ bất định này vào báo cáo và khuyên làm xét nghiệm xác nhận thay vì dựa riêng vào mô hình.", "verdict", round=1, stance="note")
        return ev, 1, "noted"

    c = hard[0]  # nhánh 3: tranh luận quanh xung đột thật đầu tiên
    # Vòng 1: phản biện – bảo vệ
    say("critic", f"Mình phản biện Bác sĩ dữ liệu. {c['evidence']} Hai kết luận không khớp nhau.", round=1, stance="challenge", reply_to="predictor")
    say("predictor", f"Mình ước lượng từ 21 chỉ số khảo sát và không có đường huyết trong đầu vào, nên mình không thấy con số này. "
        f"Với đúng những gì mình thấy, xác suất tiểu đường chỉ {_pct(risk['diabetes'])}. Mình không phủ nhận bằng chứng xét nghiệm, nhưng mình không thể tự sửa kết quả.",
        round=1, stance="defend", reply_to="critic")
    # Vòng 2: bằng chứng luật và nguồn tri thức, rồi kết luận
    if top_rule:
        say("guardian", f"Luật {top_rule['id']} đã kích hoạt: {_trim(top_rule['text'], 170)} Căn cứ: {_trim(top_rule['basis'], 80)}.", round=2, stance="evidence", reply_to="critic")
    if quote:
        say("knowledge", f"Tài liệu {_trim(quote['id'], 30)} nói: “{_trim(quote['quote'], 150)}”", "cite", round=2, stance="evidence", reply_to="critic")
    final = rv["final_level"]
    verdict = (f"Kết luận phản biện: khi mô hình và số đo trực tiếp mâu thuẫn, mình ưu tiên số đo. Nâng mức nguy cơ từ {rv['base_level']} lên {final} "
               "và đề nghị xét nghiệm xác nhận (A1C hoặc đường huyết đói lặp lại).") if rv["escalated"] else \
              (f"Kết luận phản biện: mức {final} đã đủ thận trọng, nhưng mình ghi lại bất đồng này và đề nghị xét nghiệm xác nhận.")
    say("critic", verdict, "verdict", round=2, stance="verdict", reply_to="doctor")
    say("predictor", "Mình chấp nhận. Mô hình chỉ cho biết khả năng theo dân số, còn xét nghiệm của bạn mới nói về chính bạn.", round=2, stance="concede", reply_to="critic")
    return ev, 2, "escalated" if rv["escalated"] else "kept"
