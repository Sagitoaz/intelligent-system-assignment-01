"""Nhập hồ sơ bằng giọng nói: Azure Speech-to-Text (REST) + bộ phân tích luật (regex, không dùng LLM).

Khóa và vùng lấy từ chính đối tượng Speaker của tts.py (.env: API_MODEL_TTS, TTS_ENDPOINT); khóa chỉ nằm ở máy chủ và không bao giờ được in/ghi log.
parse_speech(text) tách riêng, không cần mạng: đọc tuổi, giới, chiều cao, cân nặng, đường huyết đói, huyết áp cao, mỡ máu cao, hút thuốc,
vận động và các triệu chứng; hiểu cả số nói bằng chữ ("năm mươi lăm tuổi", "một mét sáu lăm").
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import wave

from project01.mas.kb import strip_accents

MAX_SECONDS = 16
MAX_BYTES = 16000 * 2 * MAX_SECONDS + 4096
LANGS = {"vi-VN", "en-US"}

LABELS = {"age_years": "Tuổi", "sex": "Giới tính", "height_cm": "Chiều cao", "weight_kg": "Cân nặng", "fasting_glucose": "Đường huyết đói",
          "high_bp": "Huyết áp cao", "high_chol": "Mỡ máu cao", "smoker": "Hút thuốc", "phys_activity": "Vận động", "symptoms": "Triệu chứng",
          "heart_disease": "Bệnh tim", "stroke": "Từng đột quỵ"}
CORE = ["age_years", "sex", "height_cm", "weight_kg", "fasting_glucose", "high_bp", "high_chol", "smoker", "phys_activity", "symptoms"]
SYMPTOM_LABEL = {"thirst": "khát nước", "frequent_urination": "tiểu nhiều", "fatigue": "mệt mỏi", "blurred_vision": "nhìn mờ",
                 "slow_healing": "vết thương lâu lành", "weight_loss": "sụt cân"}

# --- số nói bằng chữ -----------------------------------------------------------------------------------------------------------------
DIGITS = {"một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
ASCII_DIGITS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "lam": 5}
TENS = {"mười", "mươi", "chục", "muoi"}
HUNDRED = {"trăm", "tram"}
FILLER = {"lẻ", "linh", "le"}
NUM_NEXT = {"mươi", "muoi", "mười", "trăm", "tram", "tuổi", "tuoi", "ký", "kí", "kg", "cm", "mét", "met", "phẩy", "chấm", "mg", "chục"}


def _tok_value(tok: str, nxt: str | None, accented_ok: bool) -> str | None:
    """Loại từ: 'd' chữ số, 't' mười/mươi, 'h' trăm, 'f' lẻ/linh, hoặc None."""
    if tok in TENS:
        return "t"
    if tok in HUNDRED:
        return "h"
    if tok in FILLER:
        return "f"
    if tok in DIGITS:
        return "d"
    if tok == "nam" and not accented_ok and nxt in NUM_NEXT:  # gõ không dấu: "nam muoi lam tuoi"
        return "d"
    if tok in ASCII_DIGITS and not accented_ok:
        return "d"
    return None


def _digit(tok: str) -> int:
    if tok == "nam":
        return 5
    return DIGITS.get(tok, ASCII_DIGITS.get(tok, 0))


def _run_value(run: list[str]) -> int | None:
    """Giá trị của một chuỗi từ chỉ số (đã tách), None nếu không hợp lệ."""
    if not run:
        return None
    total, i = 0, 0
    if "trăm" in run or "tram" in run:
        j = next(k for k, w in enumerate(run) if w in HUNDRED)
        pre = run[:j]
        total = (_digit(pre[-1]) if pre else 1) * 100
        if len(pre) > 1:  # "hai ba trăm": không hợp lệ
            return None
        run = run[j + 1:]
        if run and run[0] in FILLER:
            run = run[1:]
            return total + (_digit(run[0]) if run else 0)
    elif run[0] in FILLER:
        return None
    if not run:
        return total
    if run[0] in ("mười", "muoi") and (len(run) == 1 or run[1] in DIGITS or run[1] in ASCII_DIGITS or run[1] == "nam"):
        return total + 10 + (_digit(run[1]) if len(run) > 1 else 0)
    if len(run) >= 2 and run[1] in ("mươi", "muoi"):
        return total + _digit(run[0]) * 10 + (_digit(run[2]) if len(run) > 2 else 0)
    if all(w not in TENS for w in run):  # "sáu lăm" = 65, "một" = 1
        n = 0
        for w in run:
            n = n * 10 + _digit(w)
        return total + n if total == 0 or len(run) == 1 else total + n
    return None


def words_to_digits(text: str) -> str:
    """Thay các cụm số nói bằng chữ trong câu bằng chữ số: "năm mươi lăm tuổi" -> "55 tuổi"."""
    has_accent = bool(re.search(r"[ăâêôơưđ]|[ạảãấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", text.lower()))
    toks = re.findall(r"\S+", text.lower())
    out: list[str] = []
    i = 0
    while i < len(toks):
        clean = toks[i].strip(",.;:!?")
        nxt = toks[i + 1].strip(",.;:!?") if i + 1 < len(toks) else None
        kind = _tok_value(clean, nxt, has_accent)
        if kind is None or (clean in ("tư", "năm") and out and not out[-1][-1:].isdigit() and False):
            out.append(toks[i])
            i += 1
            continue
        run = []
        j = i
        while j < len(toks):
            c = toks[j].strip(",.;:!?")
            n2 = toks[j + 1].strip(",.;:!?") if j + 1 < len(toks) else None
            if _tok_value(c, n2, has_accent) is None:
                break
            run.append(c)
            j += 1
            if toks[j - 1] != c:  # có dấu câu ngay sau từ: kết thúc cụm
                break
        val = _run_value(run)
        if val is None or (len(run) == 1 and run[0] in ("tư", "mốt", "lăm", "chục")):
            out.extend(toks[i:j])
        else:
            out.append(str(val))
        i = j
    s = " ".join(out)
    s = re.sub(r"(\d+)\s+(?:phẩy|chấm|phay|cham)\s+(\d+)", r"\1.\2", s)
    return s


# --- phân tích luật ------------------------------------------------------------------------------------------------------------------
def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _neg_before(text: str, pos: int, window: int = 14) -> bool:
    return bool(re.search(r"(không|chưa|ko|hổng|chẳng|khong)\s+(bị\s+|có\s+|hề\s+)?$", text[max(0, pos - window):pos]))


def _flag(text: str, pos_rx: str, neg_extra: str | None = None):
    """True nếu nói có; False nếu nói 'không ...'; None nếu không nhắc."""
    for m in re.finditer(pos_rx, text):
        if _neg_before(text, m.start()):
            return False
        return True
    if neg_extra and re.search(neg_extra, text):
        return False
    return None


def parse_speech(text: str) -> dict:
    """Rút các trường hồ sơ từ một câu tiếng Việt. Chỉ trả trường thực sự nghe được. Khóa theo project01/mas/data.py."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    t = words_to_digits(raw).lower()
    t = t.replace("kí lô", "kg").replace("ki lô", "kg").replace("kilogam", "kg").replace("kilô", "kg")
    f: dict = {}

    m = re.search(r"(\d{1,3})\s*tuổi", t) or re.search(r"\btuổi\s*(?:là\s*)?(\d{1,3})\b", t) or re.search(r"(\d{1,3})\s*(?:years? old|tuoi)", t)
    if m and 18 <= int(m.group(1)) <= 95:
        f["age_years"] = int(m.group(1))

    male = re.search(r"\b(nam|ông|anh|chú|đàn ông|bác trai|con trai|male|man)\b", t)
    female = re.search(r"\b(nữ|bà|chị|cô|dì|mẹ|phụ nữ|bác gái|con gái|female|woman)\b", t)
    if male and not female:
        f["sex"] = "male"
    elif female and not male:
        f["sex"] = "female"
    elif male and female:  # ví dụ "bà ấy ... nam": lấy từ xuất hiện đầu tiên
        f["sex"] = "male" if male.start() < female.start() else "female"

    h = None
    m = re.search(r"(\d)\s*(?:m|mét|met)\s*(\d{1,2})\b", t)
    if m:
        s = m.group(2)
        h = int(m.group(1)) * 100 + (int(s) * 10 if len(s) == 1 else int(s))
    if h is None:
        m = re.search(r"(\d)[.,](\d{1,2})\s*(?:m|mét)\b", t)
        if m:
            h = round(_num(m.group(1) + "." + m.group(2)) * 100)
    if h is None:
        m = re.search(r"(?:cao|chiều cao)\s*(?:là|khoảng|tầm)?\s*(\d{3})\b", t) or re.search(r"\b(\d{3})\s*(?:cm|xăng|phân)", t)
        if m:
            h = int(m.group(1))
    if h is None:
        m = re.search(r"\b(\d)\s*(?:m|mét)\b", t)
        if m and re.search(r"cao", t):
            h = int(m.group(1)) * 100
    if h is not None and 130 <= h <= 210:
        f["height_cm"] = h

    m = re.search(r"(?:nặng|cân nặng|cân)\s*(?:là|khoảng|tầm|cỡ)?\s*(\d{2,3}(?:[.,]\d)?)\b", t) or re.search(r"\b(\d{2,3}(?:[.,]\d)?)\s*(?:kg|ký|kí|cân\b)", t)
    if m and 35 <= _num(m.group(1)) <= 180:
        f["weight_kg"] = _num(m.group(1))

    m = re.search(r"(?:đường huyết|đường máu|glucose|đường)\s*(?:lúc đói|đói|buổi sáng|sáng nay|sáng)?\s*(?:là|khoảng|cỡ|tầm|đo được|bằng|lên|ở mức)?\s*(\d{2,3}(?:[.,]\d)?)\b", t)
    if m:
        v = _num(m.group(1))
        if re.search(r"mmol", t) or (v < 30 and "." in m.group(1).replace(",", ".")):
            v = round(v * 18)
        if 50 <= v <= 500:
            f["fasting_glucose"] = int(round(v))
    elif re.search(r"(?:đường huyết|đường máu|glucose)\s*(?:lúc đói\s*)?(?:là|khoảng|cỡ|tầm)?\s*(\d{1,2}[.,]\d)\s*mmol", t):
        v = _num(re.search(r"(\d{1,2}[.,]\d)\s*mmol", t).group(1)) * 18
        if 50 <= v <= 500:
            f["fasting_glucose"] = int(round(v))

    v = _flag(t, r"(huyết áp cao|cao huyết áp|tăng huyết áp|huyết áp tăng|high blood pressure)", r"huyết áp (bình thường|ổn)")
    if v is not None:
        f["high_bp"] = v
    v = _flag(t, r"(mỡ máu|cholesterol|máu nhiễm mỡ|mỡ trong máu)\s*(cao|nhiều)?|(mỡ máu cao|cholesterol cao)")
    if v is not None and re.search(r"(mỡ máu|cholesterol|nhiễm mỡ|mỡ trong máu)", t):
        f["high_chol"] = v
    v = _flag(t, r"(hút thuốc|nghiện thuốc|thuốc lá|hút điếu|smok)", r"(không hút|chưa hút|bỏ thuốc)")
    if v is not None:
        f["smoker"] = v

    if re.search(r"(ít vận động|lười vận động|ngồi nhiều|ít tập|ít đi bộ|không (hay )?(vận động|tập thể dục|tập luyện|thể dục|đi bộ)|không tập)", t):
        f["phys_activity"] = False
    elif re.search(r"(tập thể dục|vận động|đi bộ|chạy bộ|đạp xe|tập gym|bơi)", t):
        f["phys_activity"] = True

    sym = []
    for key, rx in [("thirst", r"khát"), ("frequent_urination", r"(tiểu|đái|đi vệ sinh|đi tiểu)\s*(nhiều|liên tục|đêm|luôn)"), ("fatigue", r"(mệt|uể oải|kiệt sức)"),
                    ("blurred_vision", r"(nhìn mờ|mờ mắt|mắt mờ|nhìn không rõ)"), ("slow_healing", r"(lâu lành|lâu liền|khó lành|không lành)"),
                    ("weight_loss", r"(sụt cân|sút cân|giảm cân|gầy đi|ốm đi)")]:
        m = re.search(rx, t)
        if m and not _neg_before(t, m.start()):
            sym.append(key)
    if sym:
        f["symptoms"] = sym

    if (v := _flag(t, r"(bệnh tim|nhồi máu|bệnh mạch vành)")) is not None:
        f["heart_disease"] = v
    if (v := _flag(t, r"(đột quỵ|tai biến)")) is not None:
        f["stroke"] = v
    return f


def describe(key: str, value) -> str:
    if key == "sex":
        return "Nam" if value == "male" else "Nữ"
    if key == "symptoms":
        return ", ".join(SYMPTOM_LABEL.get(s, s) for s in value)
    if isinstance(value, bool):
        if key == "phys_activity":
            return "Có vận động" if value else "Ít/không vận động"
        return "Có" if value else "Không"
    unit = {"age_years": " tuổi", "height_cm": " cm", "weight_kg": " kg", "fasting_glucose": " mg/dL"}.get(key, "")
    return f"{value:g}{unit}" if isinstance(value, float) else f"{value}{unit}"


def analyse(text: str) -> dict:
    fields = parse_speech(text)
    understood = [dict(key=k, label=LABELS.get(k, k), text=describe(k, v)) for k, v in fields.items()]
    missing = [dict(key=k, label=LABELS[k]) for k in CORE if k not in fields]
    return dict(text=str(text or "").strip(), fields=fields, understood=understood, missing=missing)


# --- Azure Speech-to-Text ------------------------------------------------------------------------------------------------------------
def check_wav(data: bytes) -> float:
    """Kiểm tra WAV PCM 16 kHz mono 16-bit, trả về thời lượng (giây)."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"Âm thanh quá dài (tối đa khoảng {MAX_SECONDS} giây)")
    try:
        with wave.open(io.BytesIO(data)) as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ValueError("Âm thanh không phải tệp WAV hợp lệ") from exc
    if (ch, width, rate) != (1, 2, 16000):
        raise ValueError("Cần WAV PCM 16 kHz, mono, 16-bit")
    dur = n / rate
    if dur < 0.3:
        raise ValueError("Âm thanh quá ngắn, hãy nói lại")
    if dur > MAX_SECONDS:
        raise ValueError(f"Âm thanh quá dài (tối đa khoảng {MAX_SECONDS} giây)")
    return dur


def recognize(speaker, wav: bytes, language: str = "vi-VN", timeout: float = 25) -> str:
    key, region = getattr(speaker, "key", ""), getattr(speaker, "region", "")
    if not key or not region:
        raise RuntimeError("Chưa cấu hình dịch vụ nhận giọng nói (thiếu khóa Azure trong .env)")
    url = (f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?"
           + urllib.parse.urlencode({"language": language, "format": "detailed"}))
    req = urllib.request.Request(url, data=wav, headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                                                         "Accept": "application/json", "User-Agent": "clinic-demo"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # không đưa nội dung lỗi (có thể chứa thông tin cấu hình) ra ngoài
        raise RuntimeError(f"Dịch vụ nhận giọng nói từ chối yêu cầu (mã {exc.code})") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("Không gọi được dịch vụ nhận giọng nói (kiểm tra mạng)") from None
    status = body.get("RecognitionStatus")
    if status == "Success":
        nbest = body.get("NBest") or []
        return (nbest[0].get("Display") if nbest else None) or body.get("DisplayText", "")
    if status in ("NoMatch", "InitialSilenceTimeout", "BabbleTimeout"):
        raise ValueError("Không nghe thấy giọng nói rõ ràng, hãy nói gần micro hơn và thử lại")
    raise RuntimeError("Dịch vụ nhận giọng nói không nhận dạng được")


def routes(clinic, speaker):
    def status() -> dict:
        return dict(enabled=bool(getattr(speaker, "key", "") and getattr(speaker, "region", "")))

    def post_stt(body: dict) -> dict:
        lang = str(body.get("language") or "vi-VN")
        if lang not in LANGS:
            raise ValueError("Ngôn ngữ chưa hỗ trợ")
        try:
            wav = base64.b64decode(str(body.get("audio_base64") or ""), validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Âm thanh mã hóa sai") from exc
        dur = check_wav(wav)
        text = recognize(speaker, wav, lang)
        out = analyse(text)
        out["seconds_audio"] = round(dur, 1)
        return out

    def post_parse(body: dict) -> dict:
        text = str(body.get("text") or "")[:600]
        if not text.strip():
            raise ValueError("Hãy nhập một câu")
        return analyse(text)

    return {"/api/stt/status": status}, {"/api/stt": post_stt, "/api/stt/parse": post_parse}
