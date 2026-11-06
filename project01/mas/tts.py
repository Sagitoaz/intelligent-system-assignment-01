"""Giọng nói cho các agent bằng Azure Speech (tùy chọn).

Khóa và endpoint lấy từ .env (API_MODEL_TTS, TTS_ENDPOINT); khóa chỉ nằm ở máy chủ, trình duyệt chỉ nhận file âm thanh.
Không có khóa hoặc lỗi mạng thì demo vẫn chạy bình thường, chỉ không có tiếng.

======================================================================================================
BẢN ĐỒ FILE
======================================================================================================
Chức năng: TTS (Text-To-Speech) - đọc lời của từng agent thành giọng nói tiếng Việt để demo sinh động hơn.
Thành phần chính:
  - VOICES / DEFAULT_VOICE : bảng giọng (tên giọng Azure, cao độ pitch, tốc độ rate) riêng cho từng agent
  - _load_env()            : đọc khóa từ file .env ở thư mục gốc repo, biến môi trường (nếu có) ghi đè
  - Speaker                : lớp chính; Speaker.enabled (có dùng được không), Speaker.clean (làm sạch văn bản),
                             Speaker.synthesize (gọi Azure, trả bytes MP3, có cache LRU)
Đọc: .env (API_MODEL_TTS, TTS_ENDPOINT). Không ghi file. Gọi mạng tới https://<region>.tts.speech.microsoft.com.
Được gọi từ: project01/server.py (SPEAKER = Speaker(); route POST /api/tts; GET /api/tts/status).
Cũng được truyền vào stt.routes(clinic, speaker) để dùng chung khóa/region Azure cho nhận dạng giọng nói.
Không có khóa -> synthesize ném RuntimeError -> server trả HTTP 503 -> giao diện tắt nút nghe, mọi thứ khác vẫn chạy.
======================================================================================================
"""
from __future__ import annotations

import os
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from xml.sax.saxutils import escape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_CHARS = 320   # giới hạn độ dài mỗi lần đọc: tránh tốn phí/độ trễ và tránh bị lạm dụng làm proxy TTS
CACHE_SIZE = 200  # số đoạn âm thanh giữ trong bộ nhớ (LRU) để câu lặp lại không phải gọi mạng lần nữa

# giọng và cao độ/tốc độ khác nhau để phân biệt các agent
# định dạng: agent -> (tên giọng Azure Neural, pitch tương đối, rate tương đối)
VOICES = {
    "doctor": ("vi-VN-NamMinhNeural", "0%", "-2%"),
    "predictor": ("vi-VN-NamMinhNeural", "-10%", "+4%"),
    "knowledge": ("vi-VN-HoaiMyNeural", "0%", "-2%"),
    "coach": ("vi-VN-NamMinhNeural", "+12%", "+8%"),
    "guardian": ("vi-VN-HoaiMyNeural", "-8%", "0%"),
    "reporter": ("vi-VN-HoaiMyNeural", "+10%", "+4%"),
    "critic": ("vi-VN-NamMinhNeural", "-18%", "-4%"),
}
DEFAULT_VOICE = VOICES["doctor"]


def _load_env() -> dict:
    """Đọc cấu hình TTS: ưu tiên biến môi trường, sau đó file .env ở gốc repo (dòng KEY=VALUE, bỏ chú thích #).

    Đầu ra: dict có thể chứa API_MODEL_TTS (khóa Azure) và TTS_ENDPOINT (URL chứa region). Thiếu file thì trả dict rỗng.
    """
    env = {}
    path = os.path.join(ROOT, ".env")
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("API_MODEL_TTS", "TTS_ENDPOINT"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class Speaker:
    """Bộ đọc giọng nói Azure Speech cho các agent: giữ khóa, URL dịch vụ, cache âm thanh và bộ đếm lỗi liên tiếp."""

    def __init__(self):
        env = _load_env()
        self.key = env.get("API_MODEL_TTS", "")
        endpoint = env.get("TTS_ENDPOINT", "")
        m = re.match(r"https?://([a-z0-9-]+)\.", endpoint)  # lấy region (vd "southeastasia") từ tiền tố tên miền của endpoint
        self.region = m.group(1) if m else ""
        self.url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1" if self.region else ""
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()  # server chạy đa luồng (ThreadingHTTPServer) nên cache cần khóa
        self.failures = 0

    @property
    def enabled(self) -> bool:
        """True nếu có đủ khóa + URL và chưa quá 5 lỗi mạng liên tiếp (tự tắt để khỏi chờ timeout mỗi lần bấm)."""
        return bool(self.key and self.url and self.failures < 5)

    @staticmethod
    def clean(text: str) -> str:
        """Làm sạch văn bản trước khi đọc: bỏ emoji, bỏ URL, gộp khoảng trắng, cắt tối đa MAX_CHARS ký tự."""
        text = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️]", "", text)  # bỏ emoji
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_CHARS]

    def synthesize(self, text: str, agent: str = "doctor") -> bytes:
        """Chuyển văn bản thành MP3 bằng giọng của `agent`.

        Đầu vào: text (văn bản), agent (khóa trong VOICES, sai thì dùng giọng bác sĩ trưởng).
        Đầu ra: bytes MP3 (24kHz, 48kbps, mono). ValueError nếu văn bản rỗng sau khi làm sạch;
        RuntimeError nếu chưa cấu hình khóa hoặc không gọi được dịch vụ.
        """
        text = self.clean(text)
        if not text:
            raise ValueError("Không có nội dung để đọc")
        if not self.enabled:
            raise RuntimeError("Chưa cấu hình giọng nói")
        voice, pitch, rate = VOICES.get(agent, DEFAULT_VOICE)
        cache_key = (voice, pitch, rate, text)
        with self._lock:
            if cache_key in self._cache:  # trúng cache: đưa lên cuối hàng (mới dùng gần nhất) rồi trả luôn
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
        # SSML là ngôn ngữ đánh dấu giọng nói của Azure; escape() để ký tự đặc biệt (&, <) trong văn bản không phá XML
        ssml = (f'<speak version="1.0" xml:lang="vi-VN"><voice name="{voice}"><prosody pitch="{pitch}" rate="{rate}">'
                f'{escape(text)}</prosody></voice></speak>')
        req = urllib.request.Request(self.url, data=ssml.encode("utf-8"), headers={
            "Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3", "User-Agent": "clinic-demo"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                audio = r.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            self.failures += 1
            raise RuntimeError("Không gọi được dịch vụ giọng nói") from exc
        self.failures = 0
        with self._lock:
            self._cache[cache_key] = audio
            while len(self._cache) > CACHE_SIZE:  # vượt dung lượng thì bỏ mục cũ nhất
                self._cache.popitem(last=False)
        return audio
