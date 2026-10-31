"""Giọng nói cho các agent bằng Azure Speech (tùy chọn).

Khóa và endpoint lấy từ .env (API_MODEL_TTS, TTS_ENDPOINT); khóa chỉ nằm ở máy chủ, trình duyệt chỉ nhận file âm thanh.
Không có khóa hoặc lỗi mạng thì demo vẫn chạy bình thường, chỉ không có tiếng.
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
MAX_CHARS = 320
CACHE_SIZE = 200

# giọng và cao độ/tốc độ khác nhau để phân biệt các agent
VOICES = {
    "doctor": ("vi-VN-NamMinhNeural", "0%", "-2%"),
    "predictor": ("vi-VN-NamMinhNeural", "-10%", "+4%"),
    "knowledge": ("vi-VN-HoaiMyNeural", "0%", "-2%"),
    "coach": ("vi-VN-NamMinhNeural", "+12%", "+8%"),
    "guardian": ("vi-VN-HoaiMyNeural", "-8%", "0%"),
    "reporter": ("vi-VN-HoaiMyNeural", "+10%", "+4%"),
}
DEFAULT_VOICE = VOICES["doctor"]


def _load_env() -> dict:
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
    def __init__(self):
        env = _load_env()
        self.key = env.get("API_MODEL_TTS", "")
        endpoint = env.get("TTS_ENDPOINT", "")
        m = re.match(r"https?://([a-z0-9-]+)\.", endpoint)
        self.region = m.group(1) if m else ""
        self.url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1" if self.region else ""
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self.key and self.url and self.failures < 5)

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️]", "", text)  # bỏ emoji
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_CHARS]

    def synthesize(self, text: str, agent: str = "doctor") -> bytes:
        text = self.clean(text)
        if not text:
            raise ValueError("Không có nội dung để đọc")
        if not self.enabled:
            raise RuntimeError("Chưa cấu hình giọng nói")
        voice, pitch, rate = VOICES.get(agent, DEFAULT_VOICE)
        cache_key = (voice, pitch, rate, text)
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
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
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return audio
