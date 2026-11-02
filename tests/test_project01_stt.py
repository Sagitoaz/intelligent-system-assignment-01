"""Kiểm thử nhập hồ sơ bằng giọng nói: bộ phân tích luật không cần mạng, kiểm tra WAV và các route."""
import base64
import io
import os
import sys
import wave

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from project01.mas import stt  # noqa: E402


def wav_bytes(seconds=1.0, rate=16000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds) * channels)
    return buf.getvalue()


CASES = [
    ("Tôi là nam, 55 tuổi, cao 1m65, nặng 70 ký, đường huyết đói 190, bị huyết áp cao và hút thuốc",
     dict(age_years=55, sex="male", height_cm=165, weight_kg=70, fasting_glucose=190, high_bp=True, smoker=True)),
    ("Bà tôi năm mươi lăm tuổi, cao một mét sáu lăm, nặng sáu mươi hai ký, hay khát nước và tiểu nhiều, mệt mỏi",
     dict(age_years=55, sex="female", height_cm=165, weight_kg=62, symptoms=["thirst", "frequent_urination", "fatigue"])),
    ("Chị 34 tuổi cao 162 cm nặng 54 kg không hút thuốc, hay tập thể dục",
     dict(age_years=34, sex="female", height_cm=162, weight_kg=54, smoker=False, phys_activity=True)),
    ("Ông 67 tuổi, mỡ máu cao, ít vận động, nhìn mờ, vết thương lâu lành, sụt cân",
     dict(age_years=67, sex="male", high_chol=True, phys_activity=False, symptoms=["blurred_vision", "slow_healing", "weight_loss"])),
    ("nam sáu mươi tuổi cao một trăm bảy mươi nặng bảy mươi lăm kg đường huyết một trăm chín mươi",
     dict(age_years=60, sex="male", height_cm=170, weight_kg=75, fasting_glucose=190)),
    ("Em là nữ hai mươi bốn tuổi, một mét năm tám, năm mươi mốt ký, không bị huyết áp cao",
     dict(age_years=24, sex="female", height_cm=158, weight_kg=51, high_bp=False)),
    ("Anh 52 tuổi nặng 88 kg đường huyết 7,2 mmol", dict(age_years=52, sex="male", weight_kg=88, fasting_glucose=130)),
    ("nam bốn mươi lăm tuổi nặng bảy mươi hai phẩy năm ký", dict(age_years=45, sex="male", weight_kg=72.5)),
    ("Cao 1,7 m nặng 80 kg lười vận động, cao huyết áp, cholesterol cao",
     dict(height_cm=170, weight_kg=80, phys_activity=False, high_bp=True, high_chol=True)),
    ("Xin chào, hôm nay trời đẹp quá", {}),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_parse_speech(text, expected):
    got = stt.parse_speech(text)
    assert got == expected, got


def test_out_of_range_values_are_dropped():
    assert "age_years" not in stt.parse_speech("5 tuổi") and "weight_kg" not in stt.parse_speech("nặng 500 kg")


def test_analyse_lists_understood_and_missing():
    r = stt.analyse("Tôi 55 tuổi, nam")
    assert {u["key"] for u in r["understood"]} == {"age_years", "sex"}
    assert "height_cm" in {m["key"] for m in r["missing"]} and "age_years" not in {m["key"] for m in r["missing"]}
    assert r["understood"][0]["text"] == "55 tuổi"


def test_check_wav():
    assert stt.check_wav(wav_bytes(1.0)) == 1.0
    for bad in (wav_bytes(1.0, rate=44100), wav_bytes(1.0, channels=2), wav_bytes(0.1), b"not a wav"):
        with pytest.raises(ValueError):
            stt.check_wav(bad)
    with pytest.raises(ValueError):
        stt.check_wav(wav_bytes(20))


class NoKey:
    key, region = "", ""


class FakeSpeaker:
    key, region = "secret-key-123", "westus"


def test_routes_without_key_give_clear_error():
    get, post = stt.routes(None, NoKey())
    assert get["/api/stt/status"]() == {"enabled": False}
    body = dict(audio_base64=base64.b64encode(wav_bytes(1.0)).decode(), language="vi-VN")
    with pytest.raises(RuntimeError) as e:
        post["/api/stt"](body)
    assert "cấu hình" in str(e.value)


def test_stt_pipeline_with_mocked_azure(monkeypatch):
    seen = {}

    def fake(speaker, wav, language="vi-VN", timeout=25):
        seen.update(n=len(wav), lang=language)
        return "Tôi 55 tuổi cao 1m65."
    monkeypatch.setattr(stt, "recognize", fake)
    get, post = stt.routes(None, FakeSpeaker())
    assert get["/api/stt/status"]() == {"enabled": True}
    r = post["/api/stt"](dict(audio_base64=base64.b64encode(wav_bytes(1.0)).decode(), language="vi-VN"))
    assert r["fields"] == dict(age_years=55, height_cm=165) and seen["lang"] == "vi-VN"
    assert "secret-key" not in str(r)
    with pytest.raises(ValueError):
        post["/api/stt"](dict(audio_base64="@@@", language="vi-VN"))
    with pytest.raises(ValueError):
        post["/api/stt"](dict(audio_base64=base64.b64encode(wav_bytes(1.0)).decode(), language="xx"))


def test_parse_route():
    _, post = stt.routes(None, NoKey())
    assert post["/api/stt/parse"](dict(text="nam 40 tuổi"))["fields"] == dict(age_years=40, sex="male")
    with pytest.raises(ValueError):
        post["/api/stt/parse"](dict(text="  "))
