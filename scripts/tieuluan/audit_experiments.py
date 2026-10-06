"""Kiểm toán độc lập kết quả đã lưu, CHỈ ĐỌC (không sửa bản ghi, manifest hay mô hình).

1. Mã băm SHA-256 của dữ liệu đã xử lý và mã nguồn hiện tại trùng với run_manifest.json lúc chạy thực nghiệm.
2. Mọi bản ghi mang đúng "dấu vân tay" của manifest; nhãn và số mẫu khớp dữ liệu đã xử lý.
3. Ngưỡng và mọi thước đo được tính lại từ dự báo đã lưu và khớp với bản ghi.
4. Nạp lại mọi checkpoint NumPy và checkpoint PyTorch/Keras của hạt giống 11, dự báo lại và so với dự báo đã lưu.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.audit_experiments
Kết quả: results/tieuluan/audit.json
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from scripts.tieuluan.run_experiments import metrics, probabilities, select_threshold
from src.tieuluan.experiment_data import OUT, ROOT
from src.tieuluan.experiment_models import build_models, predict_logits
from src.tieuluan.run_provenance import check_raw_snapshot


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    folder, models = ROOT / "results/tieuluan/full", ROOT / "models/tieuluan/full"
    records = sorted(folder.glob("*__*.json"))
    assert len(records) == 128, len(records)
    check_raw_snapshot(ROOT)
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf8"))
    drift = [name for name, digest in manifest["files"].items()
             if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    cache, reloaded, max_error = {}, 0, 0.0
    for path in records:
        r = json.loads(path.read_text(encoding="utf8"))
        assert r["run_fingerprint"] == manifest["fingerprint"], path.name
        if r["dataset"] not in cache:
            with np.load(OUT / f"{r['dataset']}.npz", allow_pickle=False) as d:
                cache[r["dataset"]] = {k: d[k] for k in d.files}
        data = cache[r["dataset"]]
        with np.load(path.with_name(path.stem + "_predictions.npz"), allow_pickle=False) as d:
            p = {k: d[k] for k in d.files}
        np.testing.assert_array_equal(p["y"], data["test_y"])
        np.testing.assert_array_equal(p["val_y"], data["val_y"])
        assert len(p["y"]) == r["n_test"] and len(data["train_y"]) == r["n_train"]
        threshold = .5 if r["framework"] == "baseline" else select_threshold(p["val_y"], p["val_probability"])
        assert np.isclose(threshold, r["metrics"]["threshold"]), path.name
        for name, value in metrics(p["y"], p["probability"], threshold).items():
            np.testing.assert_allclose(value, r["metrics"][name], atol=1e-10, err_msg=f"{path.name}:{name}")
        fw = r["framework"]
        if fw == "scratch" or (r["seed"] == 11 and fw in ("pytorch", "keras")):
            x = data["test_" + r["representation"]]
            if fw == "keras":
                import keras

                model = keras.models.load_model(models / f"{path.stem}.keras", compile=False)
            else:
                model = build_models(r["kind"], x.shape[1:], r["seed"], frameworks=(fw,))[fw]
                if fw == "scratch":
                    model.load(models / f"{path.stem}.npz")
                else:
                    import torch

                    model.load_state_dict(torch.load(models / f"{path.stem}.pt", map_location="cpu", weights_only=True))
            actual = probabilities(predict_logits(model, fw, x))
            error = float(np.max(np.abs(actual - p["probability"])))
            max_error = max(max_error, error)
            np.testing.assert_allclose(actual, p["probability"], atol=2e-6, rtol=2e-5, err_msg=path.name)
            reloaded += 1
            if fw == "keras":
                keras.backend.clear_session()
    report = {"records_recomputed": len(records), "checkpoints_reloaded": reloaded,
              "max_probability_reload_error": max_error, "fingerprint": manifest["fingerprint"],
              "files_changed_since_run": drift, "raw_snapshot_verified": True,
              "labels_counts_thresholds_metrics_verified": True}
    (ROOT / "results/tieuluan/audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if drift:
        raise SystemExit("Mã nguồn hoặc dữ liệu đã thay đổi so với lúc chạy thực nghiệm: " + ", ".join(drift))


if __name__ == "__main__":
    main()
