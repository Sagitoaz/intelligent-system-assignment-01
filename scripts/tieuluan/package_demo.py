"""Build a reviewable, unpublished Docker Space bundle from completed artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / 'deployment/tieuluan'
TARGET = DEPLOY / 'hf_space'


def main():
    # Delete only this script's generated staging directory, never project inputs.
    resolved = TARGET.resolve()
    if (resolved != (ROOT / 'deployment/tieuluan/hf_space').resolve()
            or not resolved.is_relative_to(ROOT.resolve()) or TARGET.is_symlink()):
        raise RuntimeError('Unexpected package target')
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    def copy(source, relative=None):
        destination = TARGET / (relative if relative else source.relative_to(ROOT))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for source in (ROOT / 'src/tieuluan').rglob('*.py'):
        copy(source)
    for name in ['app.py', 'requirements.txt']:
        copy(DEPLOY / name)
    copy(DEPLOY / 'Dockerfile', Path('Dockerfile'))
    copy(DEPLOY / 'README.md', Path('UPLOAD_GUIDE.md'))
    for source in (ROOT / 'data/tieuluan/processed').glob('*.npz'):
        with np.load(source, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files if key.startswith(('test_', 'scaler_', 'imputer_')) or key == 'feature_names'}
        if not payload:
            continue
        destination = TARGET / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
    inventory = ROOT / 'data/tieuluan/processed/inventory.json'
    if inventory.exists():
        # Retain provenance, not raw example rows.
        contents = json.loads(inventory.read_text(encoding='utf-8'))
        for entry in contents.values():
            entry.pop('raw_sample', None)
        destination = TARGET / inventory.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(contents, ensure_ascii=False, indent=2), encoding='utf-8')
    for source in (ROOT / 'data/tieuluan/app').glob('*'):
        if source.is_file():
            copy(source)
    for source in (ROOT / 'models/tieuluan/full').glob('*__scratch__*.npz'):
        copy(source)
    skipped = []
    for source in (ROOT / 'results/tieuluan/full').glob('*'):
        if source.suffix not in ('.json', '.csv') and not source.name.endswith('_predictions.npz'):
            continue
        try:
            # Files being written by training are skipped rather than packaged half-written.
            if source.suffix == '.json':
                json.loads(source.read_text(encoding='utf-8'))
            elif source.suffix == '.npz':
                with np.load(source, allow_pickle=False) as data:
                    for key in data.files:
                        _ = data[key]
            copy(source)
        except (OSError, ValueError, EOFError):
            skipped.append(source.name)
    (TARGET / 'README.md').write_text('''---
title: AI tai chinh - Minh hoa hoc thuat
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Minh họa AI tài chính trên dữ liệu lịch sử

Ứng dụng kiểm tra mô hình NumPy trên các mẫu test giữ riêng. Không phải khuyến nghị giao dịch.
Ngày chốt dữ liệu thị trường: 30/09/2026. Xem UPLOAD_GUIDE.md để chạy và tải lên Space.
Gói được chuẩn bị cục bộ, chưa xuất bản. Xem DATA_SOURCES.md trước khi chia sẻ công khai.
SHA256_MANIFEST.json ghi hash của từng tệp tại thời điểm đóng gói.
''', encoding='utf-8')
    (TARGET / 'DATA_SOURCES.md').write_text('''# Nguồn và phạm vi chia sẻ dữ liệu

Gói gồm dữ liệu test đã tiền xử lý, trọng số tự huấn luyện và kết quả đánh giá của tiểu luận.
Dữ liệu thị trường gốc ghi nguồn Yahoo Finance (S&P 500 và Bitcoin), DNSE Entrade / SSI iBoard (VN-Index).
Việc truy cập hoặc tải được dữ liệu không có nghĩa nhà cung cấp cấp quyền tái phân phối.
Gói này không khẳng định bất kỳ quyền tái phân phối công khai nào từ các nhà cung cấp thị trường.
Chỉ sử dụng cục bộ hoặc phạm vi học thuật được phép cho tới khi kiểm tra điều khoản và quyền cần thiết.
Nếu đưa Space công khai, xác minh quyền trước hoặc bỏ các tập thị trường cùng kết quả tương ứng.

Dữ liệu rủi ro nguồn UCI Machine Learning Repository:
- Taiwan bankruptcy: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- Default of credit card clients: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Kiểm tra trang nguồn và điều kiện ghi công trước khi phân phối. Không bổ sung danh tính khách hàng.

Các mảng test_*, scaler_*, imputer_* và feature_names được giữ trong processed; loại train/validation.
Tài nguyên app chứa giá lịch sử và dòng test UCI ở dạng gốc để hiển thị và thử nhập hồ sơ.
Kết quả predictions có thể chứa nhãn/xác suất validation để đối chiếu thực nghiệm, không chứa đặc trưng khách hàng gốc.
inventory.json giữ thông tin provenance và hash dữ liệu nguồn, đã bỏ raw_sample.
''', encoding='utf-8')
    files = {str(p.relative_to(TARGET)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(TARGET.rglob('*')) if p.is_file()}
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'published': False,
                'skipped_incomplete_files': skipped, 'sha256': files,
                'note': 'Manifest excludes itself. Re-run after training finishes.'}
    (TARGET / 'SHA256_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = shutil.make_archive(str(DEPLOY / 'tieuluan_hf_space'), 'zip', root_dir=TARGET)
    print(json.dumps({'directory': str(TARGET), 'zip': archive, 'files': len(files),
                      'skipped': skipped, 'bytes': Path(archive).stat().st_size}))


if __name__ == '__main__':
    main()
