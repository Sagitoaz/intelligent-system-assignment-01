"""Tải mô hình instruct nhỏ cho bộ sinh câu trả lời của Chương 5/7 về models/chapter05/llm.

Dùng: python scripts/chapter05/download_llm.py [repo_id]
Mặc định: Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0). Khóa HUGGING_FACE_TOKEN trong .env là tùy chọn.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
repo = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2.5-1.5B-Instruct'
env = ROOT / '.env'
if env.exists() and not os.environ.get('HF_TOKEN'):
    for line in env.read_text(encoding='utf-8').splitlines():
        if line.startswith('HUGGING_FACE_TOKEN='):
            os.environ['HF_TOKEN'] = line.split('=', 1)[1].strip().strip('"\'')
from huggingface_hub import snapshot_download
target = ROOT / 'models/chapter05/llm'
snapshot_download(repo, local_dir=str(target), allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt', 'vocab.json'])
(target / 'SOURCE.txt').write_text(repo + '\n', encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8'); print('Đã tải', repo, '->', target)
