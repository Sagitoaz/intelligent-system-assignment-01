"""Nạp Qwen2.5-1.5B-Instruct với bộ nhớ thấp khi máy không đủ bộ nhớ ảo cho bản float32 (6,2 GB + 3 GB ánh xạ tệp).

Trên máy đang chạy nhiều ứng dụng (Unity, trình duyệt...), việc cấp phát 6,2 GB làm tiến trình sập (access violation).
Chế độ `int8` (mặc định): tạo mô hình rỗng, nạp từng lớp từ safe_open rồi lượng tử hóa động int8 cho các lớp Linear ngay lập tức
(đỉnh khoảng 2,5 GB). Chế độ `fp32`: float32 đầy đủ nhưng chép từng tensor (đỉnh khoảng 6,5 GB), cùng số học với ChatGenerator.load.
Câu trả lời sinh ra bởi chế độ int8 có thể lệch nhẹ so với float32; báo cáo phải ghi rõ khi dùng.

Dùng: `import scripts.chapter05.lowmem_llm as lm; lm.install()` trước khi server/notebook gọi ChatGenerator.load, hoặc
chạy `python scripts/chapter05/lowmem_llm.py --serve 8044 [--mode fp32]` để mở server Chương 5/7 ở chế độ này.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


MODE = {'value': 'int8'}


def load_lowmem(self):
    import torch
    from torch import nn
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from src.chapter04.rag import banned_token_ids
    with self._lock:
        if self.state == 'ready':
            return
        self.state = 'loading'
        start = time.perf_counter()
        try:
            self.torch = torch
            torch.set_num_threads(self.threads)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            config = AutoConfig.from_pretrained(str(self.path), local_files_only=True)
            with torch.device('meta'):
                model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
            quant = MODE['value'] == 'int8'
            with safe_open(str(self.path / 'model.safetensors'), framework='pt') as f:
                def fill(module, prefix):
                    module.to_empty(device='cpu')
                    for name, tensor in module.state_dict().items():
                        tensor.copy_(f.get_tensor(prefix + name))
                fill(model.model.embed_tokens, 'model.embed_tokens.')
                fill(model.model.norm, 'model.norm.')
                for i, layer in enumerate(model.model.layers):
                    fill(layer, f'model.layers.{i}.')
                    if quant:
                        torch.ao.quantization.quantize_dynamic(layer, {nn.Linear}, dtype=torch.qint8, inplace=True)
            model.lm_head.weight = model.model.embed_tokens.weight
            rotary = model.model.rotary_emb
            model.model.rotary_emb = type(rotary)(config=config)
            self.model = model.eval()
            size = self.model.get_output_embeddings().weight.shape[0]
            self.mask = torch.zeros(size, dtype=torch.bool)
            self.mask[banned_token_ids(self.tokenizer, size)] = True
            self.load_seconds = time.perf_counter() - start
            self.lowmem_mode = MODE['value']
            self.state = 'ready'
        except Exception as exc:
            self.state = 'error'
            self.error = repr(exc)
            raise


def install():
    from src.chapter04.rag import ChatGenerator
    ChatGenerator.load = load_lowmem


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', type=int, default=8044)
    parser.add_argument('--mode', choices=['int8', 'fp32'], default='int8')
    args = parser.parse_args()
    MODE['value'] = args.mode
    install()
    import uvicorn
    uvicorn.run('deployment.chapter04.app:app', host='127.0.0.1', port=args.serve)
