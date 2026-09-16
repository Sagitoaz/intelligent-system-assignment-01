"""Build and execute the three final, result-driven Vietnamese A04 notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "results/assignment04/parity"
FULL = ROOT / "results/assignment04/full_scale"
MANIFESTS = ROOT / "results/assignment04/manifests"
NOTEBOOKS = ROOT / "notebooks/assignment04"
TASKS = {
    "diabetes": ("01_diabetes_cnn.ipynb", "Diabetes BRFSS", "classification"),
    "house": ("02_house_price_cnn.ipynb", "House Price", "regression"),
    "comments": ("03_comments_cnn.ipynb", "Comments Sentiment", "classification"),
}


def md(value): return nbf.v4.new_markdown_cell(value)
def code(value): return nbf.v4.new_code_cell(value)


def _task_context(task):
    if task == "diabetes":
        return (
            "Lecture minh họa 8 đặc trưng; project dùng 21 đặc trưng BRFSS. Thứ tự semantic là lựa chọn giảng dạy, không phải locality tự nhiên và không chứng minh CNN tối ưu cho dữ liệu bảng.",
            "[B,1,21]", "Không có embedding/field encoder.",
        )
    if task == "house":
        return (
            "Không xem vector one-hot 61 chiều của A03 là chuỗi locality. Sáu field state, status, acre_lot, house_size, bed, bath được biểu diễn thành 6 vị trí × 8 chiều; split theo property group; output tuyến tính.",
            "[B,8,6]", "Field encoder có 536 tham số và được tính trong tổng tham số.",
        )
    return (
        "A03 dùng TF-IDF → MLP; A04 giữ thứ tự token: token IDs → Embedding → Conv1D. MAX_LEN=192, vocabulary tối đa 20.000 gồm PAD/OOV, embedding 16 chiều; không dùng TF-IDF.",
        "[B,192] → Embedding [B,192,16] → transpose [B,16,192]",
        "Embedding có 320.000 tham số; PAD=0 được mask khi global max pooling.",
    )


def _source_cell(task, problem):
    loss = "SoftmaxCrossEntropyLoss" if problem == "classification" else "MSELoss"
    if task == "diabetes":
        imports = "from src.assignment04.scratch_models import ScratchCNN\nfrom src.assignment04.torch_models import TorchCNN\nfrom src.assignment04.tensorflow_models import TFCNN"
        objects = "[ScratchCNN.forward, ScratchCNN.backward, TorchCNN.forward, TFCNN.call]"
    elif task == "house":
        imports = "from src.assignment04.scratch_models import ScratchHouseCNN, ScratchHouseFieldEncoder\nfrom src.assignment04.torch_models import TorchHouseCNN, TorchHouseFieldEncoder\nfrom src.assignment04.tensorflow_models import TFHouseCNN, TFHouseFieldEncoder"
        objects = "[ScratchHouseFieldEncoder.forward, ScratchHouseCNN.forward, TorchHouseFieldEncoder.forward, TorchHouseCNN.forward, TFHouseFieldEncoder.call, TFHouseCNN.call]"
    else:
        imports = "from src.assignment04.scratch_models import ScratchTextCNN\nfrom src.assignment04.scratch_layers import Embedding\nfrom src.assignment04.torch_models import TorchTextCNN\nfrom src.assignment04.tensorflow_models import TFTextCNN"
        objects = "[Embedding.forward, Embedding.backward, ScratchTextCNN.forward, TorchTextCNN.forward, TFTextCNN.call]"
    return f"""from src.assignment04.scratch_layers import Conv1D, ReLU, MaxPool1D, Dense
from src.assignment04.scratch_losses import {loss}
from src.assignment04.scratch_optim import Adam
{imports}
print('--- Conv1D forward/backward (NumPy) ---')
print(inspect.getsource(Conv1D.forward)); print(inspect.getsource(Conv1D.backward))
print('--- ReLU, pooling, dense, loss, Adam ---')
for obj in [ReLU, MaxPool1D.forward, MaxPool1D.backward, Dense.backward, {loss}.forward, {loss}.backward, Adam.step]:
    print(inspect.getsource(obj))
print('--- Model/encoder thực sự dùng cho {task} ---')
for obj in {objects}:
    print(inspect.getsource(obj))"""


def _full_analysis(problem, full):
    torch = full.loc[full.implementation.eq("pytorch")].iloc[0]
    tf = full.loc[full.implementation.eq("tensorflow")].iloc[0]
    if problem == "classification":
        winner = "PyTorch" if torch.test_f1 >= tf.test_f1 else "TensorFlow"
        return (f"PyTorch test: Accuracy={torch.test_accuracy:.4f}, Precision={torch.test_precision:.4f}, Recall={torch.test_recall:.4f}, F1={torch.test_f1:.4f}, ROC-AUC={torch.test_roc_auc:.4f}. "
                f"TensorFlow test: Accuracy={tf.test_accuracy:.4f}, Precision={tf.test_precision:.4f}, Recall={tf.test_recall:.4f}, F1={tf.test_f1:.4f}, ROC-AUC={tf.test_roc_auc:.4f}. "
                f"Theo F1, {winner} cao hơn trong run đã khóa; không dùng kết quả này để tune lại.")
    winner = "PyTorch" if torch.test_rmse <= tf.test_rmse else "TensorFlow"
    return (f"PyTorch: MAE=${torch.test_mae:,.0f}, RMSE=${torch.test_rmse:,.0f}, R²={torch.test_r2:.4f}; "
            f"TensorFlow: MAE=${tf.test_mae:,.0f}, RMSE=${tf.test_rmse:,.0f}, R²={tf.test_r2:.4f}. "
            f"Theo RMSE, {winner} thấp hơn; metric đã inverse-transform về USD.")


def build_one(task, filename, title, problem):
    metadata = json.loads((MANIFESTS / f"{task}_preprocessing.json").read_text(encoding="utf-8"))
    parity = pd.read_csv(PARITY / task / "parity_results.csv")
    architecture = pd.read_csv(PARITY / task / "architecture_comparison.csv")
    full = pd.read_csv(FULL / task / "full_results.csv")
    full_history = pd.read_csv(FULL / task / "training_history.csv")
    detail, origin, special = _task_context(task)
    counts = {name: int(metadata["splits"][name]["master_rows"]) for name in ("train", "validation", "test")}
    if problem == "classification":
        selection_text = ", ".join(f"{r.architecture}: mean validation F1={r.mean_validation_f1:.4f}" for _, r in architecture.iterrows())
    else:
        selection_text = ", ".join(f"{r.architecture}: mean validation RMSE=${r.mean_validation_rmse:,.0f}" for _, r in architecture.iterrows())
    final_analysis = _full_analysis(problem, full)
    last = full_history.sort_values("epoch").groupby("implementation", as_index=False).tail(1)
    gap_text = ", ".join(f"{r.implementation}: val−train={r.val_loss-r.train_loss:+.4f}" for _, r in last.iterrows())
    if task == "comments":
        stats, audit = metadata["token_length_statistics"]["train"], metadata["duplicate_audit"]
        data_extra = (f"Duplicate overlap unique trước lọc={audit['before']}, sau lọc={audit['after']}. Train lengths: p50={stats['p50']}, p75={stats['p75']}, p90={stats['p90']}, p95={stats['p95']}, p99={stats['p99']}, max={stats['max']}; truncation@192={stats['truncation_rates']['192']:.4%}. "
                      f"Validation OOV={metadata['representation_audit']['validation']['oov_token_rate']:.4%}, test OOV={metadata['representation_audit']['test']['oov_token_rate']:.4%}.")
    elif task == "house":
        audit = metadata["group_audit"]
        data_extra = f"Group-aware split có {audit['groups']:,} groups, largest={audit['largest_group']}, zero overlap={audit['zero_group_overlap']}."
    else:
        data_extra = "Feature order đã khóa: " + " → ".join(metadata["feature_order"]) + "."

    cells = [
        md(f"""# Assignment 04 — {title}

## A–C. Bài toán, dữ liệu và preprocessing

Notebook đã execute này nối toán học → NumPy from scratch → PyTorch → TensorFlow/Keras, so sánh Basic/Improved trên parity và chạy Improved full-scale.

{detail}

Preprocessing/vocabulary chỉ fit train; cùng immutable split, initialization, Adam (lr=1e-3), batch order, tối đa 20 epochs, patience=3. Test không tham gia chọn kiến trúc."""),
        md("### Mục đích của cell\n\nNạp metadata và output thực tế đã lưu. Cell chỉ đọc artifact, không split hoặc huấn luyện lại."),
        code(f"""from pathlib import Path
import json, inspect
import numpy as np
import pandas as pd
from IPython.display import Image, display
HERE=Path.cwd().resolve()
ROOT=next((p for p in (HERE,*HERE.parents) if (p/'src/assignment04').exists()), None)
assert ROOT is not None, 'Không tìm thấy repository root'
META=json.loads((ROOT/'results/assignment04/manifests/{task}_preprocessing.json').read_text(encoding='utf-8'))
PARITY=pd.read_csv(ROOT/'results/assignment04/parity/{task}/parity_results.csv')
ARCH=pd.read_csv(ROOT/'results/assignment04/parity/{task}/architecture_comparison.csv')
FULL=pd.read_csv(ROOT/'results/assignment04/full_scale/{task}/full_results.csv')
HISTORY=pd.read_csv(ROOT/'results/assignment04/full_scale/{task}/training_history.csv')
print({{'task':'{task}','selected':'improved','parity_runs':len(PARITY),'full_runs':len(FULL)}})"""),
        md("### Phân tích kết quả\n\nĐã đọc 6 parity validation runs và đúng 2 full-scale Improved runs (PyTorch, TensorFlow). Không có full NumPy run."),
        md("### Mục đích của cell\n\nKiểm tra số dòng master và preprocessing train-only để phát hiện sai split/leakage trước metric."),
        code("display(pd.DataFrame(META['splits']).T); print('fit_split =',META['fit_split'])"),
        md(f"### Phân tích kết quả\n\nMaster split đúng train={counts['train']:,}, validation={counts['validation']:,}, test={counts['test']:,}. {data_extra}"),
        md(f"""## D–E. Toán học CNN và tensor shapes

Cross-correlation 1D: $z_{{b,o,t}}=b_o+\\sum_c\\sum_j W_{{o,c,j}}x_{{b,c,t+j-p}}$. Kernel trượt tạo local connectivity và weight sharing. ReLU $a=\\max(0,z)$; max-pooling giữ cực đại cục bộ; dense tạo logits/giá trị hồi quy. Backprop dùng chain rule qua Dense → GlobalMaxPool → Conv; Adam cập nhật parameter từ moment bậc một/hai.

Ví dụ lecture `x=[2,1,3,4,2]`, `k=[0.5,-1,0.5]`: biểu thức số đầu tiên cho **1.5**; implementation/tests theo phép tính trực tiếp, không ép thành 1.

Input: `{origin}`.

| Kiến trúc | Dòng tensor sau input |
|---|---|
| Basic | Conv1 `[B,8,L]` → Conv2 `[B,16,L]` → MaxPool `[B,16,L/2]` → GlobalMax `[B,16]` → Output |
| Improved | Conv1 `[B,8,L]` → Conv2 `[B,16,L]` → MaxPool → Conv3 `[B,32,L/2]` → GlobalMax `[B,32]` → Hidden `[B,16]` → Output |

Basic có 3 và Improved có 5 trainable conv/dense transformations; ReLU/pooling không tính là trainable layer. {special}"""),
        md("### Mục đích của cell\n\nTính sliding window và in đúng source cốt lõi/model task-specific để kiểm tra forward, gradient và update."),
        code("x=np.array([2,1,3,4,2.]); k=np.array([.5,-1,.5]); print('hand convolution:',[x[i:i+3]@k for i in range(3)])\n" + _source_cell(task, problem)),
        md("### Phân tích kết quả\n\nOutput tay là `[1.5, -0.5, -1.5]`. Source cho thấy NumPy tự tạo window, chia sẻ kernel, lưu mask/argmax cho backward, truyền gradient qua embedding/field encoder và Adam cập nhật ndarray; scratch không gọi convolution Torch/TF/SciPy."),
        md("## F–K. NumPy verification, Basic/Improved, PyTorch và TensorFlow\n\nFinite-difference kiểm tra gradient; canonical NumPy weights được transpose sang layout từng framework. Parity giữ cùng dữ liệu, batch order và objective. House dùng MSE trên standardized log1p(price), báo metric USD; classification dùng 2 logits và cross-entropy."),
        md("### Mục đích của cell\n\nHiển thị numerical parity và aggregate validation dùng chọn Basic/Improved."),
        code("VERIFY=json.loads((ROOT/'results/assignment04/parity/verification.json').read_text(encoding='utf-8')); display(VERIFY); display(ARCH)"),
        md(f"### Phân tích kết quả\n\nGradient/parity trong tolerance đã khóa. {selection_text}. Improved được chọn trước full-scale; Basic không bị mở test lặp."),
        md("### Mục đích của cell\n\nHiển thị đầy đủ metric validation parity và test chỉ của architecture đã chọn."),
        code("val_cols=['architecture','implementation','parameter_count','best_epoch','training_seconds']+[c for c in PARITY if c.startswith('validation_')]; display(PARITY[val_cols]); test_cols=['implementation']+[c for c in PARITY if c.startswith('test_')]; display(PARITY.loc[PARITY.architecture.eq('improved'),test_cols])"),
        md("### Phân tích kết quả\n\nValidation là cơ sở chọn kiến trúc. Test của Basic để trống có chủ ý; chỉ Improved được mở parity-test. Sai khác sau training có thể do floating-point kernels/optimizer dù initial forward đã parity."),
        md("## L–N. Full-scale Improved CNN\n\nKhông chạy full NumPy scratch vì mục tiêu pedagogical đã hoàn thành ở parity. PyTorch/TensorFlow dùng toàn bộ frozen train/validation và cùng full master-test. Parity-test nằm trong master-test nên full test không phải holdout hoàn toàn mới; không tune sau test."),
        md("### Mục đích của cell\n\nHiển thị best epoch, thời gian và đầy đủ validation/test metrics thực tế của 2 full runs."),
        code("cols=['implementation','train_samples','validation_samples','test_samples','parameter_count','best_epoch','epochs_ran','training_seconds']+[c for c in FULL if c.startswith('validation_') or c.startswith('test_')]; display(FULL[cols])"),
        md(f"### Phân tích kết quả\n\n{final_analysis}"),
        md("### Mục đích của cell\n\nHiển thị loss curves và confusion matrix/metric plot từ đúng history/full results."),
        code(f"display(Image(filename=str(ROOT/'figures/assignment04/{task}_full_loss_curves.png'))); display(Image(filename=str(ROOT/'figures/assignment04/{task}_full_metrics.png')))"),
        md(f"### Phân tích kết quả\n\nGap loss epoch cuối: {gap_text}. Early stopping chọn checkpoint theo validation F1 rồi loss (classification), hoặc RMSE rồi MAE (House); checkpoint tốt nhất được dùng cho test."),
        md(f"""## O–Q. So sánh, diễn giải, kết luận và giới hạn

**Basic vs Improved:** {selection_text}; Improved được chọn từ parity validation, không phải full test.

**PyTorch vs TensorFlow full-scale:** {final_analysis}

Kết quả là một seed và CPU. Với Diabetes/House, locality là cấu trúc giảng dạy nên không kết luận CNN tự nhiên tối ưu cho tabular data. House nhạy outlier trên USD dù training dùng log-target. Comments có locality token hợp lý hơn nhưng vẫn chịu vocabulary hữu hạn, OOV, truncation và embedding học từ đầu. Timing phụ thuộc backend/CPU và không đồng nghĩa chất lượng."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / filename
    client = NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute(); nbf.write(notebook, path)
    executed = sum(c.cell_type == "code" and c.get("execution_count") is not None for c in notebook.cells)
    print(f"EXECUTED {path}: cells={len(cells)} code_cells={executed}", flush=True)


if __name__ == "__main__":
    for task, values in TASKS.items():
        build_one(task, *values)
