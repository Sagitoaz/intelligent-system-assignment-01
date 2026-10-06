"""Vẽ các sơ đồ khái niệm (không phụ thuộc kết quả thực nghiệm) cho báo cáo.

Mỗi sơ đồ dùng hệ tọa độ tính theo inch với tỉ lệ 1:1 (aspect = equal) để hình tròn không bị méo
và vị trí chữ/mũi tên được đặt chính xác.

Chạy: .venv\\Scripts\\python.exe -m scripts.tieuluan.make_diagrams
Kết quả: figures/tieuluan/diagram_*.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.tieuluan.config import FIGURES_DIR

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, LINE, BOX = "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2", "#f4f3f0"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_AQUA, LIGHT_YELLOW = "#e3eefb", "#fde6dc", "#dcf3ea", "#fdf0d2"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "savefig.facecolor": "white"})


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=BOX, ec=LINE, size=8.5, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color,
            linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def node(ax, x, y, r, text="", fc=LIGHT_BLUE, ec=BLUE, size=9):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.1, zorder=3))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=4)


# ---------------------------------------------------------------------------------------------
def ai_timeline():
    """Dòng thời gian hai làn; trục năm được co giãn từng đoạn để giai đoạn gần đây đủ chỗ ghi chú."""
    width, height = 6.8, 4.1
    fig, ax = canvas(width, height)
    left, right = 0.25, 6.65
    breaks = [(1940, 0.0), (1990, 0.42), (2010, 0.62), (2027, 1.0)]

    def X(year):
        for (y0, p0), (y1, p1) in zip(breaks, breaks[1:]):
            if year <= y1:
                return left + (p0 + (year - y0) / (y1 - y0) * (p1 - p0)) * (right - left)
        return right

    band_y, band_h = 1.78, 0.5
    eras = [(1943, 1969, "Khởi đầu\n& lạc quan", LIGHT_BLUE), (1969, 1993, "Hệ chuyên gia &\nhai mùa đông", LIGHT_YELLOW),
            (1993, 2012, "Học máy\nthống kê", LIGHT_AQUA), (2012, 2022, "Học sâu", LIGHT_BLUE),
            (2022, 2027, "AI tạo sinh\n& tác tử", LIGHT_ORANGE)]
    for start, end, label, color in eras:
        ax.add_patch(Rectangle((X(start), band_y), X(end) - X(start), band_h, fc=color, ec="white", lw=1.5))
        ax.text((X(start) + X(end)) / 2, band_y + band_h / 2, label, ha="center", va="center", fontsize=6.5,
                color=INK2, linespacing=1.05)
    for x0, x1 in [(1974, 1980), (1987, 1993)]:   # hai "mùa đông AI": dải gạch chéo mảnh ở đáy dải giai đoạn
        ax.add_patch(Rectangle((X(x0), band_y), X(x1) - X(x0), 0.09, fc="none", ec=MUTED, hatch="/////", lw=0))
    for year in (1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015, 2020, 2025):
        ax.text(X(year), band_y - 0.05, str(year), ha="center", va="top", fontsize=6.2, color=MUTED)
    ax.plot([X(1990), X(1990)], [band_y - 0.02, band_y + band_h + 0.02], color="white", lw=0)
    top = band_y + band_h
    # Nhãn dùng "|" để xuống dòng; căn trái/phải khi hai mốc quá gần nhau để thân mốc không cắt chữ.
    ai = [(1950, 2.62, "Phép thử|Turing 1950", "center"), (1956, 3.3, "Dartmouth|1956", "right"),
          (1958, 3.3, "Perceptron|1958", "left"), (1986, 2.62, "Lan truyền|ngược 1986", "center"),
          (1997, 3.25, "Deep Blue,|LSTM 1997", "center"), (2006, 2.62, "Mạng niềm tin|sâu 2006", "center"),
          (2012, 3.25, "AlexNet|2012", "center"), (2016, 2.62, "AlphaGo|2016", "right"),
          (2017, 3.6, "Transformer|2017", "center"), (2022, 3.25, "ChatGPT|2022", "center"),
          (2024, 2.62, "Nobel cho|AI 2024", "center"), (2026, 3.6, "Luật AI|VN 2026", "center")]
    ai = [(year, y, label.replace("|", "\n"), align) for year, y, label, align in ai]
    fin = [(1952, 1.1, "Markowitz\n1952"), (1968, 0.5, "Z-score\nAltman 1968"), (1990, 1.1, "ANN dự báo\nphá sản 1990"),
           (2010, 0.5, "Flash Crash\n2010"), (2018, 1.1, "LSTM trên\nS&P 500 2018"), (2023, 0.5, "CNN ảnh giá,\nBloombergGPT 2023"),
           (2025, 1.1, "Kronos\n2025")]
    for year, y, label, align in ai:
        x = X(year)
        ax.plot([x, x], [top, y - 0.03 + (0.3 if align != "center" else 0)], color=BLUE, lw=0.8)
        ax.plot(x, top, "o", ms=3.2, color=BLUE)
        offset = {"center": 0, "left": 0.04, "right": -0.04}[align]
        ax.text(x + offset, y, label, ha=align, va="bottom", fontsize=6.4, linespacing=1.05)
    for year, y, label in fin:
        x = X(year)
        ax.plot([x, x], [band_y, y + 0.03], color=ORANGE, lw=0.8)
        ax.plot(x, band_y, "o", ms=3.2, color=ORANGE)
        ax.text(x, y, label, ha="center", va="top", fontsize=6.4, linespacing=1.05)
    ax.text(0.02, 4.0, "● AI nói chung", color=BLUE, fontsize=7.5, weight="bold", va="center")
    ax.text(0.02, 0.06, "● AI trong tài chính – đầu tư", color=ORANGE, fontsize=7.5, weight="bold", va="center")
    ax.text(6.78, 0.06, "Trục năm co giãn từng đoạn: 1940–1990 · 1990–2010 · 2010–2027", ha="right", va="center",
            fontsize=6, color=MUTED)
    save(fig, "diagram_ai_timeline.png")


def study_design():
    width, height = 6.8, 2.75
    fig, ax = canvas(width, height)
    cols = [("DỮ LIỆU", ["UCI: phá sản\ndoanh nghiệp", "UCI: vỡ nợ\nthẻ tín dụng", "S&P 500, VN-Index,\nBitcoin"]),
            ("BIỂU DIỄN", ["Bảng chỉ tiêu\n(Chương 2)", "Ảnh GASF 20×20\n(Chương 3)", "Chuỗi 20 lợi suất\n(Chương 4)"]),
            ("MÔ HÌNH", ["Logistic, RF,\nMLP", "CNN4, CNN8,\nCNN sâu", "SimpleRNN,\nLSTM, GRU"]),
            ("CÀI ĐẶT", ["NumPy\ntự viết", "Keras\n(TensorFlow)", "PyTorch"]),
            ("ĐÁNH GIÁ &\nTRIỂN KHAI", ["AUC, AP, F1,\nBalanced acc.", "Bootstrap CI,\nbacktest có phí", "Demo Streamlit\n(Hugging Face)"])]
    fills = [LIGHT_BLUE, LIGHT_AQUA, LIGHT_YELLOW, LIGHT_ORANGE, BOX]
    w, gap, h = 1.2, 0.2, 0.56
    for i, ((title, items), fc) in enumerate(zip(cols, fills)):
        x = 0.05 + i * (w + gap)
        ax.text(x + w / 2, 2.5, title, ha="center", va="center", fontsize=7.8, weight="bold", color=INK2, linespacing=1.1)
        for j, item in enumerate(items):
            y = 1.62 - j * 0.74
            box(ax, x, y, w, h, item, fc=fc, size=7.2)
            if i < len(cols) - 1:
                arrow(ax, x + w + 0.02, y + h / 2, x + w + gap - 0.02, y + h / 2)
    save(fig, "diagram_study_design.png")


def neuron_mlp():
    width, height = 6.8, 2.6
    fig, ax = canvas(width, height)
    ys = [2.0, 1.3, 0.6]
    for i, y in enumerate(ys):
        node(ax, 0.35, y, 0.2, f"x{'₁₂₃'[i]}")
        arrow(ax, 0.57, y, 1.48, 1.3 + (y - 1.3) * 0.12)
        ax.text(0.98, (y + 1.3) / 2 + 0.12, f"w{'₁₂₃'[i]}", fontsize=8.5, color=BLUE, ha="center")
    node(ax, 1.75, 1.3, 0.26, "Σ", fc=LIGHT_YELLOW, ec=YELLOW, size=13)
    ax.text(1.75, 0.82, "z = w₁x₁ + w₂x₂ + w₃x₃ + b", ha="center", fontsize=7.5, color=INK2)
    arrow(ax, 2.02, 1.3, 2.32, 1.3)
    box(ax, 2.33, 1.08, 0.55, 0.44, "φ(z)", fc=LIGHT_AQUA, size=10)
    arrow(ax, 2.9, 1.3, 3.18, 1.3)
    ax.text(3.3, 1.3, "ŷ", ha="center", va="center", fontsize=12)
    ax.text(1.75, 2.48, "(a) Một nơ-ron: tổng có trọng số + hàm kích hoạt", ha="center", fontsize=7.8, color=INK2)
    layers = [(4.15, [2.0, 1.3, 0.6], LIGHT_BLUE, BLUE), (5.25, [2.15, 1.6, 1.0, 0.45], LIGHT_AQUA, AQUA),
              (6.35, [1.3], LIGHT_ORANGE, ORANGE)]
    for (x0, ys0, _, _), (x1, ys1, _, _) in zip(layers[:-1], layers[1:]):
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([x0, x1], [y0, y1], color=LINE, lw=0.7, zorder=1)
    for x, ys0, fc, ec in layers:
        for y in ys0:
            node(ax, x, y, 0.15, fc=fc, ec=ec)
    for x, label in [(4.15, "Lớp vào"), (5.25, "Lớp ẩn (ReLU)"), (6.35, "Lớp ra (sigmoid)")]:
        ax.text(x, 0.08, label, ha="center", va="center", fontsize=7.2, color=INK2)
    ax.text(5.25, 2.48, "(b) MLP: nhiều nơ-ron xếp thành lớp", ha="center", fontsize=7.8, color=INK2)
    save(fig, "diagram_neuron_mlp.png")


def grid(ax, x0, y0, values, cell, highlight=(), fc_h=LIGHT_BLUE, ec_h=BLUE, title=None):
    rows = len(values)
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            hl = (r, c) in highlight
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (rows - 1 - r) * cell), cell, cell,
                                   fc=fc_h if hl else "white", ec=ec_h if hl else INK2, lw=1.3 if hl else 0.8))
            ax.text(x0 + (c + .5) * cell, y0 + (rows - 1 - r + .5) * cell, str(v).replace("-", "−"),
                    ha="center", va="center", fontsize=9.5)
    if title:
        ax.text(x0 + len(values[0]) * cell / 2, y0 + rows * cell + 0.08, title, ha="center", va="bottom",
                fontsize=8, color=INK2)


def conv_example():
    width, height = 6.8, 1.95
    fig, ax = canvas(width, height)
    cell = 0.4
    grid(ax, 0.1, 0.42, [[1, 2, 0], [0, 1, 3], [2, 1, 0]], cell, highlight={(0, 0), (0, 1), (1, 0), (1, 1)},
         title="Đầu vào X (3×3)")
    ax.text(1.55, 1.02, "⊛", fontsize=17, ha="center", va="center")
    grid(ax, 1.9, 0.62, [[1, 0], [0, -1]], cell, highlight={(0, 0), (0, 1), (1, 0), (1, 1)}, fc_h=LIGHT_ORANGE,
         ec_h=ORANGE, title="Bộ lọc K (2×2)")
    ax.text(3.02, 1.02, "=", fontsize=15, ha="center", va="center")
    grid(ax, 3.35, 0.62, [[0, -1], [-1, 1]], cell, highlight={(0, 0)}, title="Bản đồ đặc trưng")
    arrow(ax, 4.3, 1.02, 4.85, 1.02)
    ax.text(4.57, 1.12, "ReLU", fontsize=7.8, ha="center", color=INK2)
    grid(ax, 5.0, 0.62, [[0, 0], [0, 1]], cell, title="Sau ReLU")
    ax.text(3.4, 0.12, "Ô đầu tiên = 1×1 + 2×0 + 0×0 + 1×(−1) = 0: đặt bộ lọc lên vùng tô xanh, nhân từng cặp số rồi cộng lại",
            ha="center", va="center", fontsize=7.6, color=INK2)
    save(fig, "diagram_conv_example.png")


def stack(ax, x, y, w, h, n, fc, ec, d=0.07):
    for i in reversed(range(n)):
        ax.add_patch(Rectangle((x + i * d, y + i * d), w, h, fc=fc, ec=ec, lw=0.8))
    return x + (n - 1) * d + w


def cnn_pipeline():
    width, height = 6.8, 1.9
    fig, ax = canvas(width, height)
    mid = 1.2
    ax.plot([0.08, 0.28, 0.45, 0.62, 0.8], [0.95, 1.35, 1.1, 1.5, 1.28], color=BLUE, lw=1.6)
    ax.text(0.44, 0.45, "20 giá đóng cửa\nđến ngày t", ha="center", fontsize=7, color=INK2)
    arrow(ax, 0.88, mid, 1.1, mid)
    stack(ax, 1.15, 0.85, 0.7, 0.7, 1, LIGHT_BLUE, BLUE)
    ax.text(1.5, 0.45, "Ảnh GASF\n1×20×20", ha="center", fontsize=7, color=INK2)
    arrow(ax, 1.92, mid, 2.15, mid)
    end = stack(ax, 2.2, 0.85, 0.6, 0.6, 4, LIGHT_AQUA, AQUA)
    ax.text(2.63, 0.45, "Tích chập 3×3\n4 bản đồ 18×18", ha="center", fontsize=7, color=INK2)
    arrow(ax, end + 0.07, mid, end + 0.3, mid)
    x3 = end + 0.35
    end = stack(ax, x3, 0.95, 0.38, 0.38, 4, LIGHT_YELLOW, YELLOW)
    ax.text(x3 + 0.3, 0.45, "ReLU + gộp cực\nđại 2×2 → 4×9×9", ha="center", fontsize=7, color=INK2)
    arrow(ax, end + 0.07, mid, end + 0.3, mid)
    x4 = end + 0.38
    for i in range(10):
        ax.add_patch(Rectangle((x4, 0.8 + i * 0.08), 0.12, 0.075, fc=LIGHT_ORANGE, ec=ORANGE, lw=0.5))
    ax.text(x4 + 0.06, 0.45, "Duỗi phẳng\n324 số", ha="center", fontsize=7, color=INK2)
    arrow(ax, x4 + 0.2, mid, x4 + 0.45, mid)
    box(ax, x4 + 0.5, 0.92, 1.2, 0.56, "Dense → sigmoid\nP(tăng ở t+1)", fc=BOX, size=7.4)
    save(fig, "diagram_cnn_pipeline.png")


def rnn_unrolled():
    width, height = 6.8, 2.0
    fig, ax = canvas(width, height)
    y, h, w = 0.8, 0.5, 0.75
    box(ax, 0.1, y, w, h, "Ô RNN", fc=LIGHT_AQUA, size=8)
    ax.add_patch(FancyArrowPatch((0.72, y + h), (0.23, y + h), connectionstyle="arc3,rad=1.1", arrowstyle="-|>",
                                 mutation_scale=9, color=INK2, lw=1, shrinkA=0, shrinkB=0))
    ax.text(0.475, y + h + 0.38, "hₜ₋₁", ha="center", fontsize=8, color=INK2)
    arrow(ax, 0.475, 0.35, 0.475, y)
    ax.text(0.475, 0.2, "xₜ", ha="center", fontsize=9)
    ax.text(1.15, y + h / 2, "=", fontsize=15, ha="center", va="center")
    xs = [1.65, 2.75, 3.85, 5.15]
    labels = ["x₁", "x₂", "x₃", "x₂₀"]
    ax.text(1.42, y + h / 2, "h₀", fontsize=8, ha="center", va="center", color=INK2)
    arrow(ax, 1.5, y + h / 2, xs[0], y + h / 2)
    for i, (x, lab) in enumerate(zip(xs, labels)):
        box(ax, x, y, w, h, "Ô RNN", fc=LIGHT_AQUA, size=8)
        arrow(ax, x + w / 2, 0.35, x + w / 2, y)
        ax.text(x + w / 2, 0.2, lab, ha="center", fontsize=9)
        if i == 2:
            ax.text((x + w + xs[3]) / 2, y + h / 2, "· · ·", ha="center", va="center", fontsize=11, color=INK2)
        elif i < 3:
            arrow(ax, x + w, y + h / 2, xs[i + 1], y + h / 2)
            ax.text((x + w + xs[i + 1]) / 2, y + h / 2 + 0.08, f"h{'₁₂'[i]}", ha="center", fontsize=8, color=INK2)
    arrow(ax, 5.9, y + h / 2, 6.12, y + h / 2)
    ax.text(6.45, y + h / 2, "h₂₀ → Dense\n→ P(tăng)", ha="center", va="center", fontsize=7, color=INK2)
    ax.text(3.8, 1.82, "Dạng trải theo thời gian: cùng một bộ trọng số (Wₓ, Wₕ, b) được dùng lại ở mọi bước",
            ha="center", fontsize=7.6, color=INK2)
    save(fig, "diagram_rnn_unrolled.png")


def lstm_cell():
    width, height = 6.8, 3.1
    fig, ax = canvas(width, height)
    ax.add_patch(FancyBboxPatch((0.75, 0.35), 4.9, 2.35, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#fbfbfa", ec=LINE, lw=1))
    cy, hy, gy = 2.4, 0.6, 1.15
    ax.plot([0.15, 6.6], [cy, cy], color=INK2, lw=1.8)
    ax.text(0.15, cy + 0.12, "cₜ₋₁ (bộ nhớ cũ)", fontsize=7.5)
    ax.text(6.6, cy + 0.12, "cₜ", fontsize=8.5, ha="right")
    ax.plot([0.15, 4.75], [hy, hy], color=INK2, lw=1.2)
    ax.text(0.15, hy - 0.22, "hₜ₋₁ và xₜ", fontsize=7.5)
    gates = [(1.55, "σ", "fₜ: quên"), (2.55, "σ", "iₜ: vào"), (3.35, "tanh", "gₜ: ứng viên"), (4.75, "σ", "oₜ: ra")]
    for x, s, lab in gates:
        box(ax, x - 0.36, gy - 0.24, 0.72, 0.48, f"{s}\n{lab}", fc=LIGHT_AQUA if s == "σ" else LIGHT_BLUE, size=7.2)
        ax.plot([x, x], [hy, gy - 0.24], color=INK2, lw=1)
    ax.plot([1.55, 1.55], [gy + 0.24, cy - 0.15], color=INK2, lw=1)
    node(ax, 1.55, cy, 0.15, "×", fc=LIGHT_YELLOW, ec=YELLOW)
    node(ax, 2.95, 1.85, 0.15, "×", fc=LIGHT_YELLOW, ec=YELLOW)
    ax.plot([2.55, 2.55, 2.8], [gy + 0.24, 1.85, 1.85], color=INK2, lw=1)
    ax.plot([3.35, 3.35, 3.1], [gy + 0.24, 1.85, 1.85], color=INK2, lw=1)
    ax.plot([2.95, 2.95], [2.0, cy - 0.15], color=INK2, lw=1)
    node(ax, 2.95, cy, 0.15, "+", fc=LIGHT_YELLOW, ec=YELLOW)
    ax.plot([5.3, 5.3], [cy, 2.12], color=INK2, lw=1)
    box(ax, 5.0, 1.72, 0.6, 0.4, "tanh", fc=LIGHT_BLUE, size=8)
    node(ax, 5.3, 1.25, 0.15, "×", fc=LIGHT_YELLOW, ec=YELLOW)
    ax.plot([5.3, 5.3], [1.72, 1.4], color=INK2, lw=1)
    ax.plot([4.75, 4.75, 5.15], [gy + 0.24, 1.25, 1.25], color=INK2, lw=1)
    ax.plot([5.45, 6.6], [1.25, 1.25], color=INK2, lw=1.4)
    ax.text(6.6, 1.37, "hₜ (trạng thái ra)", fontsize=7.5, ha="right")
    ax.text(3.4, 2.92, "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ          hₜ = oₜ ⊙ tanh(cₜ)", ha="center", fontsize=8.5)
    save(fig, "diagram_lstm_cell.png")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ai_timeline()
    study_design()
    neuron_mlp()
    conv_example()
    cnn_pipeline()
    rnn_unrolled()
    lstm_cell()
    print("Đã vẽ sơ đồ vào", FIGURES_DIR)


if __name__ == "__main__":
    main()
