# Tiểu luận: Trí tuệ nhân tạo trong đầu tư tài chính (Mở đầu – Chương 4)

Môn Phát triển hệ thống thông minh, PTIT. Báo cáo được sinh tự động từ các file Markdown trong `chapters/`:
mọi con số, bảng và hình đều đọc trực tiếp từ kết quả thực nghiệm đã lưu, không chép tay.

| Thành phần | Vị trí |
|---|---|
| Nội dung từng chương (nguồn) | `docs/tieuluan/chapters/00_mo_dau.md` … `06_phu_luc.md` |
| Danh mục tài liệu tham khảo (IEEE) | `docs/tieuluan/references.json` |
| Mã nguồn thực nghiệm | `src/tieuluan/`, `scripts/tieuluan/` |
| Kết quả, mô hình | `results/tieuluan/`, `models/tieuluan/` |
| Notebook minh họa | `notebooks/tieuluan/` |
| Ứng dụng minh họa (Streamlit) | `deployment/tieuluan/` |

Tạo lại báo cáo Word (Windows, cần Microsoft Word để cập nhật mục lục):

```powershell
.venv\Scripts\python.exe -m scripts.tieuluan.build_report
.venv\Scripts\python.exe -m scripts.tieuluan.finalize_docx
```

File `tieuluan01_NhomLop_NhomTL_TrungNT.docx` được tạo tại thư mục này. File `.docx`/`.pdf` không được đưa lên git
(xem `.gitignore`); bản nộp được xuất riêng. Trình tự tái lập toàn bộ thực nghiệm nằm ở Phụ lục A của báo cáo.
