<!--
Nguồn nội dung có thể chỉnh sửa của TECHNICAL_REPORT.docx.
Tạo lại: .venv\Scripts\python.exe docs\report\build_report.py

TRANG BÌA
PHÁT TRIỂN HỆ THỐNG THÔNG MINH
BÀI TẬP 01
Từ biểu diễn dữ liệu đến ứng dụng thông minh

Các hệ thống:
1. Hệ thống phân loại tiểu đường
2. Hệ thống dự đoán giá nhà Việt Nam

Sinh viên: Nguyễn Thành Trung
Mã sinh viên: B23DCCN861
Lớp: D23CTPM01-B
Giảng viên: _________________________________
Thời gian: Tháng 8 năm 2026
-->

<!-- REPORT BODY -->

# 1. Giới thiệu

Bài tập 01 xây dựng hai hệ thống thông minh hoàn chỉnh từ dữ liệu thực: hệ thống phân loại tiểu đường nhị phân và hệ thống dự đoán giá nhà Việt Nam. Cả hai tuân theo tiến trình **Bài toán thực tế → Dữ liệu → Biểu diễn → Học máy truyền thống → Thí nghiệm có kiểm soát → Mô hình cuối cùng → Ứng dụng thông minh**. Hệ thống thứ nhất ánh xạ sáu chỉ số thành dự đoán lớp phục vụ giáo dục; hệ thống thứ hai ánh xạ mười một thuộc tính bất động sản thành giá niêm yết ước tính theo tỷ VND.

Mục tiêu không chỉ là đạt chỉ số cao mà còn hiểu cách biểu diễn quan sát thô, xử lý dữ liệu thiếu, so sánh các họ mô hình và lựa chọn cấu hình có căn cứ. Tập test độc lập chỉ dùng để đánh giá cuối cùng; cross-validation trên tập train dùng để chọn cấu hình. Tiền xử lý và bộ ước lượng được lưu chung trong `Pipeline` scikit-learn, bảo đảm web, mobile và API không tự tạo quy trình tiền xử lý khác.

Tầng ứng dụng gồm FastAPI, React/Vite, Expo React Native và Knowledge Graph Neo4j cho mô hình tiểu đường. Render chạy API, Vercel chạy web, Neo4j AuraDB lưu đồ thị và Expo Go phục vụ demo mobile. Đây là hệ thống giáo dục: đầu ra tiểu đường **không phải chẩn đoán y khoa**, còn đầu ra giá nhà không phải định giá chuyên nghiệp.

![Hình 1. Vòng đời học máy từ dữ liệu huấn luyện đến suy luận bằng Pipeline đã lưu.](assets/ml_pipeline.png)

# 2. Định nghĩa hệ thống thông minh

Trong bài tập, hệ thống thông minh kết hợp mô hình đã học với biểu diễn dữ liệu, tiền xử lý, đánh giá, hợp đồng suy luận và phần mềm giao tiếp người dùng. Hệ thống vì vậy không chỉ là `model.fit()`: ý nghĩa đầu vào phải rõ ràng và đúng các phép biến đổi đã fit phải được tái sử dụng khi suy luận.

**Bảng 1. Tổng quan hai hệ thống.**

| Hệ thống | Bài toán | Đầu vào thô | Đầu ra | Bộ ước lượng cuối |
|---|---|---|---|---|
| Phân loại tiểu đường | Phân loại nhị phân | Sáu thuộc tính số | Lớp 0/1, nhãn, xác suất lớp 1 | Random Forest Classifier |
| Dự đoán giá nhà | Hồi quy | Mười một thuộc tính số/phân loại | Giá niêm yết theo tỷ VND | Random Forest Regressor |

## 2.1 Hệ thống phân loại tiểu đường

Hệ thống nhận `Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, `DiabetesPedigreeFunction`, `Age` và dự đoán `Outcome`. Lớp 0 được gắn nhãn Non-diabetic, lớp 1 là Diabetic. Kết quả chỉ mô tả mô hình bài tập, chưa được kiểm định lâm sàng và không được dùng làm chẩn đoán, hướng dẫn sàng lọc hay quyết định y tế.

## 2.2 Hệ thống dự đoán giá nhà Việt Nam

Hệ thống nhận `Province`, `Area`, `Frontage`, `Access Road`, `House direction`, `Balcony direction`, `Floors`, `Bedrooms`, `Bathrooms`, `Legal status`, `Furniture state` và ước tính `Price` theo tỷ VND. Dự đoán phản ánh quan hệ học từ giá niêm yết năm 2024, không bảo đảm giá giao dịch.

# 3. Phát biểu bài toán

## 3.1 Bài toán phân loại

Với quan sát thứ *i*, vector xᵢ ∈ ℝ⁶ và yᵢ ∈ {0,1}; bộ phân loại học ŷ = fθ(x). Đánh giá báo cáo Accuracy, Precision, Recall, F1-score và ma trận nhầm lẫn. Accuracy đo tỷ lệ đúng tổng thể; Precision đo độ tinh khiết của dự đoán dương; Recall đo phần trường hợp dương thực tế được phát hiện; F1 là trung bình điều hòa của Precision và Recall.

Dữ liệu mất cân bằng: 500 quan sát (65,1%) thuộc lớp 0 và 268 (34,9%) thuộc lớp 1. Vì vậy Accuracy phải đi kèm các chỉ số lớp dương. Quy tắc luôn dự đoán lớp đa số có thể có Accuracy tương đối cao nhưng bỏ sót mọi trường hợp dương.

## 3.2 Bài toán hồi quy

Với bất động sản thứ *i*, xᵢ gồm thuộc tính số và phân loại, yᵢ ∈ ℝ là `Price`. Mô hình học ŷ = fθ(x) và được đánh giá bằng MAE, MSE, RMSE, R², MAPE. MAE và RMSE có đơn vị tỷ VND; MSE có đơn vị bình phương tỷ VND. Hồi quy không dùng Accuracy. R² đo phần phương sai được giải thích so với dự đoán trung bình, không phải tỷ lệ dự đoán đúng.

# 4. Dữ liệu

## 4.1 Bộ dữ liệu tiểu đường

Notebook đọc `data/diabetes/diabetes.csv`: 768 quan sát, chín cột gồm tám đầu vào và `Outcome`. Không có `NaN` tường minh hay dòng trùng, nhưng một số cột sinh lý chứa 0 không hợp lý và phải được xem là phép đo thiếu. Glucose có tương quan tuyệt đối lớn nhất với Outcome trong sáu thuộc tính được chọn (0.4947), tiếp theo là BMI 0.3137, Age 0.2384, Pregnancies 0.2219, DiabetesPedigreeFunction 0.1738 và BloodPressure 0.1706. Tương quan chỉ mang tính mô tả, không chứng minh nhân quả.

**Bảng 2. Giá trị 0 ẩn trong dữ liệu tiểu đường.**

| Phép đo | Số giá trị 0 | Tỷ lệ | Xử lý |
|---|---:|---:|---|
| Glucose | 5 | 0.65% | Chuyển thành thiếu |
| BloodPressure | 35 | 4.56% | Chuyển thành thiếu |
| SkinThickness | 227 | 29.56% | Chuyển thành thiếu |
| Insulin | 374 | 48.70% | Chuyển thành thiếu |
| BMI | 11 | 1.43% | Chuyển thành thiếu |

`Pregnancies = 0` được giữ vì có ý nghĩa. SkinThickness và Insulin không được chọn do tỷ lệ thiếu cao và thí nghiệm biểu diễn không cải thiện khi thêm chúng. Điều này **không có nghĩa** hai biến không quan trọng về y khoa.

## 4.2 Bộ dữ liệu nhà ở Việt Nam 2024

Notebook đọc `data/house_price/vietnam_housing_dataset.csv`, Vietnam Housing Dataset 2024 trên Kaggle, gồm 30,229 dòng và 12 cột. Mười một đầu vào ứng viên là Address, Area, Frontage, Access Road, House direction, Balcony direction, Floors, Bedrooms, Bathrooms, Legal status, Furniture state; đích `Price` có đơn vị tỷ VND.

**Bảng 3. Lược đồ dữ liệu nhà ở.**

| Trường | Kiểu | Tỷ lệ thiếu | Miền quan sát |
|---|---|---:|---|
| Address | Văn bản | 0.00% | 10,265 chuỗi |
| Area | Số | 0.00% | 3.1–595.0 m² |
| Frontage | Số | 38.25% | 1.0–77.0 m |
| Access Road | Số | 43.99% | 1.0–85.0 m |
| House direction | Phân loại | 70.26% | 8 nhóm |
| Balcony direction | Phân loại | 82.65% | 8 nhóm |
| Floors | Số/rời rạc | 11.92% | 1–10 |
| Bedrooms | Số/rời rạc | 17.08% | 1–9 |
| Bathrooms | Số/rời rạc | 23.40% | 1–9 |
| Legal status | Phân loại | 14.91% | 2 nhóm |
| Furniture state | Phân loại | 46.71% | 2 nhóm |
| Price | Đích số | 0.00% | 1.0–11.5 tỷ VND |

Không có dòng trùng, Price thiếu hay Price không dương; không dòng nào bị xóa. Address được rút thành `Province` bằng thành phần cuối sau dấu phẩy, tạo 60 nhãn tỉnh/thành và không thiếu Province. Ba hậu tố phi địa lý được giữ là `Unknown` thay vì xóa dòng.

**Bảng 4. Thống kê biến đích và phân bố địa lý.**

| Thống kê | Giá trị |
|---|---:|
| Dòng được giữ | 30,229/30,229 |
| Price trung bình | 5.8721 tỷ VND |
| Price trung vị | 5.9000 tỷ VND |
| Độ lệch Price | -0.0290 |
| Area trung bình | 68.4987 m² |
| Area trung vị | 56.0 m² |
| Tin đăng Hồ Chí Minh | 11,788 |
| Tin đăng Hà Nội | 10,464 |

# 5. Biểu diễn dữ liệu

## 5.1 Biểu diễn tiểu đường

Vector sáu chiều có thứ tự `[Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age]`. Các giá trị 0 không hợp lý ở cột phép đo được chuyển thành thiếu; với biểu diễn cuối, thao tác ảnh hưởng Glucose, BloodPressure, BMI. Trung vị và tham số scale chỉ được học trong Pipeline. Thí nghiệm 6 so với 8 thuộc tính giữ nguyên mô hình và đánh giá để đo riêng tác động của SkinThickness, Insulin.

## 5.2 Biểu diễn giá nhà

Biểu diễn gọn gồm `Province`, `Area`, `Floors`, `Bedrooms`, `Bathrooms`, `Legal status`. Thí nghiệm 3 đánh giá và chọn đủ 11 trường sử dụng được: Province, Area, Frontage, Access Road, House direction, Balcony direction, Floors, Bedrooms, Bathrooms, Legal status, Furniture state. Address bị loại vì 10,265 chuỗi gây cardinality cao và hợp đồng giao diện không ổn định.

**Thuộc tính thô** là trường người dùng cung cấp; **thuộc tính mã hóa** là cột số phát sinh như `categorical__Province_Hồ Chí Minh`; **vector số cuối** là toàn bộ ma trận sau điền thiếu, scale, one-hot encoding. Do đó thuộc tính thô ≠ thuộc tính mã hóa ≠ vector số cuối.

# 6. Phân tích khám phá dữ liệu

EDA tiểu đường xác nhận mất cân bằng 65,1%/34,9% và vấn đề 0 ẩn. Sau xử lý, Glucose có 763 phép đo hợp lệ, trung bình 121.69, trung vị 117, skewness 0.531; BMI có 757 phép đo, trung bình 32.46, trung vị 32.30, skewness 0.594. Age từ 21–81 và lệch phải 1.13; BloodPressure hợp lệ có trung bình 72.41, trung vị 72. Kết quả ủng hộ điền thiếu bằng trung vị thay vì xóa dòng.

Price gần đối xứng trên thang gốc: trung bình gần trung vị, skewness -0.0290, nên notebook không log-transform đích. Area lệch phải mạnh 3.8885 với đuôi đến 595 m²; giá trị lớn vẫn được giữ vì có thể hợp lệ. Floors có trung vị 3, khoảng 1–10. Dữ liệu tập trung ở Hồ Chí Minh và Hà Nội nên đánh giá cho tỉnh ít mẫu còn bất định.

# 7. Tiền xử lý dữ liệu

Pipeline tiểu đường dùng `SimpleImputer(strategy="median")`, `StandardScaler` rồi bộ phân loại sau khi biểu diễn 0 không hợp lý thành thiếu. Dù cây không cần scale như KNN/SVM, pipeline chung giữ giao diện thí nghiệm nhất quán.

Giá nhà dùng `ColumnTransformer`. Nhánh số: điền trung vị → `StandardScaler`. Nhánh phân loại: điền giá trị xuất hiện nhiều nhất → `OneHotEncoder(handle_unknown="ignore")`. Không dùng `LabelEncoder`. Tiền xử lý nằm trong từng Pipeline và từng fold CV, ngăn học medians, modes, scales hay categories từ validation/test.

# 8. Các mô hình học máy truyền thống

## 8.1 Mô hình phân loại

Logistic Regression học biên log-odds tuyến tính trên vector đã scale, dễ diễn giải nhưng hạn chế với tương tác phi tuyến. KNN dựa trên hàng xóm; `n_neighbors` điều khiển tính cục bộ, còn suy luận tốn chi phí và nhạy với scale. Decision Tree học luật chia theo trục; depth kiểm soát độ phức tạp nhưng cây đơn dễ overfit.

Random Forest tổng hợp nhiều cây bootstrap; `n_estimators`, `max_depth`, `random_state` là tham số chính. Mô hình nắm tương tác phi tuyến và ổn định hơn cây đơn, nhưng impurity importance không mang nghĩa nhân quả. SVM học biên margin lớn, có thể phi tuyến qua kernel, hưởng lợi từ scale nhưng nhạy cấu hình và khó diễn giải.

## 8.2 Mô hình hồi quy

Linear Regression học quan hệ cộng tuyến tính trên biểu diễn số/one-hot. KNN Regressor lấy trung bình mẫu gần, nhưng gặp hạn chế trong không gian one-hot nhiều chiều. Decision Tree Regressor dự đoán trung bình theo vùng, dễ có phương sai cao nếu không giới hạn sâu. Random Forest Regressor trung bình hóa nhiều cây, xử lý phi tuyến tốt nhưng có xu hướng làm phẳng cực trị và khó ngoại suy. SVR RBF học hàm phi tuyến với `C`, `gamma`, `epsilon`; kết quả mạnh nhưng chi phí và khả năng diễn giải kém thuận tiện hơn forest được chọn.

# 9. Thiết kế thí nghiệm

Hai notebook chia train/test 80/20 với `random_state=42`. Tiểu đường stratify theo Outcome: 614 train, 154 test. Giá nhà: 24,183 train, 6,046 test. Tập test không dùng chọn siêu tham số hoặc biểu diễn.

## 9.1 Thí nghiệm 1 – So sánh mô hình

Chỉ thuật toán học thay đổi; dữ liệu, biểu diễn, tiền xử lý và split giữ nguyên. Bảng holdout ban đầu mang tính mô tả. CV năm fold chỉ trên train đánh giá độ ổn định: tiểu đường dùng F1, giá nhà dùng RMSE với `KFold(n_splits=5, random_state=42)` có shuffle.

## 9.2 Thí nghiệm 2 – Khảo sát siêu tham số

Tiểu đường thay `max_depth` qua 2, 4, 6, 8, None và đánh giá F1 CV. Giá nhà thử 4, 8, 12, 16, None theo RMSE CV. `n_estimators=100`, `random_state=42` giữ cố định.

## 9.3 Thí nghiệm 3 – Biểu diễn thuộc tính

Tiểu đường so sánh 6 và 8 đầu vào dưới forest đã tinh chỉnh. Giá nhà so sánh 6 và 11 trường dưới forest depth 12. Chỉ biểu diễn thay đổi, giúp cô lập ảnh hưởng thông tin và missingness.

# 10. Kết quả thí nghiệm

## 10.1 Kết quả tiểu đường

`DummyClassifier(strategy="most_frequent")` luôn dự đoán lớp 0. Baseline đạt Accuracy 0.6494 vì 100/154 mẫu test là lớp 0, nhưng Precision, Recall, F1 lớp 1 đều 0; ma trận `[[100, 0], [54, 0]]`. Accuracy đơn lẻ vì vậy không đủ.

**Bảng 5. Baseline và holdout năm mô hình tiểu đường.**

| Mô hình | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline | 0.6494 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.7013 | 0.5909 | 0.4815 | 0.5306 |
| KNN | 0.7338 | 0.6327 | 0.5741 | 0.6019 |
| Decision Tree | 0.6818 | 0.5510 | 0.5000 | 0.5243 |
| Random Forest | 0.7597 | 0.6735 | 0.6111 | 0.6408 |
| SVM | 0.7338 | 0.6512 | 0.5185 | 0.5773 |

Random Forest mạnh nhất trên holdout ban đầu, nhưng KNN đứng đầu CV của năm cấu hình gốc. Một holdout duy nhất không nên là quy tắc lựa chọn duy nhất.

**Bảng 6. CV năm mô hình tiểu đường trên train.**

| Mô hình | F1 CV trung bình | Độ lệch chuẩn |
|---|---:|---:|
| KNN | 0.6404 | 0.0503 |
| Logistic Regression | 0.6393 | 0.0316 |
| SVM | 0.6377 | 0.0367 |
| Random Forest (không giới hạn) | 0.6227 | 0.0398 |
| Decision Tree | 0.5447 | 0.0194 |

![Hình 2. Phân bố F1 năm fold của các cấu hình phân loại tiểu đường.](../../figures/diabetes/model_cv_f1_boxplot.png)

**Bảng 7. Thí nghiệm 2 tiểu đường: max_depth.**

| max_depth | F1 CV trung bình |
|---:|---:|
| 2 | 0.5513 |
| 4 | 0.6224 |
| 6 | **0.6462** |
| 8 | 0.6403 |
| None | 0.6227 |

![Hình 3. Độ sâu tối đa và F1 CV trung bình của Random Forest tiểu đường.](../../figures/diabetes/experiment_2_max_depth.png)

Depth 6 có F1 CV tốt nhất 0.6462. Khi đánh giá test, F1 bằng 0.6061, thấp hơn F1 holdout 0.6408 của forest không giới hạn. Đây không phải lỗi: depth được chọn bằng CV trên train, còn test là mẫu hữu hạn độc lập. Quay lại chọn theo test sẽ gây rò rỉ thông tin.

**Bảng 8. Thí nghiệm 3 tiểu đường: biểu diễn thuộc tính.**

| Biểu diễn | Thuộc tính thô | F1 CV trung bình | Độ lệch chuẩn |
|---|---:|---:|---:|
| Biểu diễn được chọn | 6 | **0.6462** | 0.0231 |
| Toàn bộ đầu vào | 8 | 0.6243 | 0.0433 |

![Hình 4. Biểu diễn sáu so với tám thuộc tính tiểu đường.](../../figures/diabetes/experiment_3_feature_representation.png)

Thêm SkinThickness và Insulin không cải thiện F1 trung bình hay độ ổn định. Nhiều thuộc tính hơn không nhất thiết tốt hơn; kết quả không chứng minh hai phép đo vốn không quan trọng.

## 10.2 Kết quả giá nhà

`DummyRegressor(strategy="mean")` đạt MAE 1.8438, MSE 4.8760, RMSE 2.2082 tỷ VND, R² xấp xỉ 0, MAPE 44.43%. R² gần 0 là dự kiến vì baseline không dùng thuộc tính nhà.

**Bảng 9. Baseline và holdout năm mô hình giá nhà.**

| Mô hình | MAE | MSE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.8438 | 4.8760 | 2.2082 | -0.0000 | 44.43% |
| Linear Regression | 1.4782 | 3.4039 | 1.8450 | 0.3019 | 32.93% |
| KNN | 1.3627 | 3.0777 | 1.7543 | 0.3688 | 29.31% |
| Decision Tree | 1.3976 | 3.6669 | 1.9149 | 0.2480 | 29.47% |
| Random Forest | 1.3009 | 2.9549 | 1.7190 | 0.3940 | 27.84% |
| SVR | 1.2868 | 2.7615 | 1.6618 | 0.4336 | 27.31% |

![Hình 5. So sánh RMSE và R² holdout của năm mô hình giá nhà.](../../figures/house_price/model_comparison.png)

SVR mạnh nhất ở holdout và trong CV các cấu hình gốc.

**Bảng 10. Thí nghiệm 1 giá nhà: CV năm mô hình.**

| Mô hình | RMSE CV trung bình | Độ lệch chuẩn |
|---|---:|---:|
| SVR | **1.6565** | 0.0120 |
| Random Forest | 1.7130 | 0.0277 |
| KNN | 1.7601 | 0.0187 |
| Linear Regression | 1.8539 | 0.0239 |
| Decision Tree | 1.9365 | 0.0381 |

![Hình 6. Phân bố RMSE năm fold của các mô hình hồi quy gốc.](../../figures/house_price/model_cv_rmse_boxplot.png)

**Bảng 11. Thí nghiệm 2 giá nhà: max_depth.**

| max_depth | RMSE CV trung bình | Độ lệch chuẩn |
|---:|---:|---:|
| 4 | 1.8191 | 0.0109 |
| 8 | 1.6843 | 0.0138 |
| 12 | **1.6526** | 0.0189 |
| 16 | 1.6683 | 0.0258 |
| None | 1.7130 | 0.0277 |

![Hình 7. Độ sâu tối đa và RMSE CV trung bình của Random Forest giá nhà.](../../figures/house_price/experiment_2_hyperparameter.png)

Depth 12 cải thiện SVR CV chỉ 0.0039 tỷ VND; chênh lệch nhỏ không nên phóng đại. Depth 4 underfit; forest sâu hơn tăng RMSE và độ biến thiên.

**Bảng 12. Thí nghiệm 3 giá nhà: biểu diễn thuộc tính.**

| Biểu diễn | Thuộc tính thô | RMSE CV trung bình | Độ lệch chuẩn |
|---|---:|---:|---:|
| Biểu diễn gọn | 6 | 1.6526 | 0.0189 |
| Toàn bộ thuộc tính dùng được | 11 | **1.6078** | 0.0231 |

![Hình 8. Biểu diễn sáu so với mười một thuộc tính giá nhà.](../../figures/house_price/experiment_3_feature_representation.png)

Khác tiểu đường, Frontage, Access Road, hai trường hướng và Furniture state cải thiện biểu diễn, giảm RMSE CV khoảng 0.0448 tỷ VND. Pipeline tận dụng thông tin quan sát được mà không xóa dòng.

# 11. Lựa chọn mô hình cuối cùng

## 11.1 Mô hình tiểu đường cuối cùng

Mô hình là `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)` với sáu thuộc tính. Pipeline: điền thiếu trung vị → `StandardScaler` → forest. Lựa chọn dựa trên F1 CV train và biểu diễn sáu thuộc tính; test chỉ dùng đánh giá cuối.

**Bảng 13. Chỉ số test cuối của mô hình tiểu đường.**

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.7468 |
| Precision | 0.6667 |
| Recall | 0.5556 |
| F1-score | 0.6061 |

Ma trận `[[85, 15], [24, 30]]`: TN=85, FP=15, FN=24, TP=30. Có 24 mẫu lớp 1 thực tế bị dự đoán lớp 0, giải thích Recall 0.5556. Đây là lỗi phân loại giáo dục, không phải kết quả lâm sàng.

![Hình 9. Ma trận nhầm lẫn của mô hình tiểu đường cuối.](../../figures/diabetes/final_confusion_matrix.png)

Importance theo impurity: Glucose 0.4017, BMI 0.1985, Age 0.1344, DiabetesPedigreeFunction 0.1244, Pregnancies 0.0767, BloodPressure 0.0643. Chúng mô tả phép chia của mô hình đã fit, không mang ý nghĩa nhân quả/lâm sàng.

![Hình 10. Importance thuộc tính thô của Random Forest tiểu đường.](../../figures/diabetes/final_feature_importance.png)

## 11.2 Mô hình giá nhà cuối cùng

Mô hình là `RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)` với 11 trường thô. Pipeline gồm nhánh số điền trung vị/scale, nhánh phân loại điền mode/one-hot và forest.

**Bảng 14. Chỉ số test cuối của mô hình giá nhà.**

| Chỉ số | Giá trị |
|---|---:|
| MAE | 1.2529 tỷ VND |
| MSE | 2.5659 (tỷ VND)² |
| RMSE | 1.6018 tỷ VND |
| R² | 0.4738 |
| MAPE | 27.36% |

**R² = 0.4738 không có nghĩa Accuracy bằng 47.38%.** Nó cho biết mô hình giải thích khoảng 47.38% phương sai giá test so với baseline trung bình. Hồi quy không dùng Accuracy; MAE/RMSE cho thấy sai số từng tin đăng vẫn đáng kể.

![Hình 11. Giá thực tế so với giá dự đoán của mô hình giá nhà cuối.](../../figures/house_price/final_actual_vs_predicted.png)

Dự đoán theo xu hướng y=x nhưng có miền hẹp hơn giá thực tế. Residual trung bình -0.0255 tỷ VND, trung vị -0.0803, độ lệch chuẩn 1.6016 và khoảng -6.2601 đến 5.8129 tỷ VND.

![Hình 12. Residual theo giá nhà dự đoán.](../../figures/house_price/final_residual_plot.png)

Importance mã hóa lớn nhất: Area 0.2856, Bathrooms 0.2226, Floors 0.1094, Access Road 0.0770, chỉ báo Province Hồ Chí Minh 0.0676. Đây là giá trị của cột đã mã hóa, không bị cộng sai thành tổng biến thô.

![Hình 13. Hai mươi thuộc tính mã hóa quan trọng nhất của Random Forest giá nhà.](../../figures/house_price/final_feature_importance.png)

# 12. Kiến trúc ứng dụng thông minh

Ứng dụng tách artifact khoa học dữ liệu khỏi suy luận và trình bày. Notebook đã chạy tạo Pipeline đáng tin cậy trong `models/`. FastAPI nạp hai mô hình một lần, đối chiếu hợp đồng thuộc tính thô với metadata JSON và tạo DataFrame một dòng theo đúng `feature_names_in_`. API không fit lại hay tự dựng lại tiền xử lý.

React/Vite và Expo React Native là client mỏng: lấy metadata, dựng trường nhập, gửi giá trị thô và hiển thị dự đoán cùng giới hạn. Neo4j độc lập với prediction; lỗi graph không ngăn hai mô hình suy luận.

![Hình 14. Kiến trúc triển khai qua Vercel, Render, Pipeline đã lưu, Expo Go và Neo4j AuraDB.](assets/system_architecture.png)

# 13. Backend và suy luận mô hình

Backend dùng Python, FastAPI, Pydantic, pandas, joblib, scikit-learn Pipeline. Pydantic từ chối trường thiếu/thừa, dấu không hợp lệ, chuỗi rỗng, `NaN`, vô cực. Request được giới hạn kích thước, CORS theo biến môi trường và lỗi mô hình không trả raw traceback. Chỉ joblib do repository kiểm soát được nạp.

Bất biến trung tâm là **Pipeline huấn luyện = Pipeline suy luận**. Tiểu đường gọi `predict`, `predict_proba`; giá nhà chuyển trường nullable thành `numpy.nan` để Pipeline đã fit tự điền thiếu. Backend không gọi `fit`, `fit_transform`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder` khi dự đoán.

**Bảng 15. Các endpoint FastAPI.**

| Phương thức | Endpoint | Mục đích | Phản hồi chính |
|---|---|---|---|
| GET | `/health` | Trạng thái dịch vụ, mô hình, Neo4j | Đối tượng trạng thái |
| GET | `/api/v1/models` | Metadata hai hệ thống | Danh sách mô hình |
| GET | `/api/v1/models/diabetes` | Hợp đồng mô hình tiểu đường | Metadata |
| GET | `/api/v1/models/house-price` | Hợp đồng mô hình giá nhà | Metadata |
| POST | `/api/v1/diabetes/predict` | Phân loại giáo dục | Lớp, nhãn, xác suất |
| POST | `/api/v1/house-price/predict` | Ước tính giá niêm yết | Giá, đơn vị |
| GET | `/api/v1/diabetes/knowledge-graph` | Graph mô hình/nguồn gốc | Node, edge |

# 14. Ứng dụng web

Web dùng React 19, Vite 7, TypeScript với các route Home, Diabetes, House Price, Knowledge Graph, About. Metadata chung ngăn hard-code thuộc tính mã hóa: form gửi đúng tên **thuộc tính thô**, còn Pipeline backend thực hiện điền thiếu, scale, encoding. Breakpoint responsive điều chỉnh navigation, form, kết quả, card, typography và graph cho desktop, tablet, điện thoại; màn hình nhỏ dùng hamburger và grid một cột để tránh tràn ngang.

Web production: <https://intelligent-system-assignment-01.vercel.app>. `VITE_API_BASE_URL` cung cấp backend origin lúc build. Trang Knowledge Graph lấy node/edge từ FastAPI và hiển thị canvas force-directed; mọi node có vùng click mở rộng, hit-test dự phòng chuột/cảm ứng và panel thuộc tính đồng bộ.

# 15. Ứng dụng mobile

Mobile dùng Expo, React Native, TypeScript với năm màn hình Home, Diabetes, House Price, Knowledge Graph, About. Diabetes và House lấy metadata, gửi cùng hợp đồng API thô như web. Knowledge Graph gọi endpoint hiện có, hiển thị tổng node/relationship, chú giải nhãn, thuộc tính node và kết nối điều hướng theo giao diện native. `EXPO_PUBLIC_API_BASE_URL` chọn backend cho trình duyệt, emulator, thiết bị LAN hoặc API đã triển khai.

Mobile được demo qua Expo Go; repository không tuyên bố phát hành Google Play hay App Store. Code TypeScript type-check xác nhận đường gọi prediction và graph, nhưng báo cáo không tuyên bố bản mobile production độc lập.

# 16. Knowledge Graph tiểu đường

Neo4j AuraDB lưu graph minh bạch, hướng mô hình: mô hình cuối, sáu thuộc tính, target, bước Pipeline, dataset, thí nghiệm chọn mô hình, chỉ số holdout và impurity importance. Đây **không phải cơ sở tri thức y khoa**.

Web dùng canvas force-directed và panel chi tiết; mobile dùng trình khám phá node hướng cảm ứng và danh sách kết nối. Cả hai dùng cùng phản hồi FastAPI và cùng ngữ nghĩa 17 node/17 relationship.

**Bảng 16. Lược đồ Knowledge Graph trong Cypher seed.**

| Nhóm | Giá trị |
|---|---|
| Nhãn node | System; Model; Feature; Target; PipelineStep; Dataset; Experiment; Metric |
| Relationship | USES_MODEL; USES_FEATURE; PREDICTS; HAS_PIPELINE_STEP; TRAINED_ON; EVALUATED_BY; SELECTED; COMPARES_REPRESENTATION |
| Phạm vi | `domain = "diabetes_assignment_01"` |
| Nguồn gốc | 768 quan sát; CV 5 fold chỉ trên train; chỉ số holdout |
| Ranh giới | Minh bạch mô hình, không phải tri thức y khoa |

Cypher dùng `MERGE` và ràng buộc duy nhất trên key AssignmentEntity. Seed production hai lần đều cho tổng 17 node và 17 relationship: 17/17 rồi 17/17, chứng minh idempotent. API production cũng trả 17 node và 17 relationship.

# 17. Kiến trúc triển khai

GitHub lưu source tại <https://github.com/Sagitoaz/intelligent-system-assignment-01>. Render chạy FastAPI tại <https://intelligent-system-assignment-01.onrender.com>. Vercel chạy React tại <https://intelligent-system-assignment-01.vercel.app>. Neo4j AuraDB lưu graph, Expo Go phục vụ demo mobile. Báo cáo loại bỏ mật khẩu, credential và định danh bí mật.

**Bảng 17. Kiểm tra production trực tiếp.**

| Kiểm tra | Kết quả |
|---|---|
| `GET /health` | HTTP 200; status=ok; diabetes=loaded; house_price=loaded; neo4j=available |
| POST demo tiểu đường | HTTP 200; lớp 0; Non-diabetic; xác suất 0.01054453459068126 |
| POST demo giá nhà | HTTP 200; 5.1266618454336434 tỷ VND; định dạng 5.13 tỷ VND |
| GET Knowledge Graph | HTTP 200; 17 node; 17 relationship |
| Trang gốc Vercel | HTTP 200; trả HTML ứng dụng |

Các kiểm tra xác nhận prediction, graph và web tại thời điểm tạo báo cáo. Mobile dùng cùng HTTPS endpoint; chưa có kiểm thử production mobile có instrument độc lập ngoài bằng chứng code và Expo Go. Free-tier có thể cold-start nên request đầu chậm hơn.

# 18. Minh họa hệ thống

Notebook chứa ba ca tổng hợp cho mỗi hệ thống. Chúng nằm trong miền quan sát, không sao chép dòng holdout và đi thẳng qua Pipeline đã lưu.

**Bảng 18. Ca demo tiểu đường và đầu ra.**

| Ca | Pregnancies | Glucose | BloodPressure | BMI | DPF | Age | Lớp | Xác suất lớp 1 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 1 | 85 | 66 | 24.0 | 0.20 | 23 | 0 – Non-diabetic | 0.0105 |
| 2 | 4 | 125 | 72 | 32.0 | 0.50 | 35 | 0 – Non-diabetic | 0.4105 |
| 3 | 8 | 180 | 80 | 38.0 | 1.20 | 55 | 1 – Diabetic | 0.8462 |

Nhãn chỉ minh họa cơ chế phân loại, không phải chẩn đoán hay lời khuyên y tế.

**Bảng 19. Ca demo giá nhà và đầu ra.**

| Ca | Province | Area | Frontage | Access Road | Hướng nhà/ban công | Tầng/PN/PT | Pháp lý/Nội thất | Giá dự đoán |
|---|---|---:|---:|---:|---|---|---|---:|
| 1 | Hồ Chí Minh | 45 | 4 | 4 | Đông - Nam / Đông - Nam | 3 / 3 / 3 | Have certificate / Full | 5.1267 tỷ VND |
| 2 | Bình Dương | 80 | 5 | 8 | Nam / Nam | 2 / 3 / 2 | Have certificate / Basic | 3.0945 tỷ VND |
| 3 | Hưng Yên | 90 | 6 | 13 | Đông - Bắc / Đông - Bắc | 5 / 5 / 5 | Sale contract / Full | 8.6189 tỷ VND |

Đây là ước tính giáo dục, không phải giao dịch, thẩm định hay tư vấn đầu tư. Reload test xác nhận model đã lưu cho cùng kết quả với Pipeline cuối trong notebook.

Ảnh giao diện thực tế cần chụp đúng bản chạy thật: web dự đoán tiểu đường, web dự đoán giá nhà, web Knowledge Graph có panel chi tiết, mobile prediction và mobile Knowledge Graph. Quy cách và dữ liệu nằm trong `docs/report/SCREENSHOT_GUIDE.md`; báo cáo không dùng mockup làm bằng chứng chạy hệ thống.

# 19. Hạn chế

Dữ liệu tiểu đường chỉ có 768 quan sát, độ phủ quần thể hạn chế, missing bị mã hóa 0 và target mất cân bằng. Recall 0.5556, F1 0.6061 còn thấp. Chưa có calibration, fairness, đánh giá an toàn, external validation hay kiểm định lâm sàng. Hệ thống chỉ phục vụ giáo dục.

Dữ liệu nhà là tin đăng 2024, tập trung ở Hồ Chí Minh và Hà Nội, nhiều thuộc tính thiếu. Province trích từ Address chưa thay thế geocoding; giá niêm yết khác giá giao dịch; thị trường thay đổi. RMSE 1.6018 tỷ VND vẫn đáng kể nên không thể dùng như định giá chuyên nghiệp.

Ứng dụng phụ thuộc mạng và free-tier có cold-start. Graph được tách khỏi prediction; Expo Go là môi trường demo, không phải app-store deployment. Hệ thống không tuyên bố sẵn sàng lâm sàng, thương mại hay production.

**Bảng 20. Nhóm hạn chế chính.**

| Hệ thống | Dữ liệu | Mô hình/đánh giá | Ranh giới sử dụng |
|---|---|---|---|
| Tiểu đường | 768 dòng; 0 ẩn; giới hạn quần thể; mất cân bằng | Recall/F1 hạn chế; chưa validation ngoài/lâm sàng | Không chẩn đoán y khoa |
| Giá nhà | Tin đăng 2024; tập trung; thiếu trường | RMSE 1.6018; listing ≠ transaction; market drift | Không định giá chuyên nghiệp |
| Ứng dụng | Free-tier, phụ thuộc mạng | Cold-start; graph có thể gián đoạn | Giáo dục; Expo Go demo |

# 20. Phản ánh và bài học

Baseline đa số đạt Accuracy 64.94% nhưng Precision, Recall, F1 đều 0 vì luôn dự đoán lớp 0. Đánh giá phải phản ánh loại lỗi quan trọng chứ không chỉ chỉ số tổng hợp dễ đạt.

Biểu diễn có ảnh hưởng đo được. Tiểu đường tốt hơn với 6 thay vì 8 thuộc tính khi missingness cao và mẫu nhỏ. Giá nhà cho kết luận ngược: 5 trường bổ sung cải thiện RMSE CV từ 1.6526 xuống 1.6078. Biểu diễn cần được kiểm chứng riêng cho từng bài toán.

CV và test trả lời câu hỏi khác nhau. Forest tiểu đường depth 6 được chọn theo F1 trung bình trên fold train dù F1 test thấp hơn forest không giới hạn quan sát trước đó. Đổi lựa chọn sau khi xem test sẽ biến test thành tài nguyên tuning.

Pipeline đã lưu bảo đảm medians, scales, categories và tham số lúc train chính là những gì FastAPI dùng. Web/mobile chỉ là client của hợp đồng đầu vào thô. Một hệ thống thông minh là sự phối hợp dữ liệu, biểu diễn, đánh giá, suy luận, giao diện và triển khai, không chỉ `model.fit()`.

# 21. Kết luận

Bài tập cung cấp hai hệ thống giáo dục hoàn chỉnh: Phân loại tiểu đường và Hồi quy giá nhà Việt Nam. Cả hai có dữ liệu, biểu diễn, EDA, xử lý thiếu, baseline, năm họ mô hình, thí nghiệm kiểm soát, lựa chọn cuối, Pipeline đã lưu, API, web và mobile. Tiểu đường có thêm Knowledge Graph Neo4j.

Mô hình tiểu đường depth 6, 6 thuộc tính đạt Accuracy 0.7468, Precision 0.6667, Recall 0.5556, F1 0.6061. Mô hình giá nhà depth 12, 11 thuộc tính đạt MAE 1.2529, RMSE 1.6018 tỷ VND, R² 0.4738, MAPE 27.36%. Kết quả chứng minh quy trình bài tập nhất quán, không chứng minh sẵn sàng lâm sàng, định giá chuyên nghiệp hay thương mại.

# 22. Khả năng tái lập

Notebook ghi Python 3.12.0, pandas 3.0.5, NumPy 2.5.2, matplotlib 3.11.1, scikit-learn 1.9.0, joblib 1.5.3. Split, KFold shuffle, Decision Tree, Random Forest dùng `random_state=42` khi hỗ trợ. Tiểu đường dùng stratified 80/20; giá nhà dùng 80/20 và CV 5 fold trên train. Tiền xử lý đã fit nằm trong từng fold và artifact joblib.

**Bảng 21. Công cụ tái lập.**

| Tầng | Công cụ/phiên bản |
|---|---|
| Notebook ML | Python 3.12.0; pandas 3.0.5; NumPy 2.5.2; matplotlib 3.11.1; scikit-learn 1.9.0; joblib 1.5.3 |
| Backend | FastAPI; Uvicorn; Pydantic; pandas; NumPy; scikit-learn; Neo4j driver |
| Web | Node.js 20+; React 19; Vite 7; TypeScript 5.9 |
| Mobile | Expo 54; React Native 0.81; React 19; TypeScript 5.9 |
| Báo cáo | Python; python-docx 1.2.0; text, heading, bảng DOCX native |

Lệnh từ thư mục gốc:

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Set-Location web
npm install
npm run dev

Set-Location ..\mobile
npm install
npx expo start
```

Kiểm tra bằng `python -m pytest` với `PYTHONPATH=backend`, web `npm run lint`, `npm run build`, mobile `npm run typecheck`. Tạo lại báo cáo bằng:

```text
.venv\Scripts\python.exe -m pip install -r docs\report\requirements-report.txt
.venv\Scripts\python.exe docs\report\build_report.py
```

# 23. Tài liệu tham khảo

1. Kaggle. *Bộ dữ liệu tiểu đường được notebook sử dụng*. Notebook ghi Kaggle là nguồn nhưng không lưu URL dataset card; báo cáo không suy diễn tác giả.
2. Kaggle. *Vietnam Housing Dataset 2024*. <https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024>.
3. Nhóm phát triển scikit-learn. *Tài liệu scikit-learn*. <https://scikit-learn.org/stable/>.
4. FastAPI. *Tài liệu FastAPI*. <https://fastapi.tiangolo.com/>.
5. Meta Open Source. *Tài liệu React*. <https://react.dev/>.
6. Vite. *Tài liệu Vite*. <https://vite.dev/>.
7. Expo. *Tài liệu Expo*. <https://docs.expo.dev/>.
8. Neo4j. *Tài liệu Neo4j*. <https://neo4j.com/docs/>.
9. Render. *Tài liệu Render*. <https://render.com/docs>.
10. Vercel. *Tài liệu Vercel*. <https://vercel.com/docs>.
11. Source repository. <https://github.com/Sagitoaz/intelligent-system-assignment-01>.

# Phụ lục A – Các endpoint API

**Bảng 22. Hợp đồng API chi tiết.**

| Phương thức/đường dẫn | Request | Phản hồi thành công | Lỗi chính |
|---|---|---|---|
| `GET /health` | Không | Trạng thái dịch vụ, mô hình, Neo4j | Mạng/dịch vụ |
| `GET /api/v1/models` | Không | Hai metadata | 500 server |
| `GET /api/v1/models/diabetes` | Không | Sáu trường, mô hình, metrics, disclaimer | 500 server |
| `GET /api/v1/models/house-price` | Không | 11 trường, categories, target, model | 500 server |
| `POST /api/v1/diabetes/predict` | JSON đúng 6 trường | Lớp, nhãn, xác suất, model | 413; 422; 500 |
| `POST /api/v1/house-price/predict` | JSON đúng 11 key, có trường nullable | Giá, đơn vị, định dạng, model | 413; 422; 500 |
| `GET /api/v1/diabetes/knowledge-graph` | Không | Mảng node và edge | 503 Neo4j |

Key tiểu đường phân biệt hoa thường: `Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, `DiabetesPedigreeFunction`, `Age`. Key nhà: `Province`, `Area`, `Frontage`, `Access Road`, `House direction`, `Balcony direction`, `Floors`, `Bedrooms`, `Bathrooms`, `Legal status`, `Furniture state`. Key thừa bị từ chối.

# Phụ lục B – Cấu trúc dự án

```text
data/                 CSV tiểu đường và nhà ở Việt Nam
notebooks/            Notebook đã thực thi
models/               sklearn Pipeline đã fit
figures/              Hình đánh giá từ notebook
backend/              FastAPI, metadata, service, schema, test
web/                  React/Vite/TypeScript
mobile/               Expo React Native/TypeScript
knowledge_graph/      Cypher Neo4j idempotent và tài liệu
docs/                 Kiến trúc, API và báo cáo kỹ thuật
```

Sản phẩm báo cáo gồm `TECHNICAL_REPORT.md`, `TECHNICAL_REPORT.docx`, `build_report.py`, requirements riêng và assets. Sinh báo cáo không sửa notebook, model hay mã ứng dụng.

# Phụ lục C – Các ca đầu vào demo

**Bảng 23. Đầu vào demo tiểu đường từ notebook.**

| Trường | Ca 1 | Ca 2 | Ca 3 |
|---|---:|---:|---:|
| Pregnancies | 1 | 4 | 8 |
| Glucose | 85 | 125 | 180 |
| BloodPressure | 66 | 72 | 80 |
| BMI | 24.0 | 32.0 | 38.0 |
| DiabetesPedigreeFunction | 0.20 | 0.50 | 1.20 |
| Age | 23 | 35 | 55 |
| Lớp dự đoán | 0 | 0 | 1 |
| Nhãn | Non-diabetic | Non-diabetic | Diabetic |

**Bảng 24. Đầu vào demo giá nhà từ notebook.**

| Trường | Ca 1 | Ca 2 | Ca 3 |
|---|---|---|---|
| Province | Hồ Chí Minh | Bình Dương | Hưng Yên |
| Area | 45.0 | 80.0 | 90.0 |
| Frontage | 4.0 | 5.0 | 6.0 |
| Access Road | 4.0 | 8.0 | 13.0 |
| House direction | Đông - Nam | Nam | Đông - Bắc |
| Balcony direction | Đông - Nam | Nam | Đông - Bắc |
| Floors | 3.0 | 2.0 | 5.0 |
| Bedrooms | 3.0 | 3.0 | 5.0 |
| Bathrooms | 3.0 | 2.0 | 5.0 |
| Legal status | Have certificate | Have certificate | Sale contract |
| Furniture state | Full | Basic | Full |
| Giá dự đoán | 5.1267 | 3.0945 | 8.6189 tỷ VND |
