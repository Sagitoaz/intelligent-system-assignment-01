# HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG

## INTELLIGENT SYSTEM DEVELOPMENT

## BÀI TẬP 02

### FROM DATA REPRESENTATION TO A DEPLOYABLE INTELLIGENT SYSTEM

**Sinh viên:** Nguyễn Thành Trung

**Mã sinh viên:** B23DCCN861

**Lớp:** D23CTPM01-B

**Giảng viên:** Dinh Que Tran, Ph.D., Assoc. Prof.

**Năm:** 2026

---

## Tóm tắt

Báo cáo trình bày ba hệ thống học máy end-to-end: phân loại Diabetes, hồi quy Vietnam House Price và phân loại E-commerce Customer Preference. Trọng tâm của Assignment 02 là mối liên hệ giữa dữ liệu thô, biểu diễn số, thí nghiệm mô hình, artifact đã lưu và ứng dụng có thể triển khai. Ba bài toán khác nhau về modality, target và metric nhưng dùng chung nguyên tắc: preprocessing được fit trên training data, đóng gói cùng estimator trong `Pipeline`, lưu bằng joblib và tái sử dụng nguyên trạng trong FastAPI, Web React và Mobile Expo.

Kết quả test cuối là: Diabetes Accuracy 0,7468 và F1 0,6061; House MAE 1,2529 tỷ VND, RMSE 1,6018 tỷ VND và R² 0,4738; Ecommerce Accuracy 0,9566, F1 0,9746 và ROC-AUC 0,9842. Các con số được lấy từ executed notebook outputs và persisted artifacts, không chạy lại training trong giai đoạn lập báo cáo. Web được triển khai trên Vercel, backend trên Render; Mobile được minh họa bằng Expo React Native/Expo Go. Neo4j chỉ là phần mở rộng độc lập của Diabetes và không phải yêu cầu cốt lõi.

# 1. Giới thiệu

Một hệ thống thông minh triển khai được không kết thúc ở việc chọn model. Hệ thống phải định nghĩa đúng một observation, tạo representation có ý nghĩa, kiểm soát leakage, đánh giá trên dữ liệu chưa nhìn thấy, lưu toàn bộ phép biến đổi đã học và giữ hợp đồng input nhất quán giữa notebook với application. Repository giải quyết chuỗi này cho ba bài toán có tính chất khác nhau.

Mục tiêu của báo cáo là trả lời bốn câu hỏi: dữ liệu thô là gì; dữ liệu trở thành numerical representation như thế nào; bằng chứng nào dẫn tới lựa chọn model; và persisted `Pipeline` được phục vụ cho người dùng ra sao. Báo cáo không xem dự đoán Diabetes là chẩn đoán, không gọi R² của House là Accuracy, không gọi TF-IDF là embedding và không diễn giải probability Ecommerce như độ chắc chắn về hành vi mua.

# 2. Tổng quan ba hệ thống

| Tiêu chí | Diabetes | House Price | E-commerce Preference |
|---|---|---|---|
| Problem type | Binary classification | Regression | Binary text classification |
| Observation | Một hồ sơ bệnh nhân | Một tin đăng bất động sản | Một review của user cho product |
| Target | `Outcome` 0/1 | `Price`, tỷ VND | `Score`-derived Negative/Positive |
| Dataset | 768 quan sát | 30.229 tin đăng | 568.454 review thô |
| Raw feature | 6 numerical | 11 numerical/categorical | 2 text + 2 helpfulness counts |
| Numerical representation | Median + scaling | Median/mode + scaling + one-hot | TF-IDF + 5 engineered values |
| Final dimension | `B × 6` | `B × 83` | `B × 12.005` |
| Final model | Random Forest Classifier | Random Forest Regressor | Logistic Regression |
| Main metric | Recall/F1, kèm Accuracy | MAE/RMSE/R²/MAPE | F1 và ROC-AUC |
| Vấn đề dữ liệu | Hidden zero, imbalance | Missing, Address, skew/outlier | Neutral target, duplicate, imbalance |
| Triển khai | FastAPI/Web/Mobile | FastAPI/Web/Mobile | FastAPI/Web/Mobile |
| Hạn chế chính | Dataset nhỏ, không lâm sàng | Listing không phải transaction | Rating proxy không phải customer intent |

Bảng cho thấy “feature count” ở form và “model dimension” không nhất thiết bằng nhau. Diabetes giữ sáu numerical dimensions. House có 11 trường thô nhưng categorical one-hot làm tăng lên 83. Ecommerce chỉ có bốn trường API nhưng vocabulary 12.000 term và năm feature tabular tạo 12.005 chiều sparse.

# 3. Nguồn dữ liệu và xác định bài toán

## 3.1 Diabetes

Dataset local có 768 hàng và target `Outcome`: 500 quan sát lớp 0, 268 quan sát lớp 1. Tỷ lệ này tạo class imbalance vừa phải, vì vậy baseline Accuracy có thể cao dù không phát hiện lớp Diabetic. Sáu feature cuối là `Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, `DiabetesPedigreeFunction`, `Age`. `SkinThickness` và `Insulin` không thuộc representation cuối vì tỷ lệ hidden-zero lớn và thí nghiệm 6-vs-8 feature cho thấy bộ sáu feature có CV F1 trung bình tốt và ổn định hơn.

## 3.2 Vietnam House Price

Dataset Kaggle Vietnam Housing 2024 có 30.229 hàng. Target `Price` là giá niêm yết theo tỷ VND, không phải giá giao dịch. Dữ liệu trộn numerical và categorical; nhiều trường thiếu, đặc biệt Balcony direction, House direction và Furniture state. `Address` có cardinality cao nên được biến đổi xác định thành `Province`, tạo 60 nhãn chuẩn hóa và không còn Province missing.

## 3.3 E-commerce Customer Preference

Amazon Fine Food Reviews có 568.454 dòng thô. Quy tắc target là `Score` 4–5 → Positive, 1–2 → Negative, còn Score 3 bị loại. Sau khi loại 42.640 neutral rows, 2 helpfulness records không hợp lệ và 829 duplicate reviews, còn 524.983 dòng usable. Stratified modeling sample có 120.000 dòng, gồm 18.686 Negative và 101.314 Positive. `Score` chỉ tạo target rồi bị loại khỏi `X`; `Id`, `ProductId`, `UserId`, `ProfileName` và `Time` không phải predictive input. Đây là ranh giới quan trọng chống leakage và ghi nhớ identifier.

# 4. Biểu diễn dữ liệu

| Ứng dụng | Dữ liệu thô | Biểu diễn số | Model input |
|---|---|---|---|
| Diabetes | CSV numerical patient record | Hidden-zero handling, median-imputed, standardized vector | `X ∈ R^(B×6)` |
| House | CSV mixed numerical/categorical listing | Numerical scaling + categorical one-hot | `X ∈ R^(B×83)` |
| Ecommerce | Review text + helpfulness counts | TF-IDF 12.000 + 5 numerical values | `X ∈ R^(B×12.005)` |

Trong bảng, `B` là số observations được suy luận cùng lúc. Với Ecommerce, `V=12.000` là vocabulary size, `d_tabular=5`, do đó `d_final=V+d_tabular=12.005`. Với House, `d_raw=11` nhưng `d_encoded=83` vì mỗi category trở thành một hoặc nhiều indicator columns. Người dùng vẫn chỉ nhập raw fields; application không gửi encoded vector.

## 4.1 Bằng chứng code: hidden-zero của Diabetes

```python
invalid_zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
df_clean[invalid_zero_columns] = (
    df_clean[invalid_zero_columns].replace(0, np.nan)
)
```

**Giải thích:** Code chỉ đổi zero thành missing cho các phép đo sinh lý mà zero không hợp lý. `Pregnancies=0` được giữ vì là giá trị có nghĩa. `SimpleImputer(strategy="median")` nằm trong `Pipeline`, nên median được học từ training folds thay vì toàn dataset.

**Phân tích kết quả:** Việc xử lý hidden zero ngăn model coi “không đo được” là mức sinh lý cực thấp. Nó cũng giải thích tại sao raw representation và clean representation khác nhau dù đều có sáu cột.

## 4.2 Bằng chứng code: Address thành Province

```python
def extract_province(address):
    if pd.isna(address):
        return np.nan
    normalized_address = unicodedata.normalize("NFC", str(address)).strip()
    components = [c.strip(" .") for c in normalized_address.split(",") if c.strip(" .")]
    province = components[-1] if components else np.nan
    return PROVINCE_ALIASES.get(province, province)

df_clean["Province"] = df_clean["Address"].apply(extract_province)
```

**Giải thích:** Thành phần địa lý cuối được chuẩn hóa Unicode và ánh xạ alias. Input là `Address`; output là một category `Province`. Phép biến đổi không đọc `Price`, nên không tạo target leakage.

**Phân tích kết quả:** Output notebook xác nhận 0 Province missing và 60 labels. Cách này giảm cardinality so với one-hot trực tiếp 10.265 địa chỉ, đồng thời vẫn giữ tín hiệu thị trường.

## 4.3 Bằng chứng code: TF-IDF và FeatureUnion

```python
text = Pipeline([
    ("combine_text", ReviewTextTransformer()),
    ("tfidf", TfidfVectorizer(
        max_features=12000, ngram_range=(1, 2), min_df=3,
        max_df=0.98, sublinear_tf=True, strip_accents="unicode",
        dtype=np.float32,
    )),
])
tabular = Pipeline([
    ("engineer", ReviewTabularTransformer()),
    ("scale", MaxAbsScaler()),
])
features = FeatureUnion([("text", text), ("tabular", tabular)])
```

**Giải thích:** `TfidfVectorizer` học vocabulary chỉ từ training data và tạo sparse unigram/bigram weights. Nhánh tabular tạo numerator, denominator, helpfulness ratio, review length, summary length; `MaxAbsScaler` giữ khả năng xử lý sparse. `FeatureUnion` ghép hai nhánh theo cột.

**Phân tích kết quả:** Token ID chỉ là vị trí trong vocabulary; TF-IDF vector chứa weighted term values; embedding là dense learned vector. Notebook không đồng nhất ba khái niệm. TF-IDF được chọn thay vì tensor Transformer `B×T×d` vì phù hợp CPU, 120.000 rows và deployment gọn; đổi lại nó mất phần lớn context ngoài bigram.

# 5. Tiền xử lý và EDA

Diabetes EDA xác nhận imbalance 500/268 và hidden zeros. Sau làm sạch, median imputation giảm ảnh hưởng outlier hơn mean; scaling giúp các model distance/linear được so sánh công bằng. Random Forest không cần scaling về mặt toán học, nhưng việc giữ preprocessing chung trong Pipeline giúp hợp đồng inference nhất quán.

House có Area lệch phải mạnh, missing values ở cả numerical/categorical và geographic imbalance. Outlier của Area không tự động là lỗi; các giá trị dương đến 595 m² có thể là tài sản lớn hợp lệ nên được giữ. Numerical branch dùng median imputation + scaling; categorical branch dùng most-frequent imputation + one-hot với unknown category ignored.

Ecommerce EDA cho thấy Positive chiếm ưu thế, review length có đuôi phải và helpfulness tập trung gần zero. Review dài được giữ; clipping chỉ dùng khi vẽ histogram. Exact duplicates được loại trước split. Các quyết định này dẫn đến stratification và đánh giá đa metric thay vì dựa vào Accuracy.

![Hình 1. Class distribution của modeling sample Ecommerce.](../../figures/ecommerce/target_distribution.png)

*Hình 1. Positive chiếm tỷ lệ lớn hơn rõ rệt; F1, ROC-AUC và confusion matrix cần được đọc cùng Accuracy.*

# 6. Phát triển và so sánh mô hình

## 6.1 Diabetes

Baseline dự đoán toàn bộ là Non-diabetic, đạt Accuracy 0,6494 nhưng Recall/F1 của lớp 1 bằng 0. Năm model gồm Logistic Regression, KNN, Decision Tree, Random Forest và SVM. Cross-validation trên training data giúp tránh dùng test để chọn cấu hình. Thí nghiệm `max_depth` cho CV F1 trung bình: depth 2 = 0,5513; 4 = 0,6224; 6 = 0,6462; sâu hơn không tạo lợi ích ổn định tương ứng. Thí nghiệm representation cho sáu feature CV F1 0,6462 so với tám feature 0,6243.

Random Forest không giới hạn đạt hold-out F1 0,6408, cao hơn final depth-6 F1 0,6061. Đây không phải lỗi: depth 6 được chọn trước bằng training-only CV nhằm cân bằng hiệu suất và độ phức tạp; test set không được dùng ngược để lựa chọn lại sau khi nhìn kết quả.

![Hình 2. Thí nghiệm max_depth của Random Forest Diabetes.](../../figures/diabetes/experiment_2_max_depth.png)

*Hình 2. CV training-only đạt cực đại ở depth 6, là bằng chứng cho cấu hình cuối.*

## 6.2 House Price

Baseline mean có MAE 1,8438 và RMSE 2,2082 tỷ VND. Các họ model A1 gồm Linear Regression, KNN, Decision Tree, Random Forest, SVR; Assignment 02 bổ sung Ridge và Gradient Boosting trên training folds nhưng không dùng test để thay artifact. Random Forest `max_depth=12` là điểm thấp nhất của CV RMSE trong thí nghiệm độ sâu. Mở rộng từ 6 lên 11 raw features giảm CV RMSE từ 1,6526 xuống 1,6078, vì Frontage, Access Road, directions và Furniture bổ sung tín hiệu.

![Hình 3. Thí nghiệm biểu diễn 6 và 11 feature của House Price.](../../figures/house_price/experiment_3_feature_representation.png)

*Hình 3. Biểu diễn 11 raw fields cải thiện CV RMSE và được chọn cho persisted Pipeline.*

## 6.3 Ecommerce

Controlled representation experiment giữ Logistic Regression và mọi điều kiện khác cố định:

| Representation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tabular only | 0,8442 | 0,8445 | 0,9994 | 0,9155 | 0,6410 |
| Text only | 0,9570 | 0,9627 | 0,9873 | 0,9749 | 0,9836 |
| Text + tabular | 0,9574 | 0,9631 | 0,9874 | 0,9751 | 0,9840 |

Text tạo gần như toàn bộ mức cải thiện; năm tabular features chỉ thêm một gain nhỏ. Chênh lệch Text-only và Combined không được thổi phồng.

![Hình 4. So sánh representation Ecommerce.](../../figures/ecommerce/representation_comparison.png)

*Hình 4. TF-IDF tạo cải thiện lớn so với tabular-only; Combined nhỉnh hơn Text-only ở mức nhỏ.*

Sáu model dùng cùng combined sparse matrix:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,9574 | 0,9631 | 0,9874 | 0,9751 | **0,9840** |
| Decision Tree | 0,8989 | 0,9230 | 0,9604 | 0,9413 | 0,8568 |
| Random Forest | 0,8588 | 0,8567 | 1,0000 | 0,9228 | 0,9512 |
| Linear SVC | **0,9598** | 0,9703 | 0,9825 | **0,9764** | 0,9832 |
| SGD Classifier | 0,9531 | 0,9565 | 0,9894 | 0,9727 | 0,9828 |
| Complement NB | 0,9031 | 0,9847 | 0,8992 | 0,9400 | 0,9707 |

Linear SVC có F1 cao nhất; Logistic Regression không tốt nhất ở mọi metric. Logistic Regression được chọn vì F1 chỉ thấp hơn 0,0013, ROC-AUC cao nhất trong so sánh, có native probability và không cần calibration layer, nhờ đó đơn giản hơn cho API/UI.

![Hình 5. So sánh sáu model Ecommerce trên validation.](../../figures/ecommerce/model_comparison.png)

*Hình 5. Linear SVC dẫn đầu F1, còn Logistic Regression thỏa selection rule có probability.*

# 7. Kết quả và phân tích

| Hệ thống | Final test metrics | Diễn giải |
|---|---|---|
| Diabetes | Accuracy 0,7468; Precision 0,6667; Recall 0,5556; F1 0,6061 | 30 TP, 24 FN; không phải chẩn đoán |
| House | MAE 1,2529; MSE 2,5659; RMSE 1,6018; R² 0,4738; MAPE 27,36% | Error theo tỷ VND; R² không phải Accuracy |
| Ecommerce | Accuracy 0,9566; Precision 0,9639; Recall 0,9855; F1 0,9746; ROC-AUC 0,9842 | Target là rating-derived proxy |

## 7.1 Diabetes

Confusion matrix `[[85, 15], [24, 30]]` cho 85 true negatives, 15 false positives, 24 false negatives và 30 true positives. False negative quan trọng trong bối cảnh sàng lọc, nhưng dataset nhỏ và output mang tính giáo dục; hệ thống không thể thay thế thăm khám.

![Hình 6. Confusion matrix cuối của Diabetes.](../../figures/diabetes/final_confusion_matrix.png)

*Hình 6. Recall lớp Diabetic 0,5556 phản ánh 24 trường hợp lớp 1 bị bỏ sót.*

## 7.2 House Price

MAE 1,2529 tỷ VND biểu thị sai lệch tuyệt đối trung bình theo thang giá gốc; RMSE 1,6018 phạt lỗi lớn mạnh hơn. R² 0,4738 nghĩa là mô hình giải thích khoảng 47,38% phương sai giá trên test set, **không phải 47,38% Accuracy**. MAPE 27,36% cung cấp góc nhìn tương đối nhưng vẫn bị ảnh hưởng bởi mức giá.

![Hình 7. Actual vs predicted của House Price.](../../figures/house_price/final_actual_vs_predicted.png)

*Hình 7. Điểm phân tán quanh đường lý tưởng cho thấy tín hiệu dự đoán có thật nhưng sai số còn đáng kể.*

Residual mean -0,0255 tỷ VND và tương quan residual–prediction 0,0189 gần zero, cho thấy signed bias tổng thể nhỏ. Tuy vậy, dispersion vẫn lớn ở từng listing; feature importance dựa trên impurity chỉ mô tả model, không chứng minh quan hệ nhân quả.

## 7.3 Ecommerce

Final confusion matrix `[[2242, 561], [221, 14976]]` cho Recall Positive rất cao 0,9855. Tuy nhiên Positive chiếm đa số nên cần xem thêm negative recall 0,7999, ROC-AUC và error examples. False positives/negatives thường liên quan câu ngắn, mixed sentiment hoặc rating/text không nhất quán.

![Hình 8. ROC curve cuối của Ecommerce.](../../figures/ecommerce/roc_curve.png)

*Hình 8. ROC-AUC 0,9842 cho khả năng xếp hạng hai lớp tốt trên test sample; không biến proxy target thành customer intent thật.*

# 8. Kiến trúc hệ thống và triển khai

![Hình 9. Kiến trúc production của Assignment 02.](assets/system_architecture.png)

*Hình 9. Web Vercel và Mobile Expo gửi raw input qua HTTPS tới FastAPI Render; backend validation rồi gọi một trong ba persisted Pipelines.*

Production Web: `https://intelligent-system-assignment-01.vercel.app`. Production Backend: `https://intelligent-system-assignment-01.onrender.com`. FastAPI tải ba joblib artifacts một lần trong lifespan. Metadata xác định raw fields; Pydantic từ chối unknown/invalid fields; service tạo một-row `DataFrame` đúng thứ tự `feature_names_in_` rồi gọi `predict`/`predict_proba`. Không có `fit` tại inference.

Web React/Vite và Mobile Expo dùng cùng REST contract. Vercel route rewrite cho phép direct refresh. CORS cho phép production Vercel origin một cách explicit. Ecommerce artifact phụ thuộc các custom classes trong `shared_ml/ecommerce_transformers.py`, được copy/import trên Render; production health đã xác nhận cả ba model loaded.

Neo4j Diabetes Knowledge Graph được giữ như extension từ Assignment 01. Nó độc lập với prediction pipeline và không phải yêu cầu cốt lõi Assignment 02. Tại thời điểm kiểm tra báo cáo, production Neo4j unavailable nên báo cáo không khẳng định graph đang live và không dùng screenshot Knowledge Graph.

# 9. Minh chứng Web và Mobile

## 9.1 Web production

![Hình 10. Bốn màn hình Web production: Home, Diabetes, House Price và Ecommerce.](assets/web_production.png)

*Hình 10. Composite dùng nguyên bốn screenshot thật `Home.png`, `Diabetes.png`, `HousePricing.png`, `Ecommerce.png`. Các form gửi raw fields tới Render; kết quả lần lượt thể hiện ba hệ thống và không expose encoded vector.*

## 9.2 Mobile Expo

![Hình 11. Ba màn hình kết quả Mobile: Diabetes, House Price và Ecommerce.](assets/mobile_results.png)

*Hình 11. Composite dùng ba screenshot thật `diabetes2.png`, `house4.png`, `e2.png`. Mỗi ảnh giữ nguyên portrait ratio, hiển thị prediction result và dùng cùng production-compatible API contract.*

# 10. So sánh ba hệ thống

Diabetes là representation nhỏ nhất và dễ deploy nhất, nhưng dataset chỉ có 768 observations và domain y tế đòi hỏi diễn giải thận trọng. House có modality tabular hỗn hợp; one-hot làm tăng số chiều nhưng vẫn dense vừa phải. Ecommerce có dataset và dimension lớn nhất; sparse matrix là điều kiện kỹ thuật để controlled comparison khả thi.

Classification Diabetes cần quan tâm FN và Recall; regression House đo magnitude lỗi bằng MAE/RMSE và explained variance bằng R²; classification Ecommerce cần F1/ROC-AUC vì imbalance. Không thể dùng một metric chung để xếp hạng ba hệ thống. Điểm chung là `Pipeline` giữ phép biến đổi đã học, kiểm soát leakage và tạo contract ổn định giữa notebook và application.

# 11. Hạn chế và thảo luận

Diabetes dựa trên observational dataset nhỏ, hidden zeros và sáu measurements; feature importance không phải causal evidence. Output không phải medical diagnosis. House dùng listing price thay vì transaction price, thiếu nhiều thuộc tính và có geographic imbalance; prediction không phải professional valuation. Ecommerce target là proxy suy ra từ rating, Positive imbalance mạnh và có thể còn near-duplicate/user/product dependence; probability không phải certainty về intention.

Đánh giá hold-out phản ánh đúng split hiện tại nhưng không thay cho external validation. Các model có thể suy giảm khi distribution thay đổi, xuất hiện category mới hoặc văn phong review khác. Monitoring, model cards, data versioning và privacy review là hướng mở rộng hợp lý nhưng nằm ngoài phạm vi Assignment 02.

# 12. Kết luận

Ba hệ thống chứng minh một workflow thống nhất từ raw data đến deployment trong khi vẫn tôn trọng đặc thù từng bài toán. Diabetes tạo vector `B×6`; House biến 11 raw fields thành `B×83`; Ecommerce ghép TF-IDF và numerical features thành sparse `B×12.005`. Controlled experiments giải thích lựa chọn representation và model, còn persisted Pipelines bảo đảm inference không tái triển khai preprocessing bằng tay.

Assignment 02 đạt mục tiêu kỹ thuật chính: notebook có bằng chứng executed, model được lưu nguyên vẹn, backend phục vụ ba contract, Web/Mobile dùng raw inputs, và production deployment hoạt động cho prediction. Các limitation và disclaimer được giữ rõ để kết quả không bị diễn giải vượt quá bằng chứng.

# Tài liệu tham khảo

1. Đề bài Assignment 02, *From Data Representation to a Deployable Intelligent System*, tài liệu môn học Intelligent System Development, 2026.
2. Kaggle, *Pima Indians Diabetes Database*. URL được README ghi nhận; provenance card gốc chưa được lưu cùng notebook Assignment 01.
3. Kaggle, *Vietnam Housing Dataset 2024*: https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024
4. Kaggle, *Amazon Product Reviews / Amazon Fine Food Reviews*: https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews
5. scikit-learn documentation: https://scikit-learn.org/stable/
6. FastAPI documentation: https://fastapi.tiangolo.com/
7. React documentation: https://react.dev/
8. Expo documentation: https://docs.expo.dev/
9. Vercel documentation: https://vercel.com/docs
10. Render documentation: https://render.com/docs

# Phụ lục A. Các demo case và persisted inference

Diabetes demo: `Pregnancies=1`, `Glucose=85`, `BloodPressure=66`, `BMI=24`, `DiabetesPedigreeFunction=0.2`, `Age=23` → prediction 0, Non-diabetic, probability lớp 1 khoảng 0,0105.

House demo: Hồ Chí Minh, Area 45, Frontage 4, Access Road 4, hai hướng Đông–Nam, Floors/Bedrooms/Bathrooms đều 3, Legal status `Have certificate`, Furniture state `Full` → khoảng 5,1267 tỷ VND.

Ecommerce demo: Summary `Excellent`, Text `Fresh, tasty and exactly as described. I would buy it again.`, helpfulness 1/1 → Positive, positive-class probability khoảng 0,99982. Fresh-load predictions `[1,0,0]` bằng với predictions trước khi lưu.
