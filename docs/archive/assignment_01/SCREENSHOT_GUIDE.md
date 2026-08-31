# Checklist ảnh minh họa web và mobile

Mục tiêu là bổ sung bằng chứng giao diện chạy thật vào Mục 18 của báo cáo. Không chụp credential, terminal chứa biến môi trường, QR Expo hoặc thông tin cá nhân. Dùng đúng bản mới nhất trên `main`.

## Trạng thái ảnh đã nhận

Đã nhận và chèn sáu ảnh vào báo cáo dưới số Hình 15–20. Hai file `mobile_prediction.png` và `mobile_knowledge_graph.png` có nội dung là web responsive, không phải ứng dụng Expo React Native: giao diện và canvas graph trùng với web, trong khi mobile dùng node explorer native. Vì vậy caption trong báo cáo mô tả trung thực chúng là ảnh web ở viewport điện thoại. Báo cáo không tuyên bố đây là ảnh Expo.

## Ảnh bắt buộc đề xuất

1. `web_diabetes_prediction.png`
   - Mở web production, trang Diabetes, trên desktop rộng khoảng 1366–1440 px.
   - Nhập ca 1: Pregnancies 1; Glucose 85; BloodPressure 66; BMI 24.0; DiabetesPedigreeFunction 0.20; Age 23.
   - Chụp cả form và kết quả Non-diabetic; không cắt disclaimer.

2. `web_house_price_prediction.png`
   - Trang House Price trên desktop.
   - Nhập ca 1: Hồ Chí Minh; Area 45; Frontage 4; Access Road 4; hướng nhà/ban công Đông - Nam; Floors/Bedrooms/Bathrooms đều 3; Have certificate; Full.
   - Chụp form và giá dự đoán khoảng 5.13 tỷ VND.

3. `web_knowledge_graph.png`
   - Trang Knowledge Graph, bấm Fit graph rồi chọn node Random Forest Classifier.
   - Chụp được toàn graph, dòng 17 nodes · 17 relationships và panel Node details.

4. `web_responsive_mobile.png`
   - Mở DevTools ở kích thước 390 × 844 px.
   - Mở menu hamburger để thể hiện navigation responsive; ưu tiên chụp trang Home hoặc Diabetes.

5. `mobile_prediction.jpg`
   - Mở dự án bằng Expo Go trên điện thoại dọc.
   - Dùng ca tiểu đường 1 ở trên và chụp màn hình kết quả, gồm disclaimer nếu vừa khung.

6. `mobile_knowledge_graph.jpg`
   - Mở tab Graph, chọn Random Forest Classifier.
   - Chụp phần tổng 17 node/17 relationship, thuộc tính node và ít nhất một connection.

## Yêu cầu chất lượng

- Ảnh rõ, không qua ứng dụng chat làm giảm chất lượng; gửi file gốc PNG/JPG.
- Không ghép nhiều màn hình vào một ảnh và không dùng ảnh mockup.
- Desktop nên dùng tỷ lệ 16:9; mobile giữ ảnh dọc nguyên bản.
- Sau khi nhận ảnh, đặt chúng trong `docs/report/assets/` với đúng tên trên để chèn và đánh số Hình 15 trở đi.
