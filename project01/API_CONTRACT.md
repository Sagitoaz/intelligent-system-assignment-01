# Hợp đồng API của "Phòng khám AI" (project01)

Máy chủ: `python project01/server.py` (dùng `.venv`), cổng **8200**, tệp tĩnh ở `project01/static/`. Mọi API trả JSON UTF-8.
Nhãn hiển thị bằng tiếng Việt. Đây là demo giáo dục, KHÔNG phải chẩn đoán y khoa.

## Nhân vật (agent)

| id | Tên hiển thị | Vai trò | Màu |
|---|---|---|---|
| `doctor` | Bác sĩ trưởng | Điều phối, kết luận | `#2589e0` |
| `predictor` | Bác sĩ dữ liệu | Deep Learning dự đoán nguy cơ (BRFSS 253.680 hồ sơ) | `#7060d8` |
| `knowledge` | Thủ thư tri thức | KB (đồ thị) + RAG trích dẫn tài liệu NIDDK | `#0d9f8f` |
| `coach` | Huấn luyện viên | Reinforcement learning, gợi ý ăn uống và vận động | `#f0603a` |
| `guardian` | Hộ lý an toàn | Luật (rule) cảnh báo khẩn | `#e8a317` |
| `reporter` | Thư ký báo cáo | Sinh hình và bảng cho báo cáo | `#e8508a` |

## GET /api/meta
```json
{"agents":[{"id":"predictor","name":"Bác sĩ dữ liệu","role":"Deep Learning","color":"#7060d8","blurb":"..."}],
 "personas":[{"id":"minh","name":"Anh Minh, 52 tuổi","story":"...","profile":{...}}],
 "fields":[{"key":"age_years","label":"Tuổi","type":"number","min":18,"max":95,"default":45,"unit":"tuổi","help":"..."},
           {"key":"high_bp","label":"Huyết áp cao","type":"bool","default":false}, 
           {"key":"gen_health","label":"Tự đánh giá sức khỏe","type":"select","options":[[1,"Rất tốt"],[2,"Tốt"],[3,"Khá"],[4,"Kém"],[5,"Rất kém"]],"default":3}],
 "model":{"name":"MLP 3 lớp","train_rows":202944,"test_rows":50736,"accuracy":0.0,"macro_f1":0.0,"auc_ovr":0.0},
 "disclaimer":"..."}
```
`profile` dùng các khóa: `age_years, sex ("female"|"male"), height_cm, weight_kg, high_bp, high_chol, chol_check, smoker, stroke, heart_disease, phys_activity, fruits_daily, veggies_daily, heavy_alcohol, any_healthcare, no_doc_cost, diff_walking, gen_health(1-5), mental_bad_days(0-30), physical_bad_days(0-30), education(1-6), income(1-8), fasting_glucose (mg/dL, tùy chọn), symptoms (mảng trong: thirst, frequent_urination, fatigue, blurred_vision, slow_healing)`.

## POST /api/consult  `{"profile":{...}}`
Trả kịch bản hội chẩn đã tính xong để giao diện diễn hoạt theo thứ tự:
```json
{"events":[
  {"i":0,"agent":"doctor","type":"speak","text":"Chào anh Minh...","focus":"doctor","ms":2200},
  {"i":1,"agent":"doctor","type":"handoff","text":"Mời Bác sĩ dữ liệu...","to":"predictor","ms":1200},
  {"i":2,"agent":"predictor","type":"tool","text":"Chạy mô hình...","payload":{"risk":{"none":0.61,"pre":0.03,"diabetes":0.36}},"ms":2600},
  {"i":3,"agent":"knowledge","type":"cite","text":"...","payload":{"kg_nodes":["high_bp","obesity"],"passages":[{"id":"...","quote":"...","url":"..."}]},"ms":3000},
  {"i":4,"agent":"coach","type":"speak","text":"...","payload":{"actions":["walk","light_meal"]},"ms":2400},
  {"i":5,"agent":"guardian","type":"alert","level":"warn","text":"...","ms":2400},
  {"i":6,"agent":"reporter","type":"tool","text":"Đang vẽ biểu đồ...","ms":1800},
  {"i":7,"agent":"doctor","type":"final","text":"Tóm tắt...","ms":3000}],
 "summary":{"risk":{"none":0.61,"pre":0.03,"diabetes":0.36},"level":"cao","level_color":"#e8503a",
   "headline":"...","advice":["..."],"red_flags":[],"citations":[{"id":"...","quote":"...","url":"..."}],"kg_nodes":["..."]},
 "report":{"figures":[{"id":"risk_bars","title":"...","caption":"...","png_base64":"..."}],
           "tables":[{"id":"profile","title":"...","columns":["..."],"rows":[["..."]]}]}}
```
`type` ∈ `speak | handoff | tool | cite | alert | final`; `level` ∈ `info | warn | danger`.

## GET /api/kg  → `{"nodes":[{"id":"high_bp","label":"High blood pressure","vi":"Huyết áp cao","type":"risk_factor","desc":"..."}],"edges":[{"source":"high_bp","target":"type2","rel":"raises_risk"}]}`
`type` ∈ `condition | symptom | risk_factor | test | treatment | lifestyle | complication`.
## POST /api/kg/search `{"q":"triệu chứng tiểu đường"}` → `{"nodes":["thirst",...],"passages":[{"id","quote","url","score"}],"answer":"..."}`

## Sống 24 giờ (bệnh nhân ảo; mô phỏng đồ chơi, không phải y khoa thật)
`POST /api/life/start {"profile":{...},"seed":7}` →
`{"patient":{"name":"...","insulin_sensitivity":0.8,"baseline_glucose":112},"state":{"hour":6,"glucose":112.0,"carbs_on_board":0.0,"activity_on_board":0.0,"log":[]},
  "actions":[{"id":"rest","label":"Nghỉ ngơi","emoji":"🛋️","desc":"..."},{"id":"light_meal",...},{"id":"normal_meal",...},{"id":"heavy_meal",...},{"id":"walk",...},{"id":"workout",...},{"id":"skip",...}],
  "advice":{"coach":{"action":"light_meal","why":"..."},"rule":{"action":"rest","why":"..."}}}`
`POST /api/life/step {"state":{...},"action":"walk","seed":7}` →
`{"state":{...},"glucose_before":120.0,"glucose_after":105.0,"reward":1.2,"zone":"ok","events":["..."],"advice":{...},"done":false,"score":{"time_in_range":0.75,"hypo_hours":0,"hyper_hours":2}}` (`zone` ∈ `low | ok | high | danger`; một ngày mô phỏng từ giờ 6 đến giờ 30 = 24 bước).
`POST /api/life/compare {"profile":{...},"days":30,"seed":7}` →
`{"coaches":[{"id":"rl","name":"Huấn luyện viên RL","time_in_range":0.82,"mean_glucose":131.0,"hypo_hours":0.1,"hyper_hours":2.4,"curve":[...24 giá trị...]},{"id":"rule",...},{"id":"none",...}],"table":{"columns":[...],"rows":[[...]]}}`

## Cổng và tệp
`GET /` → `static/index.html` (ứng dụng 3D). Tệp tĩnh: `/app.js`, `/scene.js`, `/style.css`, `/vendor/three.min.js`, `/vendor/OrbitControls.js`.
