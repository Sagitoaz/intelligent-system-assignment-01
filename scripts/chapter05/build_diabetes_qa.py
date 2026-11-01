"""Tạo bộ câu hỏi đánh giá truy hồi cho kho tiểu đường (data/chapter04/diabetes_qa_demo.json).

Cách tạo (ghi rõ để người đọc báo cáo kiểm chứng): mỗi câu hỏi do người viết tự đặt từ nội dung MỘT đoạn của kho; câu "needle" là cụm
văn bản nguyên văn chỉ có ở đoạn đó. Nhãn đúng (gold_ids) là mọi đoạn chứa needle, nên nhãn được neo vào văn bản nguồn chứ không
phải do mô hình truy hồi chọn. Không có câu hỏi, đáp án hay nhãn nào được đưa vào chỉ mục truy hồi. Bộ này không được người
ngoài rà soát, chỉ là bộ minh họa để so sánh các phương pháp truy hồi.
Chạy: .venv\\Scripts\\python.exe scripts/chapter05/build_diabetes_qa.py
"""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/chapter04'

# (câu hỏi tiếng Anh, needle nguyên văn, đáp án chấp nhận được)
SPECS = [
    ('What organ makes insulin?', 'Insulin is a hormone made by the pancreas', ['pancreas', 'the pancreas']),
    ('What is diabetes?', 'Diabetes is a disease that occurs when your blood glucose', ['blood glucose is too high']),
    ('What are the common symptoms of diabetes?', 'Common symptoms of diabetes include', ['increased urination', 'feeling very thirsty']),
    ('What are the symptoms of type 1 diabetes in children?', 'eating more but losing weight', ['urinate more often', 'losing weight']),
    ('Do people with type 2 diabetes always have symptoms?', 'Symptoms of type 2 diabetes can develop slowly', ['no', 'many people have no symptoms']),
    ('Does gestational diabetes cause symptoms?', 'Most people with this type of diabetes have no symptoms', ['most people have no symptoms']),
    ('What causes type 1 diabetes?', 'Type 1 diabetes develops when the body’s immune system destroys the cells in the pancreas', ['immune system', 'the immune system destroys beta cells']),
    ('What causes type 2 diabetes?', 'Type 2 diabetes is the most common form of diabetes. Type 2 diabetes develops', ['insulin resistance', 'the pancreas cannot make enough insulin']),
    ('Which medicines can cause diabetes?', 'antipsychotics that are used', ['antipsychotics', 'glucocorticoids']),
    ('Who should be tested for type 2 diabetes?', 'Experts recommend routine testing for type 2 diabetes if you have certain risk factors', ['age 35 or older', 'people with risk factors']),
    ('When is gestational diabetes tested?', 'usually occurs between 24 and 28 weeks of pregnancy', ['24 to 28 weeks', 'between 24 and 28 weeks']),
    ('Is there a cure for type 1 diabetes?', 'have the disease for the rest of their lives', ['no', 'for the rest of their lives']),
    ('What eye problems can diabetes cause?', 'These conditions include diabetic retinopathy, diabetic macular edema, cataracts, and glaucoma', ['retinopathy', 'macular edema', 'cataracts', 'glaucoma']),
    ('What is diabetic kidney disease?', 'Diabetic kidney disease is a type of kidney disease caused by diabetes', ['kidney disease caused by diabetes']),
    ('Is diabetes a leading cause of kidney disease?', 'Diabetes is the leading cause of kidney disease', ['yes']),
    ('What is peripheral neuropathy?', 'Peripheral neuropathy is a type of nerve damage that typically affects the feet and legs', ['nerve damage in the feet and legs']),
    ('What does autonomic neuropathy damage?', 'Autonomic neuropathy is damage to nerves that control your internal organs', ['nerves that control internal organs']),
    ('How often should I check my feet?', 'Check your feet every day', ['every day', 'each day']),
    ('How does diabetes affect the heart?', 'Having diabetes means you are more likely to develop heart disease', ['more likely to develop heart disease']),
    ('What should I do when my blood glucose is below 70 mg/dL?', 'Eat or drink 15 to 20 grams of glucose or carbohydrates right away', ['15 to 20 grams of glucose', 'glucose tablets']),
    ('What are the complications of severe low blood glucose?', 'severely low blood glucose can cause serious complications, including passing out, coma, or death', ['passing out', 'coma', 'death']),
    ('What is the plate method?', 'The plate method helps you control portion sizes', ['control portion sizes', 'divides a plate into three sections']),
    ('What drinks should a person with diabetes choose?', 'Also try to choose drinks with little or no added sugar', ['water', 'drinks with little or no added sugar']),
    ('How many minutes of physical activity per week are recommended?', 'at least 150 minutes of moderate-intensity physical activity every week', ['150 minutes', 'at least 150 minutes']),
    ('Should I eat before, during, or after physical activity?', 'Some diabetes medicines, such as sulfonylureas', ['ask your health care team']),
    ('What is a normal A1C level?', 'A normal A1C level is below 5.7 percent', ['below 5.7 percent', '5.7 percent']),
    ('What does the A1C test measure?', 'average blood glucose levels over the past 3 months', ['average blood glucose over 3 months', '3 months']),
    ('What blood glucose targets do many people aim for before a meal?', 'Before a meal: 80 to 130 mg/dL', ['80 to 130 mg/dL']),
    ('What is continuous glucose monitoring?', 'Continuous glucose monitoring means using a device to automatically estimate your blood glucose level', ['a device that estimates blood glucose all day and night']),
    ('What is an artificial pancreas?', 'An artificial pancreas is a system of three devices', ['a system of three devices']),
    ('Can insulin be taken with an inhaler?', 'breathing powdered insulin into your mouth', ['yes', 'powdered insulin']),
    ('What do most people with type 2 diabetes take first?', 'Most people with type 2 diabetes start with metformin pills', ['metformin']),
    ('How does metformin work?', 'Metformin helps your liver make less glucose and helps your body use insulin better', ['helps your liver make less glucose']),
    ('How can I lower my chances of developing type 2 diabetes?', 'Here are some things you can change to lower your risk', ['lose weight', 'be physically active']),
    ('How much weight loss may prevent or delay diabetes?', 'losing 5 to 7 percent of your starting weight', ['5 to 7 percent']),
    ('What is prediabetes?', 'People have prediabetes if they have blood glucose levels that are higher than normal', ['blood glucose higher than normal but not diabetes']),
    ('What is diabetic ketoacidosis?', 'High levels of ketones can cause DKA', ['a medical emergency caused by ketones']),
    ('What are the symptoms of diabetic ketoacidosis?', 'having fruity-smelling breath', ['fruity-smelling breath', 'nausea or vomiting']),
    ('What is hyperglycemia?', 'Hyperglycemia is', ['high blood glucose']),
    ('What are the symptoms of low blood glucose?', 'shaky', ['shaky', 'sweating']),
]


def main():
    corpus = [json.loads(x) for x in (DATA / 'diabetes_corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    qa = []
    for i, (question, needle, refs) in enumerate(SPECS):
        gold = [x['id'] for x in corpus if needle.lower() in x['text'].lower()]
        if not gold:
            raise SystemExit(f'Needle không có trong kho: {needle!r}')
        qa.append({'id': f'DM{i + 1:02d}', 'question': question, 'references': refs, 'gold_ids': gold, 'needle': needle,
                   'annotation': 'authored demonstration; needle-anchored to source text; not independently reviewed'})
    (DATA / 'diabetes_qa_demo.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print(len(qa), 'câu hỏi;', sum(len(q['gold_ids']) > 1 for q in qa), 'câu có nhiều hơn một đoạn đúng')


if __name__ == '__main__':
    main()
