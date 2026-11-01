"""Dựng kho tài liệu tiểu đường (diabetes_corpus.jsonl) từ các trang y tế công khai của Mỹ.

Chạy:  .venv\\Scripts\\python.exe scripts/chapter04/build_diabetes_corpus.py [--refresh]
  * Tải HTML thô vào data/chapter04/raw_pages/ (bỏ qua nếu đã có, trừ khi --refresh).
  * Trích văn bản, tách đoạn bằng src/chapter04/corpus.py (giữ danh sách gạch đầu dòng và tiêu đề mục).
  * Ghi data/chapter04/diabetes_corpus.jsonl và data/chapter04/diabetes_corpus_manifest.json.
Sau đó chạy scripts/chapter05/evaluate_diabetes_retrieval.py để làm mới embedding, đồ thị và số đánh giá.
"""
from pathlib import Path
import collections, hashlib, json, sys, time
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.chapter04.corpus import chunk_page, extract_blocks  # noqa: E402

DATA = ROOT / 'data/chapter04'
RAW = DATA / 'raw_pages'
ACCESSED = time.strftime('%Y-%m-%d')
NIDDK = 'https://www.niddk.nih.gov/health-information/diabetes/overview/'
CDC = 'https://www.cdc.gov/diabetes/'
HEADERS = {'User-Agent': 'Mozilla/5.0 (educational coursework, contact trung241az@gmail.com)'}
LICENSES = {
    'NIDDK': 'US government work (NIH/NIDDK); public domain, may be reproduced with attribution (niddk.nih.gov/about-niddk/policies-and-guidelines)',
    'CDC': 'US government work (CDC); public domain, content may be copied and used without permission (cdc.gov/other/policies.html)',
    'MedlinePlus': 'US National Library of Medicine; MedlinePlus health topic text is not copyrighted unless noted (medlineplus.gov/about/using/usingcontent)',
}
# (slug, nguồn, URL, chủ đề)
PAGES = [
    ('what-is-diabetes', 'NIDDK', NIDDK + 'what-is-diabetes', 'tổng quan, các loại tiểu đường'),
    ('symptoms-causes', 'NIDDK', NIDDK + 'symptoms-causes', 'triệu chứng và nguyên nhân'),
    ('tests-diagnosis', 'NIDDK', NIDDK + 'tests-diagnosis', 'xét nghiệm và chẩn đoán'),
    ('managing-diabetes', 'NIDDK', NIDDK + 'managing-diabetes', 'kiểm soát bệnh, ABC, theo dõi đường huyết'),
    ('type-1-diabetes', 'NIDDK', NIDDK + 'what-is-diabetes/type-1-diabetes', 'tiểu đường típ 1'),
    ('type-2-diabetes', 'NIDDK', NIDDK + 'what-is-diabetes/type-2-diabetes', 'tiểu đường típ 2'),
    ('gestational-diabetes', 'NIDDK', NIDDK + 'what-is-diabetes/gestational/all-content', 'tiểu đường thai kỳ'),
    ('prediabetes', 'NIDDK', NIDDK + 'what-is-diabetes/prediabetes-insulin-resistance', 'tiền tiểu đường, kháng insulin'),
    ('risk-factors-type-2', 'NIDDK', NIDDK + 'risk-factors-type-2-diabetes', 'yếu tố nguy cơ típ 2'),
    ('preventing-type-2', 'NIDDK', NIDDK + 'preventing-type-2-diabetes', 'phòng ngừa típ 2'),
    ('healthy-living', 'NIDDK', NIDDK + 'healthy-living-with-diabetes', 'ăn uống, hoạt động thể chất, sống chung với bệnh'),
    ('insulin-medicines', 'NIDDK', NIDDK + 'insulin-medicines-treatments', 'insulin, metformin và các thuốc'),
    ('a1c-test', 'NIDDK', NIDDK + 'tests-diagnosis/a1c-test', 'xét nghiệm A1C, mục tiêu'),
    ('cgm', 'NIDDK', NIDDK + 'managing-diabetes/continuous-glucose-monitoring', 'đo đường huyết liên tục'),
    ('eye-disease', 'NIDDK', NIDDK + 'preventing-problems/diabetic-eye-disease', 'biến chứng mắt'),
    ('kidney-disease', 'NIDDK', NIDDK + 'preventing-problems/diabetic-kidney-disease', 'biến chứng thận'),
    ('foot-problems', 'NIDDK', NIDDK + 'preventing-problems/foot-problems', 'biến chứng bàn chân'),
    ('heart-disease-stroke', 'NIDDK', NIDDK + 'preventing-problems/heart-disease-stroke', 'tim mạch, đột quỵ'),
    ('neuropathy', 'NIDDK', NIDDK + 'preventing-problems/nerve-damage-diabetic-neuropathies/all-content', 'biến chứng thần kinh'),
    ('hypoglycemia', 'NIDDK', NIDDK + 'preventing-problems/low-blood-glucose-hypoglycemia', 'hạ đường huyết'),
    ('cdc-symptoms', 'CDC', CDC + 'signs-symptoms/index.html', 'triệu chứng (CDC)'),
    ('cdc-treatment', 'CDC', CDC + 'treatment/index.html', 'điều trị (CDC)'),
    ('cdc-type-2', 'CDC', CDC + 'about/about-type-2-diabetes.html', 'tiểu đường típ 2 (CDC)'),
    ('cdc-heart', 'CDC', CDC + 'diabetes-complications/diabetes-and-your-heart.html', 'tim mạch (CDC)'),
    ('cdc-vision', 'CDC', CDC + 'diabetes-complications/diabetes-and-vision-loss.html', 'mắt, thị lực (CDC)'),
    ('cdc-physical-activity', 'CDC', CDC + 'living-with/physical-activity.html', 'hoạt động thể chất (CDC)'),
    ('cdc-type-1', 'CDC', CDC + 'about/about-type-1-diabetes.html', 'tiểu đường típ 1 (CDC)'),
    ('medlineplus-hyperglycemia', 'MedlinePlus', 'https://medlineplus.gov/hyperglycemia.html', 'tăng đường huyết'),
]


def fetch(url, name, refresh):
    path = RAW / name
    if refresh or not path.exists():
        for attempt in range(3):
            response = requests.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                path.write_bytes(response.content)
                break
            time.sleep(2)
        else:
            raise RuntimeError(f'{url}: HTTP {response.status_code}')
    return path.read_bytes()


def main():
    refresh = '--refresh' in sys.argv
    RAW.mkdir(parents=True, exist_ok=True)
    corpus, manifest = [], []
    for slug, org, url, topic in PAGES:
        raw = fetch(url, slug + '.html', refresh)
        title, blocks = extract_blocks(raw)
        chunks = chunk_page(slug, title or slug, blocks, url, ACCESSED, org)
        corpus.extend(chunks)
        words = [len(c['text'].split()) for c in chunks]
        manifest.append({'slug': slug, 'org': org, 'url': url, 'title': title, 'topic': topic, 'accessed': ACCESSED, 'license': LICENSES[org],
                         'raw_file': f'raw_pages/{slug}.html', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                         'chunks': len(chunks), 'words_min': min(words, default=0), 'words_max': max(words, default=0),
                         'words_mean': round(sum(words) / max(1, len(words)), 1)})
        (DATA / f'{slug}.txt').write_text('\n\n'.join(c['text'] for c in chunks), encoding='utf-8')
        print(f'{slug:28s} {org:12s} {len(chunks):3d} đoạn  ({min(words, default=0)}-{max(words, default=0)} từ)')
    (DATA / 'diabetes_corpus.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in corpus) + '\n', encoding='utf-8')
    by_org = collections.Counter(c['source_org'] for c in corpus)
    summary = {'pages': len(PAGES), 'chunks': len(corpus), 'chunks_by_org': dict(by_org), 'accessed': ACCESSED,
               'chunker': 'src/chapter04/corpus.py (giữ danh sách gạch đầu dòng, tiêu đề mục trong đoạn, 55-150 từ)', 'pages_detail': manifest}
    (DATA / 'diabetes_corpus_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Tổng:', len(PAGES), 'trang,', len(corpus), 'đoạn', dict(by_org))


if __name__ == '__main__':
    main()
