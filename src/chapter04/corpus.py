"""Trích văn bản từ trang web y tế và tách thành các đoạn (chunk) cho kho RAG.

Điểm khác với cách tách cũ (mỗi thẻ p/li là một đoạn, bỏ đoạn dưới 12 từ):
  * danh sách gạch đầu dòng được gộp với câu dẫn đứng trước thành MỘT đoạn
    ("Common symptoms of diabetes include: increased urination; feeling very thirsty; ...");
  * tiêu đề trang và tiêu đề mục nằm ngay trong văn bản đoạn, để BM25 và embedding khớp được câu hỏi
    như "What causes type 2 diabetes?";
  * các khối được gom lại trong cùng một mục cho đủ ~60-150 từ, không cắt ngang danh sách.
"""
import re

SKIP_TAGS = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'noscript', 'button', 'svg', 'figure']
SKIP_SECTIONS = re.compile(r'^(clinical trials?|our research|research|share|acknowledg|related|references?|sources?|more information|'
                           r'last reviewed|find a health care|español|on this page|in this section|contact|print|translations?|'
                           r'content from the national|related topics|health topic|see also|for more information)', re.I)
SKIP_PARAGRAPH = re.compile(r'(provided as a service|last reviewed|reviewed by|page last|^source:|^español|sign up for|subscribe|'
                            r'share this|print this|^in this section|content produced by|^en español|clinical trials? (?:for|on)|nih publication)', re.I)

MIN_WORDS = 55
TARGET_WORDS = 120
MAX_WORDS = 150
HARD_MAX = 185


def decode_html(raw):
    """Giải mã byte của trang: UTF-8 nếu hợp lệ, ngược lại cp1252 (các trang NIDDK cũ)."""
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('cp1252', errors='replace')


def clean(text):
    text = re.sub(r'\s+', ' ', text.replace('\xa0', ' ')).strip()
    return re.sub(r'\s+([,.;:?!])', r'\1', text)


def extract_blocks(html):
    """Trả về (tiêu đề trang, danh sách khối). Khối: ('h', mức, text) | ('p', text) | ('ul', [mục])."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(decode_html(html), 'lxml')
    main = soup.find('main') or soup.find(id='main-content') or soup.body or soup
    for tag in main(SKIP_TAGS):
        tag.decompose()
    for tag in main.find_all(class_=re.compile(r'breadcrumb|share|social|toc|on-this-page|language|related', re.I)):
        tag.decompose()
    blocks = []
    title = ''
    for el in main.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol']):
        if el.find_parent(['ul', 'ol', 'table']) is not None and el.name in ('p', 'ul', 'ol'):
            continue
        if el.name in ('h1', 'h2', 'h3', 'h4'):
            text = clean(el.get_text(' ', strip=True))
            if not text:
                continue
            if el.name == 'h1' and not title:
                title = text
            blocks.append(('h', int(el.name[1]), text))
        elif el.name == 'p':
            text = clean(el.get_text(' ', strip=True))
            if text.lower().startswith('in this section'):
                blocks.append(('toc',))
            elif len(text.split()) >= 4 and not SKIP_PARAGRAPH.search(text):
                blocks.append(('p', text))
        else:
            items = []
            for li in el.find_all('li', recursive=False):
                text = clean(li.get_text(' ', strip=True)).rstrip('.;')
                if text:
                    items.append(text)
            if items and not _looks_like_menu(items):
                blocks.append(('ul', items))
    # bỏ khối "In this section" (mục lục) và danh sách đứng ngay sau
    cleaned = []
    skip_list = False
    for b in blocks:
        if b[0] == 'toc':
            skip_list = True
            continue
        if b[0] == 'ul' and skip_list:
            skip_list = False
            continue
        skip_list = False
        cleaned.append(b)
    return title, cleaned


def _looks_like_menu(items):
    """Mục lục/menu: phần lớn mục là câu hỏi ngắn hoặc dòng "Clinical Trials for ..."."""
    questions = sum(x.endswith('?') or x.lower().startswith('clinical trials') for x in items)
    return len(items) >= 3 and questions >= 0.6 * len(items)


def sections(blocks, title):
    """Gom khối theo mục: [(đường tiêu đề, [mảnh văn bản])]. Danh sách gộp vào đoạn dẫn ngay trước."""
    result = []
    stack = {}
    current = None

    def start():
        nonlocal current
        path = [title] if title else []
        path += [stack[k] for k in sorted(stack)]
        # bỏ tiêu đề trùng liên tiếp
        dedup = []
        for p in path:
            if not dedup or dedup[-1].lower() != p.lower():
                dedup.append(p)
        current = (dedup, [])
        result.append(current)

    skipping = None
    for b in blocks:
        if b[0] == 'h':
            level, text = b[1], b[2]
            if level == 1:
                continue
            if skipping is not None and level > skipping:
                continue
            skipping = None
            if SKIP_SECTIONS.match(text):
                skipping = level
                continue
            for k in [k for k in stack if k >= level]:
                del stack[k]
            stack[level] = text
            current = None
            continue
        if skipping is not None:
            continue
        if current is None:
            start()
        pieces = current[1]
        if b[0] == 'p':
            pieces.append(b[1])
        else:
            items = '; '.join(b[1])
            if pieces and not pieces[-1].startswith('\x00'):
                lead = pieces[-1]
                if lead.endswith(':') or not lead.endswith(('.', '?', '!')):
                    pieces[-1] = lead.rstrip(':') + ': ' + items + '.'
                else:
                    pieces[-1] = lead + ' ' + items[:1].upper() + items[1:] + '.'
            else:
                pieces.append(items[:1].upper() + items[1:] + '.')
    return [(path, pieces) for path, pieces in result if pieces]


def _split_piece(piece, limit):
    """Cắt một mảnh quá dài theo câu (hoặc theo mục danh sách) thành các phần không vượt giới hạn."""
    words = piece.split()
    if len(words) <= limit:
        return [piece]
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', piece)
    if len(parts) == 1:
        parts = re.split(r'(?<=;)\s+', piece)
    out, cur = [], ''
    for part in parts:
        if cur and len((cur + ' ' + part).split()) > limit:
            out.append(cur)
            cur = part
        else:
            cur = (cur + ' ' + part).strip()
    if cur:
        out.append(cur)
    return out


def heading_text(path):
    return ' — '.join(path)


def chunk_page(slug, title, blocks, url, accessed, org, license_note=''):
    """Chia các khối của một trang thành các đoạn có id ổn định dạng `slug:số`."""
    chunks = []
    for path, pieces in sections(blocks, title):
        expanded = []
        for piece in pieces:
            expanded.extend(_split_piece(piece, MAX_WORDS))
        groups, cur = [], []
        for piece in expanded:
            size = sum(len(x.split()) for x in cur)
            if cur and size + len(piece.split()) > MAX_WORDS and size >= MIN_WORDS:
                groups.append(cur)
                cur = []
            cur.append(piece)
            if sum(len(x.split()) for x in cur) >= TARGET_WORDS:
                groups.append(cur)
                cur = []
        if cur:
            if groups and sum(len(x.split()) for x in cur) < MIN_WORDS // 2 and \
                    sum(len(x.split()) for x in groups[-1] + cur) <= HARD_MAX:
                groups[-1].extend(cur)
            else:
                groups.append(cur)
        head = heading_text(path)
        for group in groups:
            body = ' '.join(group)
            text = f'{head}. {body}' if head and not head.endswith(('?', '.')) else f'{head} {body}'.strip()
            if len(body.split()) < 8:
                continue
            chunks.append({'id': f'{slug}:{len(chunks)}', 'text': text, 'title': title, 'section': path[-1] if path else title,
                           'source_url': url, 'domain': 'diabetes', 'language': 'en', 'accessed': accessed, 'source_org': org})
    return chunks
