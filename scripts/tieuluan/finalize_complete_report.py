"""Update Word fields, validate structure, and preview pages without a PDF."""
import io
import json
import re
import sys
import zipfile
from docx import Document
from .build_complete_report import OUTPUT, DEST


def validate():
    doc = Document(OUTPUT)
    texts = [p.text for p in doc.paragraphs]
    texts += [c.text for t in doc.tables for row in t.rows for c in row.cells]
    merged = '\n'.join(texts)
    assert not re.search(r'\{\{(?:complete:|[tv]:)|\[@\w', merged)
    chapter_heads = [p.text for p in doc.paragraphs
                     if p.style.name == 'Heading 1'
                     and p.text.startswith('CHƯƠNG ')]
    assert [re.search(r'CHƯƠNG (\d)',s)[1] for s in chapter_heads] == list('123456')
    assert 'Chương 7 –' not in merged
    captions = [p.text for p in doc.paragraphs if p.style.name=='CaptionTable']
    labels = [re.match(r'Bảng ([\dA-Z]+\.\d+)\.', s)[1] for s in captions]
    assert len(labels) == len(set(labels)), 'Duplicate table labels'
    for n in '3456':
        ids = [int(s.split('.')[1]) for s in labels if s.startswith(n+'.')]
        assert ids == list(range(1,len(ids)+1)), (n, ids)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
    for term in ['CNN','RNN','LSTM','Knowledge Graph','RAG','Agent',
                 'scratch_deep','keras_deep','torch_deep','train_batch',
                 'MultiAgentTeam','Phụ lục C','Phụ lục D']:
        assert term in merged, term
    notebook_labels=['CNN-03','CNN-07','CNN-09','RNN-05','RNN-07',
                     'LSTM-04','LSTM-07','KG-02','KG-05',
                     'AGENT-04','AGENT-05']
    if (DEST.parent.parent/'results/tieuluan/complete/notebooks/execution.json').exists():
        for label in notebook_labels:
            assert any(p.text.startswith('Cell '+label+' ') for p in doc.paragraphs),label
    return dict(chapters=chapter_heads,tables=len(doc.tables),
                captions=len(captions),inline_shapes=len(doc.inline_shapes),
                embedded_notebook_cells=sum(bool(re.match(
                    r'^Cell (?:CNN|RNN|LSTM|KG|AGENT)-\d{2}\b',p.text))
                    for p in doc.paragraphs),
                file_bytes=OUTPUT.stat().st_size)


def main():
    sys.stdout.reconfigure(encoding='utf8')
    import win32com.client
    from PIL import Image
    word = win32com.client.DispatchEx('Word.Application')
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    previews = DEST/'previews_notebook'
    previews.mkdir(exist_ok=True)
    info = validate()
    try:
        document = word.Documents.Open(str(OUTPUT.resolve()),
                                       ReadOnly=False,AddToRecentFiles=False)
        assert not document.ReadOnly
        document.ActiveWindow.View.Type = 3
        document.Fields.Update()
        for toc in document.TablesOfContents:
            toc.Update()
        document.Repaginate()
        for toc in document.TablesOfContents:
            toc.Update()
        document.Repaginate()
        info['total_pages'] = document.ComputeStatistics(2)
        heads = []
        code_pages=[]
        for p in document.Paragraphs:
            if p.Range.Text.strip().startswith(('Cell CNN-03','Cell KG-02','Cell AGENT-04')):
                code_pages.append(p.Range.Information(3))
            if p.OutlineLevel==1:
                text = p.Range.Text.strip()
                if text:
                    heads.append(dict(title=text,page=p.Range.Information(3)))
        info['headings'] = heads
        document.Save()
        print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
        # Word's native page rendering is an EMF; no PDF is generated.
        selected = [1,2]+code_pages
        selected += [x['page'] for x in heads
                     if x['title'].startswith(('CHƯƠNG 3','CHƯƠNG 4','CHƯƠNG 5','CHƯƠNG 6','PHỤ LỤC'))]
        errors = []
        for number in sorted(set(selected)):
            try:
                document.ActiveWindow.ScrollIntoView(
                    document.GoTo(1,1,number),True)
                page = document.ActiveWindow.Panes(1).Pages.Item(number)
                raw = bytes(page.EnhMetaFileBits)
                im = Image.open(io.BytesIO(raw))
                im.load()
                im.convert('RGB').save(previews/f'page_{number:03}.png')
            except Exception as exc:
                errors.append(str(exc)[:250])
                break
        info['preview_errors'] = errors
        info['preview_images'] = [str(p) for p in previews.glob('*.png')]
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    info.update(validate())
    (DEST/'verification_docx.json').write_text(
        json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'pages':info['total_pages'],
                      'previews':info['preview_images'],
                      'preview_errors':info['preview_errors']},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
