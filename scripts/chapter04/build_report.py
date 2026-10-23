"""Build the Vietnamese chapter as DOCX, with original diagrams and real metrics."""
from pathlib import Path
import json, re
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from report_content import enrich_report

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/report_chapter_04'
RESULT=ROOT/'results/chapter04'
summary=json.loads((RESULT/'summary.json').read_text(encoding='utf-8'))
manifest=json.loads((RESULT/'dataset_manifest.json').read_text(encoding='utf-8'))
SOURCE=DEST/'CHUONG_04_KNOWLEDGE_RAG.md'
BLUE='#15395B'; TEAL='#167C86'; GRAY='#556575'
fontroot=Path('C:/Windows/Fonts')
def font(size,bold=False):
    return ImageFont.truetype(str(fontroot/('arialbd.ttf' if bold else 'arial.ttf')),size)
def wrap(draw,text,width,f):
    lines=[]
    for paragraph in text.split('\n'):
        line=''
        for word in paragraph.split():
            trial=(line+' '+word).strip()
            if draw.textlength(trial,font=f)>width and line: lines.append(line); line=word
            else: line=trial
        lines.append(line)
    return lines
def box(draw,xy,title,desc,color=BLUE):
    x,y,w,h=xy
    draw.rounded_rectangle((x,y,x+w,y+h),radius=18,fill='#F1F6FA',outline=color,width=3)
    yy=y+20
    for line in wrap(draw,title,w-32,font(26,True)):
        draw.text((x+16,yy),line,font=font(26,True),fill=color); yy+=34
    yy+=8
    for line in wrap(draw,desc,w-32,font(22)):
        draw.text((x+16,yy),line,font=font(22),fill=GRAY); yy+=29
def arrow(draw,start,end):
    draw.line((start,end),fill=TEAL,width=5)
    x,y=end; sx,sy=start
    if abs(x-sx)>abs(y-sy):
        sign=1 if x>sx else -1; draw.polygon([(x,y),(x-sign*15,y-9),(x-sign*15,y+9)],fill=TEAL)
    else:
        sign=1 if y>sy else -1; draw.polygon([(x,y),(x-9,y-sign*15),(x+9,y-sign*15)],fill=TEAL)
def canvas(title,height=760):
    im=Image.new('RGB',(1500,height),'white'); d=ImageDraw.Draw(im)
    d.text((45,25),title,font=font(34,True),fill=BLUE)
    return im,d

im,d=canvas('KB và RAG: hai luồng xử lý',720)
items=[('Nguồn','Văn bản, bảng, dữ liệu gốc'),('Chuẩn hóa','Chunk + metadata + nguồn'),('Chỉ mục / KG','Từ khóa, vector, quan hệ'),('Câu hỏi','Miền, thời điểm, quyền'),('Truy hồi','Lọc nguồn, xếp hạng'),('Trả lời','LLM + dẫn nguồn + kiểm tra')]
for i,(title,desc) in enumerate(items):
    row=i//3; col=i%3
    box(d,(65+col*490,145+row*300,390,185),title,desc)
    if col<2: arrow(d,(455+col*490,237+row*300),(535+col*490,237+row*300))
d.text((65,105),'Ngoại tuyến: nhập liệu và lập chỉ mục',font=font(24,True),fill=TEAL)
d.text((65,405),'Trực tuyến: hỏi đáp theo bằng chứng',font=font(24,True),fill=TEAL)
arrow(d,(1120,330),(850,445))
im.save(DEST/'rag_pipeline.png')

im,d=canvas('KG tài chính: giữ nguồn, kỳ và đơn vị',760)
for xy,title,desc in [((65,160,340,165),'Company','Ticker + thị trường'),((580,160,340,165),'Report','Ngày công bố + URL'),((1095,160,340,165),'Observation','Giá trị + đơn vị + tiền tệ'),((65,500,340,165),'Indicator','Mã + tên chỉ tiêu'),((580,500,340,165),'Period','Bắt đầu / kết thúc'),((1095,500,340,165),'TableRow','Dòng + cột + chú thích')]: box(d,xy,title,desc)
arrow(d,(405,240),(580,240)); d.text((425,202),'PUBLISHED',font=font(18),fill=TEAL)
arrow(d,(920,240),(1095,240));d.text((930,200),'HAS_OBSERVATION',font=font(16),fill=TEAL)
arrow(d,(1265,325),(1265,500))
arrow(d,(1160,325),(750,500))
arrow(d,(1110,325),(235,500))
for pos,label in [((1120,398),'SUPPORTED_BY'),((840,420),'FOR_PERIOD'),((435,388),'OF_INDICATOR')]:
    bounds=d.textbbox(pos,label,font=font(20))
    d.rectangle((bounds[0]-5,bounds[1]-5,bounds[2]+5,bounds[3]+5),fill='white')
    d.text(pos,label,font=font(20),fill=TEAL)
im.save(DEST/'finance_graph.png')

im,d=canvas('BM25 trên 883 câu hỏi FinQA dev',870)
metrics=[('Hit@1','hit_at_1'),('Hit@5','hit_at_5'),('Recall@5','recall_at_5'),('MRR@5','mrr_at_5')]
for tick in range(0,101,20):
    y=680-tick*5
    d.line((125,y,1430,y),fill='#DAE2EA',width=2)
    d.text((55,y-14),str(tick)+'%',font=font(21),fill=GRAY)
for i,(label,key) in enumerate(metrics):
    base=210+i*320
    for j,mode in enumerate(['global','report_filtered']):
        value=summary['retrieval'][mode][key]; x=base+j*100; top=680-value*500
        d.rectangle((x,top,x+78,680),fill=BLUE if j==0 else TEAL)
        d.text((x-7,top-35),f'{value*100:.1f}%',font=font(22,True),fill=BLUE)
    d.text((base+30,702),label,font=font(25,True),fill=BLUE)
d.rectangle((210,775,235,800),fill=BLUE);d.text((247,772),'Toàn kho',font=font(23),fill=GRAY)
d.rectangle((540,775,565,800),fill=TEAL);d.text((577,772),'Biết đúng báo cáo (oracle)',font=font(23),fill=GRAY)
im.save(DEST/'retrieval_metrics.png')

im,d=canvas('Agent: điều phối công cụ trong kiến trúc đề xuất',800)
for xy,title,desc in [((60,130,370,165),'Web / Mobile','Câu hỏi + miền + thời điểm'),((560,130,380,165),'FastAPI + Agent','Xác thực, chọn công cụ'),((1080,130,350,165),'LLM adapter','Sinh trả lời có dẫn nguồn'),((60,490,370,180),'Retriever','BM25 / vector + metadata'),((560,490,380,180),'Graph service','Quan hệ + truy nguồn'),((1080,490,350,180),'Calculator / Verify','Số học + kiểm tra citation')]: box(d,xy,title,desc)
arrow(d,(430,210),(560,210));arrow(d,(940,210),(1080,210))
arrow(d,(670,295),(245,490));arrow(d,(750,295),(750,490));arrow(d,(835,295),(1255,490))
d.text((70,720),'Đã chạy: KG tài liệu + BM25/dense/hybrid + FLAN-T5-small + Agent API cục bộ.',font=font(24),fill=GRAY)
im.save(DEST/'agent_deployment.png')

def table_md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
uci=summary['uci']; splits=summary['finqa_splits']; r=summary['retrieval']
replacements={
 '{{DATASET_TABLE}}':table_md(['Nguồn','Quy mô đã kiểm tra','Vai trò'],[
 ['UCI Diabetes','101.766 lượt; 50 cột; 71.518 bệnh nhân','Dữ liệu cấu trúc về nhập viện'],
 ['NIDDK','4 trang HTML; '+str(summary['diabetes_chunks'])+' đoạn đủ điều kiện','Corpus kiến thức tiểu đường'],
 ['FinQA','Train: 6.251; dev: 883; test: 1.147','Hỏi đáp và bằng chứng tài chính']]),
 '{{UCI_TABLE}}':table_md(['Nhãn readmitted','Số lượt','Tỷ lệ'],[[k,f'{uci["readmitted"][k]:,}'.replace(',','.'),f'{100*uci["readmitted"][k]/uci["rows"]:.2f}%'] for k in ['NO','>30','<30']]),
 '{{MISSING_TABLE}}':table_md(['Cột','Số giá trị ?','Tỷ lệ thiếu'],[[k,f'{uci["missing_question_mark"][k]:,}'.replace(',','.'),f'{100*uci["missing_question_mark"][k]/uci["rows"]:.2f}%'] for k in ['weight','medical_specialty','payer_code','race','diag_1']]),
 '{{RETRIEVAL_TABLE}}':table_md(['Cấu hình','Hit@1','Hit@5','Recall@5','MRR@5','Thời gian'],[[label]+[f'{100*r[mode][key]:.2f}%' for key in ['hit_at_1','hit_at_5','recall_at_5','mrr_at_5']]+[f'{r[mode]["seconds"]:.2f} s'] for label,mode in [('Global BM25','global'),('Report-filtered BM25 (oracle)','report_filtered')]]),
 '{{RESULT_DISCUSSION}}':f'Tập ứng viên gồm {summary["finqa_dev_chunks"]:,} chunk thuộc {summary["finqa_dev_reports"]} context báo cáo sau khử trùng. Cả 883 câu hỏi dev có đủ khóa bằng chứng để chấm, không có câu bị loại. Khi thu hẹp vào đúng context, Hit@5 tăng {(r["report_filtered"]["hit_at_5"]-r["global"]["hit_at_5"])*100:.2f} điểm phần trăm; Recall@5 tăng {(r["report_filtered"]["recall_at_5"]-r["global"]["recall_at_5"])*100:.2f} điểm phần trăm. Kết quả gợi ý giá trị của việc xác định đúng nguồn trước khi tìm đoạn, nhưng không đo một bộ chọn báo cáo tự động. Tổng thời gian vòng truy hồi giảm từ {r["global"]["seconds"]:.2f} xuống {r["report_filtered"]["seconds"]:.2f} giây trên máy thực thi. Sự khác biệt thời gian chỉ có ý nghĩa trong baseline và môi trường này.'
}
text=enrich_report(SOURCE.read_text(encoding='utf-8'),ROOT)
for key,val in replacements.items(): text=text.replace(key,val)
assert '{{' not in text
(DEST/'CHUONG_04_KNOWLEDGE_RAG_HOAN_CHINH.md').write_text(text,encoding='utf-8')
doc=Document(); sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(2.2);sec.bottom_margin=Cm(2.2);sec.left_margin=Cm(2.7);sec.right_margin=Cm(2)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(13)
normal.paragraph_format.line_spacing=1.25;normal.paragraph_format.space_after=Pt(7)
for name,size in [('Title',27),('Heading 1',19),('Heading 2',16),('Heading 3',14)]:
    s=doc.styles[name];s.font.name='Times New Roman';s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor.from_string('15395B')
    s.paragraph_format.space_before=Pt(14);s.paragraph_format.space_after=Pt(8)
    s.paragraph_format.keep_with_next=True
sec.header.paragraphs[0].text='HỆ THỐNG THÔNG MINH  |  CHƯƠNG 4: KNOWLEDGE & RAG'
sec.header.paragraphs[0].runs[0].font.size=Pt(9)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('Trang ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
sec.different_first_page_header_footer=True
doc.core_properties.title='Chương 4: Knowledge & RAG'
doc.core_properties.subject='KB/KG, RAG, tiểu đường, chứng khoán và triển khai Agent'
doc.core_properties.author=''
doc.add_paragraph('BÁO CÁO HỌC PHẦN\nHỆ THỐNG THÔNG MINH',style='Subtitle').alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('\n\n')
doc.add_paragraph('CHƯƠNG 4',style='Title').alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('KNOWLEDGE & RAG',style='Title').alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('Cơ sở tri thức • Đồ thị tri thức • Truy hồi tăng cường\nNotebook tiểu đường và tài chính đã thực thi\nAgent công cụ chạy qua API cục bộ')
p.alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('\n\n')
p=doc.add_paragraph('4 notebook đã chạy • Code và output • Dataset và model đã tải\nKG, BM25/dense/hybrid, sinh câu trả lời và HTTP Agent\nNgày hoàn thiện: 13/10/2026');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_page_break()
doc.add_heading('MỤC LỤC CHƯƠNG',level=1)
for line in text.splitlines():
    if line.startswith('## '):
        doc.add_paragraph(line[3:])
doc.add_paragraph('Phạm vi: chương báo cáo độc lập, có thể ghép vào báo cáo tổng thể. Kết quả đã thực hiện và thiết kế đề xuất được phân biệt trong từng phần.')
doc.add_page_break()

def hyperlink(p,label,url):
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    rid=p.part.relate_to(url,RT.HYPERLINK,is_external=True)
    h=OxmlElement('w:hyperlink');h.set(qn('r:id'),rid)
    run=OxmlElement('w:r');pr=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'156B8A');pr.append(color)
    run.append(pr);t=OxmlElement('w:t');t.text=label;run.append(t);h.append(run);p._p.append(h)
def rich(p,body):
    offset=0
    for match in re.finditer(r'https?://\S+',body):
        p.add_run(body[offset:match.start()]);url=match.group().rstrip('.,;');hyperlink(p,url,url);offset=match.end()
    p.add_run(body[offset:])
def make_table(rows):
    t=doc.add_table(rows=1,cols=len(rows[0]));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for rownum,row in enumerate(rows):
        cells=t.rows[0].cells if rownum==0 else t.add_row().cells
        for cell,value in zip(cells,row):
            cell.text=value
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4);p.paragraph_format.line_spacing=1.05
                for run in p.runs:
                    run.font.name='Times New Roman';run.font.size=Pt(10.5)
                    if rownum==0:run.font.bold=True;run.font.color.rgb=RGBColor(255,255,255)
            if rownum==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'15395B');cell._tc.get_or_add_tcPr().append(shade)
        trpr=t.rows[rownum]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
        if rownum==0:trpr.append(OxmlElement('w:tblHeader'))
    doc.add_paragraph().paragraph_format.space_after=Pt(2)

lines=text.splitlines();i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        p=doc.add_paragraph();p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(8)
        for j,c in enumerate(code):
            run=p.add_run(c+('\n' if j<len(code)-1 else ''));run.font.name='Consolas';run.font.size=Pt(9)
        shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F0F4F8');p._p.get_or_add_pPr().append(shade)
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            vals=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',v) for v in vals):rows.append(vals)
            i+=1
        make_table(rows);continue
    elif line.startswith('!['):
        m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(DEST/m.group(2)),width=Cm(15.8))
        p.paragraph_format.keep_with_next=True
        p=doc.add_paragraph(m.group(1));p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:run.font.size=Pt(11);run.font.italic=True
    elif line.startswith('# '):doc.add_heading(line[2:],level=1)
    elif line.startswith('## '):
        if not line.startswith('## 4.1.'):
            doc.add_page_break()
        doc.add_heading(line[3:],level=2)
    elif line.startswith('### '):doc.add_heading(line[4:],level=3)
    else:
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;rich(p,line)
    i+=1
doc.save(DEST/'CHUONG_04_KNOWLEDGE_RAG.docx')
print('DOCX:',DEST/'CHUONG_04_KNOWLEDGE_RAG.docx')
print('Words:',len(text.split()),'Paragraphs:',len(doc.paragraphs),'Tables:',len(doc.tables),'Images:',len(doc.inline_shapes))
