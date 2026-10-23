from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
nb=nbformat.read(ROOT/'agent01_notebook_mo_rong.ipynb',as_version=4)
codes=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in codes)
assert not any(o.output_type=='error' for c in codes for o in c.outputs)
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
css="""<style>
body{background:white!important;--jp-code-font-size:18px}
main{max-width:980px;margin:auto;padding:8px}
.jp-Cell{padding:12px 8px!important;margin-bottom:18px!important}
.jp-CodeCell{border:1px solid #cbd5e1;border-radius:6px;background:white}
.jp-CodeCell pre{font-size:18px!important;line-height:1.25!important;white-space:pre-wrap!important;overflow-wrap:anywhere!important}
.jp-InputPrompt,.jp-OutputPrompt{font-size:13px!important;min-width:55px!important}
.jp-OutputArea-output{overflow:visible!important;max-height:none!important}
.jp-RenderedHTMLCommon table{font-size:16px!important}
.jp-OutputArea img{max-width:100%!important}
</style>"""
html=html.replace('</head>',css+'</head>')
out=ROOT/'agent01_notebook_mo_rong.html'
out.write_text(html,encoding='utf-8')
shots=ROOT/'screenshots';shots.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1040,'height':1000},device_scale_factor=1.6)
    page.goto(out.as_uri(),wait_until='load')
    page.evaluate('document.fonts.ready')
    loc=page.locator('.jp-CodeCell')
    for i in range(loc.count()):
        loc.nth(i).screenshot(path=str(shots/f'cell_{i+1:02d}.png'))
    print('Screenshots:',loc.count())
    browser.close()
