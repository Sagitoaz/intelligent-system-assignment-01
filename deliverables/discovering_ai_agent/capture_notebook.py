from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent
nb = nbformat.read(root / 'agent01_thuc_hanh.ipynb', as_version=4)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
exporter = HTMLExporter(template_name='lab')
html, _ = exporter.from_notebook_node(nb)
css = '''<style>
body {background:white !important; --jp-code-font-size:18px;}
main {max-width:980px; margin:auto; padding:8px;}
.jp-Cell {padding:14px 10px !important; margin-bottom:18px !important;}
.jp-CodeCell {border:1px solid #cbd5e1; border-radius:6px; background:#fff;}
.jp-CodeCell pre {font-size:18px !important; line-height:1.28 !important;
  white-space:pre-wrap !important; overflow-wrap:anywhere !important;}
.jp-InputPrompt,.jp-OutputPrompt {font-size:13px !important; min-width:65px !important;}
.jp-OutputArea-output {overflow:visible !important; max-height:none !important;}
.jp-OutputArea img {max-width:100% !important;}
</style>'''
html = html.replace('</head>', css + '</head>')
(root / 'agent01_thuc_hanh.html').write_text(html, encoding='utf-8')
images = root / 'screenshots'
images.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', headless=True)
    page = browser.new_page(viewport={"width":1040, "height":900}, device_scale_factor=1.6)
    page.goto((root / 'agent01_thuc_hanh.html').as_uri(), wait_until='load')
    page.evaluate('document.fonts.ready')
    locators = page.locator('.jp-CodeCell')
    for i in range(locators.count()):
        locators.nth(i).screenshot(path=str(images / f'cell_{i+1:02d}.png'))
    print('Screenshots:', locators.count())
    browser.close()
