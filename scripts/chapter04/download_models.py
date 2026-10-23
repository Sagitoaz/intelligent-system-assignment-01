"""Download exact public model revisions into the project, without API keys."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time
import requests
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'models/chapter04'
BASE.mkdir(parents=True,exist_ok=True)
REPOS={'embedding':'sentence-transformers/all-MiniLM-L6-v2','generator':'google/flan-t5-small'}
def fetch(role,repo):
    target=BASE/role;target.mkdir(exist_ok=True)
    metadata=requests.get(f'https://huggingface.co/api/models/{repo}',timeout=60)
    metadata.raise_for_status();meta=metadata.json();revision=meta['sha']
    existing=target/'manifest.json'
    if existing.exists():revision=json.loads(existing.read_text())['revision']
    names={x['rfilename'] for x in meta['siblings']}
    desired=['config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','spiece.model','vocab.txt','generation_config.json','model.safetensors','README.md']
    records=[]
    for name in desired:
        if name not in names:continue
        path=target/name;url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        if not path.exists():
            for attempt in range(3):
                try:
                    part=path.with_suffix(path.suffix+'.partial')
                    with requests.get(url,stream=True,timeout=(60,180)) as response:
                        response.raise_for_status();count=0;last=0
                        with part.open('wb') as f:
                            for block in response.iter_content(1024*1024):
                                f.write(block);count+=len(block)
                                if count-last>=32*1024*1024:
                                    print(f'{role}/{name}: {count/1e6:.1f} MB',flush=True);last=count
                    part.replace(path);break
                except requests.RequestException:
                    if attempt==2:raise
                    time.sleep(2)
        hasher=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):hasher.update(block)
        records.append({'file':name,'url':url,'bytes':path.stat().st_size,'sha256':hasher.hexdigest()})
        print('Saved',role,name,path.stat().st_size,flush=True)
    assert (target/'model.safetensors').exists()
    result={'repo':repo,'revision':revision,'files':records,'accessed':'2026-10-13'}
    existing.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(lambda kv:fetch(*kv),REPOS.items()))
    (BASE/'model_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('All models downloaded.',flush=True)
