"""Reject incompatible resume rather than relabel old experiments as new ones."""
import hashlib
import json


def make_manifest(paths,environment):
    hashes={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    content={'files':hashes,'environment':environment}
    fingerprint=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    return {**content,'fingerprint':fingerprint}


def ensure_manifest(folder,manifest):
    target=folder/'run_manifest.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf8'))
        if previous['fingerprint']!=manifest['fingerprint']:
            raise ValueError('incompatible experiment resume: preserve old results and use a new output directory')
    else:
        target.write_text(json.dumps(manifest,indent=2),encoding='utf8')


def experiment_manifest(root,environment):
    paths=list((root/'data/tieuluan/processed').glob('*.npz'))
    paths += list((root/'src/tieuluan/scratch').glob('*.py'))
    paths += [root/'src/tieuluan/config.py',root/'src/tieuluan/experiment_data.py',root/'src/tieuluan/experiment_models.py',
              root/'scripts/tieuluan/run_experiments.py']
    return make_manifest(paths,environment)


def check_raw_snapshot(root):
    inventory=json.loads((root/'data/tieuluan/processed/inventory.json').read_text(encoding='utf8'))
    raw=root/'data/tieuluan/raw'
    for key,record in inventory.items():
        if key in ('sp500','vnindex','btc'):path=raw/'indices'/f'{key}.csv'
        elif key=='taiwan_bankruptcy':path=raw/'uci'/key/'data.csv'
        else:path=next((raw/'uci'/key).glob('*.xls'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'raw snapshot changed: {key}; prepare datasets again and use a new output label')
