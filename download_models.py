"""Download official pretrained assets to this workspace, never contest samples."""
import argparse
import hashlib
import urllib.request
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

MODELS = {
    'bert': ('google-bert/bert-base-uncased', ['config.json', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']),
    'aligner': ('facebook/wav2vec2-base-960h', ['config.json', 'vocab.json', 'preprocessor_config.json', 'tokenizer_config.json', 'pytorch_model.bin'])}


def download(which):
    repo, files = MODELS[which]
    target = ROOT / 'pretrained' / which
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / 'download_manifest.json'
    expected = {row['file']: row for row in json.loads(manifest_path.read_text(encoding='utf-8'))} if manifest_path.exists() else {}
    manifest = []
    for name in files:
        dest = target / name
        url = f'https://huggingface.co/{repo}/resolve/main/{name}'
        if not dest.exists():
            print(f'Downloading {repo}/{name}', flush=True)
            tmp = dest.with_suffix(dest.suffix + '.part')
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as output:
                while block := response.read(1024 * 1024): output.write(block)
            tmp.replace(dest)
        with dest.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if name in expected and digest != expected[name]['sha256']:
            raise ValueError(f'{name}: SHA256 differs from the supplied manifest. Obtain the original model version before continuing.')
        manifest.append({'file': name, 'source': url, 'sha256': digest, 'bytes': dest.stat().st_size})
    save_json(target / 'download_manifest.json', manifest)
    print(target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', default=['bert'], choices=['bert', 'aligner'])
    args = parser.parse_args()
    for name in args.models: download(name)
