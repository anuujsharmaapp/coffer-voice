"""Probe: list files and fetch the small text files (model cards, configs, example code) of candidate Indian TTS models,
so the generation step can follow each model's own instructions exactly."""
import os, json
from huggingface_hub import HfApi, hf_hub_download
REPOS = ['maya-research/Veena', 'kenpath/svara-tts-v1', 'ai4bharat/indic-parler-tts']
api = HfApi(); out = 'probe/out'; os.makedirs(out, exist_ok=True)
summary = {}
for r in REPOS:
    d = os.path.join(out, r.replace('/', '__')); os.makedirs(d, exist_ok=True)
    try:
        info = api.model_info(r, files_metadata=True)
        files = [(s.rfilename, s.size) for s in info.siblings]
        summary[r] = {'gated': getattr(info, 'gated', None), 'license': (info.card_data or {}).get('license') if info.card_data else None,
                      'files': files}
        for f, size in files:
            if (size or 0) < 2_000_000 and f.split('.')[-1] in ('md', 'json', 'py', 'txt', 'yaml', 'yml', 'jinja', 'model') :
                try:
                    p = hf_hub_download(r, f, local_dir=d)
                except Exception as e:
                    summary[r].setdefault('errors', []).append(f'{f}: {e.__class__.__name__}: {str(e)[:200]}')
    except Exception as e:
        summary[r] = {'error': f'{e.__class__.__name__}: {str(e)[:300]}'}
json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str)[:4000])
