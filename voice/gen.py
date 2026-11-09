"""Generate test voice clips with open Indian TTS models on a GitHub runner (CPU).
usage: python voice/gen.py <model: veena|svara> <speaker> <outdir>
Reads voice/jobs.json: {"lines": [{"id": ..., "text": ...}], ...}. Prompt formats follow each model's own published code:
- Veena (maya-research/Veena, Apache-2.0): [SOH] "<spk_{speaker}> {text}" [EOH][SOA][SOS] (model card example)
- Svara (kenpath/svara-tts-v1, Apache-2.0): [BOS][SOH][AUDIO] "{speaker}: {text}" [EOH][EOT][SOA][SOS] (svara-tts-inference encoder.py)
Audio tokens -> SNAC 24 kHz (hubertsiuzdak/snac_24khz), 7 tokens per frame with per-position offsets 128266 + k*4096."""
import json, os, sys, time
import numpy as np, soundfile as sf, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

SOS, EOS_SPEECH, SOH, EOH, SOA, EOA = 128257, 128258, 128259, 128260, 128261, 128262
BOS, EOT, AUDIO = 128000, 128009, 156939
BASE = 128266
REPO = {'veena': 'maya-research/Veena', 'svara': 'kenpath/svara-tts-v1'}
SAMPLING = {'veena': dict(temperature=0.4, top_p=0.9, repetition_penalty=1.05),
            'svara': dict(temperature=0.6, top_p=0.95, repetition_penalty=1.1)}

def prompt_ids(tok, model, speaker, text):
    if model == 'veena':
        return [SOH, *tok.encode(f'<spk_{speaker}> {text}', add_special_tokens=False), EOH, SOA, SOS]
    return [BOS, SOH, AUDIO, *tok.encode(f'{speaker}: {text}', add_special_tokens=False), EOH, EOT, SOA, SOS]

def decode(snac, ids):
    ids = [t for t in ids if BASE <= t < BASE + 7 * 4096]
    ids = ids[: len(ids) // 7 * 7]
    if not ids: return None
    l0, l1, l2 = [], [], []
    for i in range(0, len(ids), 7):
        c = [ids[i + k] - BASE - k * 4096 for k in range(7)]
        l0.append(c[0]); l1 += [c[1], c[4]]; l2 += [c[2], c[3], c[5], c[6]]
    codes = [torch.tensor(x, dtype=torch.int32).clamp(0, 4095).unsqueeze(0) for x in (l0, l1, l2)]
    with torch.no_grad():
        return snac.decode(codes).squeeze().clamp(-1, 1).numpy()

def main():
    model_name, speaker, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    jobs = json.load(open('voice/jobs.json'))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(REPO[model_name])
    lm = AutoModelForCausalLM.from_pretrained(REPO[model_name], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    snac = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval()
    print(f'loaded in {time.time() - t0:.0f}s', flush=True)
    report = []
    for line in jobs['lines']:
        ids = prompt_ids(tok, model_name, speaker, line['text'])
        x = torch.tensor([ids])
        t1 = time.time()
        with torch.no_grad():
            y = lm.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=int(line.get('max_tokens', 900)), do_sample=True,
                            eos_token_id=[EOS_SPEECH, EOA], pad_token_id=tok.pad_token_id or 128263, **SAMPLING[model_name])
        gen = y[0][len(ids):].tolist()
        dt = time.time() - t1
        audio = decode(snac, gen)
        slug = speaker.replace(' ', '').replace('(', '_').replace(')', '').lower()
        fn = f"{out}/{model_name}_{slug}_{line['id']}.wav"
        if audio is not None: sf.write(fn, audio, 24000)
        r = {'file': os.path.basename(fn), 'tokens': len(gen), 'secs': round(dt, 1), 'tok_per_s': round(len(gen) / max(dt, 1e-6), 2),
             'audio_s': round(len(audio) / 24000, 2) if audio is not None else 0, 'ended': bool(gen and gen[-1] in (EOS_SPEECH, EOA))}
        report.append(r); print(json.dumps(r), flush=True)
    json.dump(report, open(f'{out}/{model_name}_{speaker.replace(" ", "").lower()}_report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
