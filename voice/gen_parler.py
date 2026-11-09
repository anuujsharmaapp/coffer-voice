"""Indic Parler-TTS (ai4bharat/indic-parler-tts, Apache-2.0, gated: needs HF_TOKEN) on a GitHub runner (CPU).
usage: python voice/gen_parler.py <speaker name e.g. Rohit> <outdir>; reads voice/parler_jobs.json {lines:[{id,text}], descriptions:[...{name}...]}.
Code follows the model card's own example (description tokenizer = the text encoder's tokenizer)."""
import json, os, sys, time
import soundfile as sf, torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

def main():
    speaker, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    jobs = json.load(open('voice/parler_jobs.json'))
    t0 = time.time()
    model = ParlerTTSForConditionalGeneration.from_pretrained('ai4bharat/indic-parler-tts').eval()
    tok = AutoTokenizer.from_pretrained('ai4bharat/indic-parler-tts')
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    print(f'loaded in {time.time() - t0:.0f}s', flush=True)
    report = []
    for line in jobs['lines']:
        for di, desc in enumerate(jobs['descriptions']):
            torch.manual_seed(1234 + di)
            d = dtok(desc.format(name=speaker), return_tensors='pt'); p = tok(line['text'], return_tensors='pt')
            t1 = time.time()
            with torch.no_grad():
                g = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            a = g.cpu().numpy().squeeze()
            fn = f"{out}/parler_{speaker.lower()}_{line['id']}_d{di}.wav"
            sf.write(fn, a, model.config.sampling_rate)
            r = {'file': os.path.basename(fn), 'secs': round(time.time() - t1, 1), 'audio_s': round(len(a) / model.config.sampling_rate, 2)}
            report.append(r); print(json.dumps(r), flush=True)
    json.dump(report, open(f'{out}/parler_{speaker.lower()}_report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
