---
base_model: canopylabs/3b-hi-ft-research_release
license: apache-2.0
language:
- hi   # Hindi
- bn   # Bengali
- mr   # Marathi
- te   # Telugu
- kn   # Kannada
- bho  # Bhojpuri
- mag  # Magahi
- hne  # Chhattisgarhi
- mai  # Maithili
- as   # Assamese
- brx  # Bodo
- doi  # Dogri
- gu   # Gujarati
- ml   # Malayalam
- pa   # Punjabi
- ta   # Tamil
- ne   # Nepali
- sa   # Sanskrit
- en   # English (Indian)
tags:
- text-to-speech
- speech-synthesis
- transformers
- multilingual
- indic
- orpheus
- lora
- low-latency
- gguf
- zero-shot
- emotions
- discrete-audio-tokens
task_categories:
- text-to-speech
pipeline_tag: text-to-speech
pretty_name: Svara-TTS v1
datasets:
- SYSPIN
- RASA
- IndicTTS
- SPICOR
---

# svara-TTS v1 — Open Multilingual TTS for India’s Voices

[![🤗 Hugging Face - svara-tts-v1 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-black)](https://huggingface.co/kenpath/svara-tts-v1) 
[![🤗 Hugging Face - Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/kenpath/svara-tts)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=flat&logo=github&logoColor=white)](https://github.com/Kenpath/svara-tts-inference)

**svara-TTS** is a developer-first multilingual TTS model for **19 languages** (18 Indic + Indian English).  
Built on an Orpheus-style discrete audio token approach, it targets **clarity, expressiveness, and low-latency** on commodity GPUs/CPUs.  
It supports light-weight **emotion/style control** (e.g., `<happy>`, `<sad>`, `<anger>`, `<fear>`) and simple **speaker identities** (`Language (Gender)`), with **zero-shot** adaptation paths.

---

## At a Glance

- **Languages (19):** Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi, Chhattisgarhi, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam, Punjabi, Tamil, Nepali, Sanskrit, Indian English.  
- **Expressivity:** End-of-utterance style tags; natural prosody; code-switch aware.  
- **Latency & Deployment:** Works well with **GGUF** exports; suitable for edge/CPU scenarios.  
- **Adaptability:** **LoRA-friendly** for quick speaker/domain specialization.

Try it live on the [**Demo Space**](https://huggingface.co/spaces/kenpath/svara-tts), or on **[Colab](https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg)**
Deployment scripts and **inference repo** will be available soon. Watch our [Github](https://github.com/Kenpath/svara-tts-inference) for updates

---

## Prompting (Orpheus-style)

- Place style/emotion tags **at the end** of the sentence:  
  `आज... सच में अच्छी खबर है — शाम को मिलते हैं! <happy>`
- Use punctuation to hint prosody (ellipses, commas, exclamation).
- For technical or dense text, end with `<clear>` to prioritize intelligibility.

> Speaker IDs follow a simple convention: **`Language (Gender)`** (e.g., `Marathi (Male)`).

---

## Training Data Summary

Trained on **2000+ hours** of open, high-quality speech from **SYSPIN**, **RASA**, **IndicTTS**, and **SPICOR**, covering **~50 speakers** (balanced male/female) across **19 languages**.  
Data was curated to encourage natural prosody, broad coverage, and stable multilingual transfer. See **Acknowledgments** for provenance.

---

## Intended Uses

- Multilingual assistants, IVR, learning apps, reading aids, accessibility tools  
- Content localization (education, public-information, civic services)  
- Research on Indic prosody, emotion control, cross-lingual transfer

## Out-of-Scope / Not Intended

- Impersonation of private individuals or public figures without consent  
- Deceptive content (fraud, harassment, misinformation)  
- Safety-critical deployments without human oversight

---

## Limitations

- **Proper nouns & rare entities:** may require spelling hints or `<clear>`.  
- **Very long sentences:** chunk or add punctuation for natural prosody.  
- **Emotion strength:** varies by language due to data density.  
- **Code-mixing:** common patterns work; it’s not a deterministic rules engine.

Many of these improve with targeted LoRA finetuning and better preprocessing.

---

## Responsible Use

By using this model, you agree to follow applicable laws and ethical guidelines.  
Avoid impersonation, harassment, targeted deception, or other harmful uses.  
Where appropriate, disclose synthetic speech to end users.

---

## Sources & Links

- **Model:** https://huggingface.co/kenpath/svara-tts-v1  
- **Demo Space:** https://huggingface.co/spaces/kenpath/svara-tts  
- **Inference repo:** https://github.com/Kenpath/svara-tts-inference  
- **Colab:** https://colab.research.google.com/drive/15YxFo1DzdQNbFUIZ1HJA4AN4oHqKxGtg

---

## 🙏 Acknowledgments

This work was developed by **[Kenpath Technologies](https://kenpath.ai/)** for the open-source community. We also thank RunPod for the startup credits that supported our GPU compute.

- **Canopy Labs — Orpheus:** foundational ideas & open release  
  Release: https://canopylabs.ai/releases/orpheus_can_speak_any_language  
- **SPIRE Lab, IISc Bangalore** — **SYSPIN** (multilingual studio) and **SPICOR** (Indian English)  
- **AI4Bharat** — **RASA** expressive speech  
- **IIT Madras** — **IndicTTS**  
- **Unsloth** — helpful notes & tooling  
- **RunPod** — startup GPU credits that accelerated experiments

---

## License

**Apache-2.0**

---

## Versioning & Changelog

- **v1.0.0:** Initial public release (19 languages)