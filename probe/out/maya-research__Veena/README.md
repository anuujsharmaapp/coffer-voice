---
license: apache-2.0
language:
- en
- hi
library_name: transformers
tags:
- text-to-speech
- tts
- hindi
- english
- llama
- audio
- speech
- india
datasets:
- proprietary
pipeline_tag: text-to-speech
co2_eq_emissions:
  emissions: 0
  source: "Not specified"
  training_type: "unknown"
  geographical_location: "unknown"
---

# Veena - Text to Speech for Indian Languages

Veena is a state-of-the-art neural text-to-speech (TTS) model developed by Maya Research, designed for English and Indian languages. Built on a Llama architecture backbone, Veena generates natural, expressive speech with emotional tone, remarkable quality, and ultra-low latency. It represents the foundation of our voice intelligence work, bringing human-like voice to the two most spoken languages in the world.
## Model Overview

**Veena** is a 3B parameter autoregressive transformer model based on the Llama architecture. It is designed to synthesize high-quality speech from text in Hindi and English, including code-mixed scenarios. The model outputs audio at a 24kHz sampling rate using the SNAC neural codec.

* **Model type:** Autoregressive Transformer
* **Base Architecture:** Llama (3B parameters)
* **Languages:** Hindi, English
* **Audio Codec:** SNAC @ 24kHz
* **License:** Apache 2.0
* **Developed by:** Maya Research
* **Model URL:** [https://huggingface.co/maya-research/veena](https://huggingface.co/maya-research/veena)

## Key Features

* **4 Distinct Voices:** `kavya`, `agastya`, `maitri`, and `vinaya` - each with unique vocal characteristics.
* **Multilingual Support:** Native Hindi and English capabilities with code-mixed support.
* **Ultra-Fast Inference:** Sub-80ms latency on H100-80GB GPUs.
* **High-Quality Audio:** 24kHz output with the SNAC neural codec.
* **Production-Ready:** Optimized for real-world deployment with 4-bit quantization support.

## How to Get Started with the Model

### Installation

To use Veena, you need to install the `transformers`, `torch`, `torchaudio`, `snac`, and `bitsandbytes` libraries.

```bash
pip install transformers torch torchaudio
pip install snac bitsandbytes  # For audio decoding and quantization
```

### Basic Usage

The following Python code demonstrates how to generate speech from text using Veena with 4-bit quantization for efficient inference.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
import soundfile as sf

# Model configuration for 4-bit inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "maya-research/veena-tts",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)

# Initialize SNAC decoder
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

# Available speakers
speakers = ["kavya", "agastya", "maitri", "vinaya"]

def generate_speech(text, speaker="kavya", temperature=0.4, top_p=0.9):
    """Generate speech from text using specified speaker voice"""

    # Prepare input with speaker token
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Construct full sequence: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH]
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]

    input_ids = torch.tensor([input_tokens], device=model.device)

    # Calculate max tokens based on text length
    max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

    # Generate audio tokens
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )

    # Extract SNAC tokens
    generated_ids = output[0][len(input_tokens):].tolist()
    snac_tokens = [
        token_id for token_id in generated_ids
        if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
    ]

    if not snac_tokens:
        raise ValueError("No audio tokens generated")

    # Decode audio
    audio = decode_snac_tokens(snac_tokens, snac_model)
    return audio

def decode_snac_tokens(snac_tokens, snac_model):
    """De-interleave and decode SNAC tokens to audio"""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    # Get the device of the SNAC model. Fixed by Shresth to run on colab notebook :)
    snac_device = next(snac_model.parameters()).device

    # De-interleave tokens into 3 hierarchical levels
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        # Level 0: Coarse (1 token)
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        # Level 1: Medium (2 tokens)
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        # Level 2: Fine (4 tokens)
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

    # Convert to tensors for SNAC decoder
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)

    # Decode with SNAC
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)

    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()

# --- Example Usage ---

# Hindi
text_hindi = "आज मैंने एक नई तकनीक के बारे में सीखा जो कृत्रिम बुद्धिमत्ता का उपयोग करके मानव जैसी आवाज़ उत्पन्न कर सकती है।"
audio = generate_speech(text_hindi, speaker="kavya")
sf.write("output_hindi_kavya.wav", audio, 24000)

# English
text_english = "Today I learned about a new technology that uses artificial intelligence to generate human-like voices."
audio = generate_speech(text_english, speaker="agastya")
sf.write("output_english_agastya.wav", audio, 24000)

# Code-mixed
text_mixed = "मैं तो पूरा presentation prepare कर चुका हूं! कल रात को ही मैंने पूरा code base चेक किया।"
audio = generate_speech(text_mixed, speaker="maitri")
sf.write("output_mixed_maitri.wav", audio, 24000)
```

## Uses

Veena is ideal for a wide range of applications requiring high-quality, low-latency speech synthesis for Indian languages, including:

  * **Accessibility:** Screen readers and voice-enabled assistance for visually impaired users.
  * **Customer Service:** IVR systems, voice bots, and automated announcements.
  * **Content Creation:** Dubbing for videos, e-learning materials, and audiobooks.
  * **Automotive:** In-car navigation and infotainment systems.
  * **Edge Devices:** Voice-enabled smart devices and IoT applications.

## Technical Specifications

### Architecture

Veena leverages a 3B parameter transformer-based architecture with several key innovations:

  * **Base Architecture:** Llama-style autoregressive transformer (3B parameters)
  * **Audio Codec:** SNAC (24kHz) for high-quality audio token generation
  * **Speaker Conditioning:** Special speaker tokens (`<spk_kavya>`, `<spk_agastya>`, `<spk_maitri>`, `<spk_vinaya>`)
  * **Parameter-Efficient Training:** LoRA adaptation with differentiated ranks for attention and FFN modules.
  * **Context Length:** 2048 tokens

### Training

#### Training Infrastructure

  * **Hardware:** 8× NVIDIA H100 80GB GPUs
  * **Distributed Training:** DDP with optimized communication
  * **Precision:** BF16 mixed precision training with gradient checkpointing
  * **Memory Optimization:** 4-bit quantization with NF4 + double quantization

#### Training Configuration

  * **LoRA Configuration:**
      * `lora_rank_attention`: 192
      * `lora_rank_ffn`: 96
      * `lora_alpha`: 2× rank (384 for attention, 192 for FFN)
      * `lora_dropout`: 0.05
      * `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
      * `modules_to_save`: `["embed_tokens"]`
  * **Optimizer Configuration:**
      * `optimizer`: AdamW (8-bit)
      * `optimizer_betas`: (0.9, 0.98)
      * `optimizer_eps`: 1e-5
      * `learning_rate_peak`: 1e-4
      * `lr_scheduler`: cosine
      * `warmup_ratio`: 0.02
  * **Batch Configuration:**
      * `micro_batch_size`: 8
      * `gradient_accumulation_steps`: 4
      * `effective_batch_size`: 256

#### Training Data

Veena was trained on **proprietary, high-quality datasets** specifically curated for Indian language TTS.

  * **Data Volume:** 15,000+ utterances per speaker (60,000+ total)
  * **Languages:** Native Hindi and English utterances with code-mixed support
  * **Speaker Diversity:** 4 professional voice artists with distinct characteristics
  * **Audio Quality:** Studio-grade recordings at 24kHz sampling rate
  * **Content Diversity:** Conversational, narrative, expressive, and informational styles

**Note:** The training datasets are proprietary and not publicly available.

## Performance Benchmarks

| Metric                | Value                     |
| --------------------- | ------------------------- |
| Latency (H100-80GB)   | \<80ms                     |
| Latency (A100-40GB)   | \~120ms                    |
| Latency (RTX 4090)    | \~200ms                    |
| Real-time Factor      | 0.05x                     |
| Throughput            | \~170k tokens/s (8×H100)   |
| Audio Quality (MOS)   | 4.2/5.0                   |
| Speaker Similarity    | 92%                       |
| Intelligibility       | 98%                       |

## Risks, Limitations and Biases

  * **Language Support:** Currently supports only Hindi and English. Performance on other Indian languages is not guaranteed.
  * **Speaker Diversity:** Limited to 4 speaker voices, which may not represent the full diversity of Indian accents and dialects.
  * **Hardware Requirements:** Requires a GPU for real-time or near-real-time inference. CPU performance will be significantly slower.
  * **Input Length:** The model is limited to a maximum input length of 2048 tokens.
  * **Bias:** The model's performance and voice characteristics are a reflection of the proprietary training data. It may exhibit biases present in the data.

## Future Updates

We are actively working on expanding Veena's capabilities:

  * Support for Tamil, Telugu, Bengali, Marathi, and other Indian languages.
  * Additional speaker voices with regional accents.
  * Emotion and prosody control tokens.
  * Streaming inference support.
  * CPU optimization for edge deployment.

## Citing

If you use Veena in your research or applications, please cite:

```bibtex
@misc{veena2025,
  title={Veena: Open Source Text-to-Speech for Indian Languages},
  author={Maya Research Team},
  year={2025},
  publisher={HuggingFace},
  url={[https://huggingface.co/maya-research/veena-tts](https://huggingface.co/maya-research/veena-tts)}
}
```

## Acknowledgments

We thank the open-source community and all contributors who made this project possible. Special thanks to the voice artists who provided high-quality recordings for training.