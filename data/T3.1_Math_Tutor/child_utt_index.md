# Child Utterance Sample — Manifest

The full child speech corpus is **not bundled here** to keep the seed folder small.
Use the public datasets below and the included `download_child_audio.py` script.

## Public sources

| Dataset | URL | Use |
|---|---|---|
| Mozilla Common Voice (en, fr, rw) — child age band | https://commonvoice.mozilla.org/datasets | Child / teen utterances filtered by age tag |
| DigitalUmuganda Kinyarwanda speech corpus | https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda_speech_8khz | Adult Kinyarwanda; pitch-shift for child-like prosody |
| Children's Mind synthetic numeracy dataset | (generate with `make_synthetic_child.py`) | TTS-synthesised number words 1–20 in EN/FR/KIN |

## Sample manifest (10 utterances, included for shape reference)

See `child_utt_sample_seed.csv` — schema: utt_id, audio_path, transcript, language, correctness.
