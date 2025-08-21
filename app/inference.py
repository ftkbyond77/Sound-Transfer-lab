import os
import numpy as np
from app import preprocess, model

SOURCE_FILE = "/workspace/data/source_voice/Queen-bohemian.wav"
OUTPUT_FILE = "/workspace/data/processed/Spieluhr.wav"

# Preprocess
y = preprocess.preprocess_audio(SOURCE_FILE)
mel = preprocess.extract_mel_spectrogram(y)

# Load model
vc_model = model.load_model("app/checkpoint.pth")

# Convert voice
converted_mel = model.convert(vc_model, mel, target_speaker_id=1)

# Vocoder: mel -> waveform
wav = model.vocoder_inference(converted_mel)

# Save
import soundfile as sf
sf.write(OUTPUT_FILE, wav, 16000)
print(f"Converted voice saved at {OUTPUT_FILE}")
