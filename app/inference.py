import torch
import numpy as np
import librosa
import soundfile as sf
import os
import argparse
from model import Generator

def load_model(model_path, device):
    """Load the trained generator model"""
    G = Generator().to(device)
    
    if os.path.exists(model_path):
        try:
            # Try loading as full checkpoint first
            checkpoint = torch.load(model_path, map_location=device)
            if 'generator_state_dict' in checkpoint:
                G.load_state_dict(checkpoint['generator_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                # Load as state dict only
                G.load_state_dict(checkpoint)
                print("Loaded model state dict")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file not found: {model_path}")
        return None
    
    G.eval()
    return G

def preprocess_audio_for_inference(audio_path, target_length=None):
    """Preprocess audio file to mel spectrogram"""
    # Parameters (should match training preprocessing)
    SAMPLE_RATE = 16000
    N_MELS = 80
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    
    try:
        # Load and normalize audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
        y, _ = librosa.effects.trim(y)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y, sr=SAMPLE_RATE, n_fft=WIN_LENGTH, 
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_spec = np.log1p(mel_spec)
        
        return mel_spec
    
    except Exception as e:
        print(f"Error preprocessing audio {audio_path}: {e}")
        return None

def convert_voice(model_path, source_audio_path, output_path, target_speaker_id=1, device="cuda"):
    """Convert voice from source to target speaker"""
    
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    G = load_model(model_path, device)
    if G is None:
        return False
    
    # Preprocess source audio
    source_mel = preprocess_audio_for_inference(source_audio_path)
    if source_mel is None:
        return False
    
    # Convert to tensor and add batch dimension
    source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    target_id = torch.tensor([target_speaker_id], dtype=torch.long).to(device)
    
    print(f"Source mel shape: {source_mel.shape}")
    
    # Perform conversion
    with torch.no_grad():
        converted_mel = G(source_mel, target_id)
    
    # Convert back to numpy
    converted_mel = converted_mel.squeeze(0).cpu().numpy()
    
    # Save the mel spectrogram
    mel_output_path = output_path.replace('.wav', '_mel.npy')
    np.save(mel_output_path, converted_mel)
    print(f"Converted mel spectrogram saved to: {mel_output_path}")
    
    # Placeholder: Convert mel to audio using Griffin-Lim
    try:
        mel_db = converted_mel
        mel_linear = np.expm1(mel_db)  # Inverse of log1p
        audio_reconstructed = librosa.feature.inverse.mel_to_audio(
            mel_linear, sr=16000, n_fft=1024, hop_length=256
        )
        sf.write(output_path, audio_reconstructed, 16000)
        print(f"Converted audio saved to: {output_path}")
        print("Note: Using Griffin-Lim reconstruction. For better quality, use a neural vocoder.")
    except Exception as e:
        print(f"Error converting mel to audio: {e}")
        print("Mel spectrogram saved successfully, but audio conversion failed.")
    
    return True

def batch_convert(model_path, input_dir, output_dir, target_speaker_id=1):
    """Convert all audio files in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.flac')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"converted_{filename}")
            
            print(f"Converting: {filename}")
            success = convert_voice(model_path, input_path, output_path, target_speaker_id)
            
            if success:
                print(f"✓ Successfully converted: {filename}")
            else:
                print(f"✗ Failed to convert: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice conversion with StarGAN-VC")
    parser.add_argument("--model_path", default="/workspace/checkpoints/stargan_vc_epoch_100.pth", 
                        help="Path to trained model or checkpoint")
    parser.add_argument("--source_file", default="/workspace/processed/source/Queen-bohemian.npy", 
                        help="Source audio or .npy file")
    parser.add_argument("--output_file", default="/workspace/converted_output.wav", 
                        help="Output audio file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.source_file.endswith('.npy'):
        try:
            G = load_model(args.model_path, device)
            if G is not None:
                source_mel = np.load(args.source_file)
                source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
                target_id = torch.tensor([1], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    converted = G(source_mel, target_id)
                
                converted = converted.squeeze(0).cpu().numpy()
                np.save("/workspace/output/converted_mel.npy", converted)
                print("Conversion completed. Output saved as /workspace/output/converted_mel.npy")
                print("Use a vocoder to convert mel spectrogram to audio.")
        except Exception as e:
            print(f"Error: {e}")
    
    elif os.path.exists(args.source_file) and args.source_file.endswith('.wav'):
        convert_voice(args.model_path, args.source_file, args.output_file, target_speaker_id=1)
    else:
        print("Please provide a valid .wav or .npy file for conversion.")