import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
import os
import argparse
from pathlib import Path

def mel_to_linear_spectrogram(mel_spec, n_fft=1024, n_mels=80, sample_rate=16000):
    """Convert mel spectrogram back to linear spectrogram"""
    # Create mel filter bank
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    
    # Convert mel to linear using pseudo-inverse
    # This is an approximation since mel->linear conversion is not perfect
    mel_basis_inv = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(mel_basis_inv, mel_spec)
    
    # Ensure positive values
    linear_spec = np.maximum(linear_spec, 0.0)
    
    return linear_spec

def npy_to_wav_hifigan(npy_path, output_wav_path, sample_rate=16000, use_gpu=True):
    """Convert numpy mel spectrogram to WAV using HiFiGAN vocoder"""
    try:
        # Load mel spectrogram
        mel_spec = np.load(npy_path)
        print(f"Loaded mel spectrogram shape: {mel_spec.shape}")
        
        # Ensure mel_spec is in correct format [n_mels, frames]
        if len(mel_spec.shape) == 3:
            # Remove batch dimension if present [1, n_mels, frames] -> [n_mels, frames]
            mel_spec = mel_spec.squeeze(0)
        
        # Convert to log mel if not already (assuming your model outputs log1p values)
        # If your mel values are in log1p format, convert back to log mel for vocoder
        if np.min(mel_spec) >= 0:  # Likely log1p format
            mel_spec = np.log(np.expm1(mel_spec) + 1e-8)  # Convert log1p to log
        
        # Ensure reasonable mel values (HiFiGAN expects log mel around -11.5 to 2.3)
        mel_spec = np.clip(mel_spec, -11.5, 2.3)
        
        # Convert to tensor and add batch dimension
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
        
        # Setup device
        device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        mel_tensor = mel_tensor.to(device)
        print(f"Using device: {device}")
        print(f"Mel tensor shape for vocoder: {mel_tensor.shape}")
        
        # Load HiFiGAN vocoder
        bundle = torchaudio.pipelines.HIFIGAN_VOCODER_V3_16KHZ
        vocoder = bundle.get_vocoder().to(device)
        vocoder.eval()
        
        # Generate audio
        with torch.no_grad():
            audio = vocoder(mel_tensor)
        
        # Convert to numpy and ensure correct shape
        audio = audio.squeeze().cpu().numpy()
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
        
        # Save audio
        sf.write(output_wav_path, audio, sample_rate)
        print(f"✓ Audio saved to: {output_wav_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error with HiFiGAN conversion: {e}")
        return False

def npy_to_wav_griffin_lim(npy_path, output_wav_path, sample_rate=16000, n_fft=1024, hop_length=256):
    """Convert numpy mel spectrogram to WAV using Griffin-Lim algorithm (fallback method)"""
    try:
        # Load mel spectrogram
        mel_spec = np.load(npy_path)
        print(f"Loaded mel spectrogram shape: {mel_spec.shape}")
        
        # Ensure correct format
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.squeeze(0)
        
        # Convert log1p back to linear scale if needed
        if np.min(mel_spec) >= 0:  # Likely log1p format
            mel_linear = np.expm1(mel_spec)
        else:
            mel_linear = np.exp(mel_spec)
        
        # Convert mel to audio using Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_linear, 
            sr=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length,
            n_iter=60  # More iterations for better quality
        )
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Create output directory
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
        
        # Save audio
        sf.write(output_wav_path, audio, sample_rate)
        print(f"✓ Audio saved to: {output_wav_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error with Griffin-Lim conversion: {e}")
        return False

def convert_npy_to_wav(npy_path, output_wav_path, method="hifigan", sample_rate=16000):
    """
    Convert .npy mel spectrogram file to .wav audio file
    
    Args:
        npy_path: Path to input .npy file containing mel spectrogram
        output_wav_path: Path for output .wav file
        method: "hifigan" or "griffin_lim"
        sample_rate: Output sample rate
    """
    if not os.path.exists(npy_path):
        print(f"✗ Input file not found: {npy_path}")
        return False
    
    print(f"Converting {npy_path} to {output_wav_path} using {method}...")
    
    if method.lower() == "hifigan":
        success = npy_to_wav_hifigan(npy_path, output_wav_path, sample_rate)
        if not success:
            print("HiFiGAN failed, trying Griffin-Lim as fallback...")
            success = npy_to_wav_griffin_lim(npy_path, output_wav_path, sample_rate)
    else:
        success = npy_to_wav_griffin_lim(npy_path, output_wav_path, sample_rate)
    
    return success

def batch_convert_npy_to_wav(input_dir, output_dir, method="hifigan"):
    """Convert all .npy files in a directory to .wav files"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(input_path.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files to convert...")
    
    for npy_file in npy_files:
        output_file = output_path / f"{npy_file.stem}.wav"
        print(f"\nProcessing: {npy_file.name}")
        
        success = convert_npy_to_wav(str(npy_file), str(output_file), method)
        if success:
            print(f"✓ Converted: {npy_file.name} -> {output_file.name}")
        else:
            print(f"✗ Failed: {npy_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy mel spectrograms to .wav audio files")
    parser.add_argument("--input", "-i", 
                       help="Input .npy file or directory containing .npy files")
    parser.add_argument("--output", "-o", 
                       help="Output .wav file or directory")
    parser.add_argument("--method", "-m", 
                       choices=["hifigan", "griffin_lim"], 
                       default="hifigan",
                       help="Conversion method (default: hifigan)")
    parser.add_argument("--sample_rate", "-sr", 
                       type=int, 
                       default=16000,
                       help="Output sample rate (default: 16000)")
    parser.add_argument("--batch", "-b", 
                       action="store_true",
                       help="Batch convert all .npy files in input directory")
    
    args = parser.parse_args()
    
    # Default paths if not provided (matching your file structure)
    if not args.input:
        # Try common locations
        possible_inputs = [
            "/workspace/output/converted_mel.npy",
            "./output/converted_mel.npy",
            "./processed/source/",
            "./processed/target/"
        ]
        
        for path in possible_inputs:
            if os.path.exists(path):
                args.input = path
                break
        
        if not args.input:
            print("No input specified and no default files found.")
            print("Please specify input with --input or -i")
            exit(1)
    
    if not args.output:
        if os.path.isfile(args.input):
            # Single file conversion
            args.output = args.input.replace('.npy', '.wav')
            if args.output == args.input:  # Fallback if replace didn't work
                args.output = "/workspace/output/converted_output.wav"
        else:
            # Directory conversion
            args.output = "./output/"
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print(f"Sample Rate: {args.sample_rate}")
    
    # Convert
    if args.batch or os.path.isdir(args.input):
        batch_convert_npy_to_wav(args.input, args.output, args.method)
    else:
        success = convert_npy_to_wav(args.input, args.output, args.method, args.sample_rate)
        if success:
            print(f"\n✓ Conversion completed successfully!")
        else:
            print(f"\n✗ Conversion failed!")