import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse

# --- Parameters ---
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
MIN_AUDIO_LENGTH = 1.0  # Minimum audio length in seconds

# --- Default Paths (Docker-compatible) ---
DEFAULT_SOURCE_DIR = "/workspace/data/source_voice"
DEFAULT_TARGET_DIR = "/workspace/data/target_voice"
DEFAULT_PROCESSED_DIR = "/workspace/processed"

def validate_audio(y, sr, min_length=MIN_AUDIO_LENGTH):
    """Validate audio quality and length"""
    if len(y) == 0:
        return False, "Empty audio"
    
    if len(y) / sr < min_length:
        return False, f"Audio too short (< {min_length}s)"
    
    if np.max(np.abs(y)) < 1e-6:
        return False, "Audio too quiet"
    
    return True, "Valid"

def preprocess_audio(file_path, target_sr=SAMPLE_RATE):
    """Preprocess audio file with validation"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=target_sr)
        
        # Validate
        is_valid, message = validate_audio(y, sr)
        if not is_valid:
            print(f"Warning: {file_path} - {message}")
            return None
        
        # Normalize
        y = y / np.max(np.abs(y))
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Final validation after trimming
        is_valid, message = validate_audio(y, sr)
        if not is_valid:
            print(f"Warning: {file_path} after trimming - {message}")
            return None
            
        return y
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_mel_spectrogram(y, sr=SAMPLE_RATE):
    """Extract mel spectrogram with error handling"""
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=WIN_LENGTH, 
            hop_length=HOP_LENGTH, 
            n_mels=N_MELS,
            fmin=0,
            fmax=sr//2
        )
        
        # Convert to log scale
        mel_spec = np.log1p(mel_spec)
        
        # Validate mel spectrogram
        if mel_spec.shape[1] < 10:  # At least 10 frames
            print("Warning: Mel spectrogram too short")
            return None
            
        return mel_spec
        
    except Exception as e:
        print(f"Error extracting mel spectrogram: {e}")
        return None

def process_directory(input_dir, output_dir, speaker_name="unknown"):
    """Process all audio files in a directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported audio formats
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).glob(f'*{ext}'))
        audio_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return 0
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    processed_count = 0
    for file_path in audio_files:
        try:
            print(f"Processing: {file_path.name}")
            
            # Preprocess audio
            y = preprocess_audio(str(file_path))
            if y is None:
                continue
            
            # Extract mel spectrogram
            mel = extract_mel_spectrogram(y)
            if mel is None:
                continue
            
            # Save mel spectrogram
            output_filename = file_path.stem + '.npy'
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, mel)
            
            print(f"  ✓ Saved: {output_filename} (shape: {mel.shape})")
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            continue
    
    print(f"Successfully processed {processed_count}/{len(audio_files)} files for {speaker_name}")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files for StarGAN-VC training")
    parser.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR, 
                       help="Source speaker audio directory")
    parser.add_argument("--target_dir", default=DEFAULT_TARGET_DIR, 
                       help="Target speaker audio directory")
    parser.add_argument("--output_dir", default=DEFAULT_PROCESSED_DIR, 
                       help="Output directory for processed files")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE, 
                       help="Target sample rate")
    parser.add_argument("--n_mels", type=int, default=N_MELS, 
                       help="Number of mel frequency bins")
    
    args = parser.parse_args()
    
    # Update global parameters
    global SAMPLE_RATE, N_MELS
    SAMPLE_RATE = args.sample_rate
    N_MELS = args.n_mels
    
    # Create output directories
    source_output = os.path.join(args.output_dir, "source")
    target_output = os.path.join(args.output_dir, "target")
    
    print("=" * 50)
    print("StarGAN-VC Audio Preprocessing")
    print("=" * 50)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Mel Bins: {N_MELS}")
    print(f"Hop Length: {HOP_LENGTH}")
    print(f"Window Length: {WIN_LENGTH}")
    print("=" * 50)
    
    total_processed = 0
    
    # Process source directory
    if os.path.exists(args.source_dir):
        print(f"\nProcessing SOURCE speaker: {args.source_dir}")
        count = process_directory(args.source_dir, source_output, "source")
        total_processed += count
    else:
        print(f"Warning: Source directory not found: {args.source_dir}")
    
    # Process target directory
    if os.path.exists(args.target_dir):
        print(f"\nProcessing TARGET speaker: {args.target_dir}")
        count = process_directory(args.target_dir, target_output, "target")
        total_processed += count
    else:
        print(f"Warning: Target directory not found: {args.target_dir}")
    
    print("\n" + "=" * 50)
    print(f"Preprocessing completed!")
    print(f"Total files processed: {total_processed}")
    print(f"Processed files saved to: {args.output_dir}")
    
    # Provide next steps
    if total_processed > 0:
        print("\nNext steps:")
        print("1. Check the processed files in the output directory")
        print("2. Run training: python train.py")
        print("3. After training, use inference.py for voice conversion")
    else:
        print("\nNo files were processed. Please check your input directories and audio files.")

if __name__ == "__main__":
    main()