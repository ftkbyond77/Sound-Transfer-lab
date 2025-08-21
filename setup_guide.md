# StarGAN Voice Conversion Setup Guide

## Project Structure
```
Sound-Transfer-lab/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.sh                    # Helper script (make executable)
├── SETUP_GUIDE.md           # This file
├── app/
│   ├── preprocess.py        # Audio preprocessing
│   ├── dataset.py           # Dataset handling
│   ├── model.py             # StarGAN-VC model
│   ├── train.py             # Training script
│   └── inference.py         # Voice conversion
├── data/
│   ├── source_voice/        # Source speaker audio files (.wav)
│   └── target_voice/        # Target speaker audio files (.wav)
├── processed/
│   ├── source/              # Preprocessed source files (.npy)
│   └── target/              # Preprocessed target files (.npy)
├── checkpoints/             # Training checkpoints
├── models/                  # Final trained models
└── output/                  # Conversion results
```

## Setup Instructions

### 1. Prerequisites
- Docker with GPU support (nvidia-docker)
- NVIDIA Container Runtime
- At least 8GB GPU memory recommended

### 2. Prepare Data
```bash
# Add your audio files
cp source_speaker_files.wav data/source_voice/
cp target_speaker_files.wav data/target_voice/
```

**Audio Requirements:**
- Format: WAV, MP3, FLAC, M4A, AAC
- Quality: 16kHz+ sample rate recommended
- Duration: At least 1 second per file
- Quantity: 100+ files per speaker for better results

### 3. Make Run Script Executable
```bash
chmod +x run.sh
```

### 4. Build Docker Image
```bash
./run.sh build
```

### 5. Preprocess Audio Files
```bash
./run.sh preprocess
```
This will:
- Convert audio to 16kHz
- Extract mel spectrograms
- Save preprocessed data to `processed/` directory

### 6. Train the Model
```bash
./run.sh train
```
Training will:
- Load preprocessed data
- Train StarGAN-VC model
- Save checkpoints every 10 epochs
- Save final model as `starGAN_G.pth`

**Training Parameters:**
- Default epochs: 100
- Batch size: 8
- Learning rate: 1e-4
- Loss: Adversarial + Reconstruction + Identity

### 7. Run Voice Conversion
```bash
./run.sh inference
```
This will convert source voice to target voice style.

## Manual Docker Commands

If you prefer manual control:

```bash
# Build
docker-compose build

# Preprocess
docker-compose run --rm voice-vc python3 /workspace/app/preprocess.py

# Train
docker-compose --profile train run --rm voice-vc-train

# Inference
docker-compose --profile inference run --rm voice-vc-inference

# Interactive shell
docker-compose run --rm -it voice-vc bash
```

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Out of memory during training**
   - Reduce batch size in `train.py`
   - Use gradient accumulation
   - Process shorter audio segments

3. **No audio files found**
   - Check file formats (WAV, MP3, etc.)
   - Verify directory structure
   - Check file permissions

4. **Poor conversion quality**
   - Train longer (more epochs)
   - Use more training data
   - Adjust loss weights
   - Use a neural vocoder for final audio generation

### Monitoring Training

```bash
# View logs
./run.sh logs

# Interactive monitoring
docker-compose run --rm -it voice-vc bash
# Then inside container:
python3 /workspace/app/train.py
```

### Custom Parameters

Edit the Python files directly for custom parameters:

- `preprocess.py`: Audio processing parameters
- `model.py`: Model architecture
- `train.py`: Training hyperparameters
- `dataset.py`: Data loading settings

## Advanced Usage

### Using Checkpoints
```python
# In inference.py, use checkpoint instead of final model
model_path = "/workspace/checkpoints/stargan_vc_epoch_50.pth"
```

### Batch Conversion
```python
# Use batch_convert function in inference.py
batch_convert("/workspace/starGAN_G.pth", 
              "/workspace/input_audio/", 
              "/workspace/output/")
```

### Adding More Speakers
- Modify `model.py` to support more speaker IDs
- Update `dataset.py` for multi-speaker datasets
- Adjust training loop in `train.py`

## File Formats

### Input Audio
- WAV (recommended)
- MP3, FLAC, M4A, AAC (automatically converted)

### Processed Data
- NumPy arrays (.npy) containing mel spectrograms
- Shape: [n_mels, n_frames] = [80, variable]

### Model Output
- Converted mel spectrograms
- Requires vocoder (HiFi-GAN, etc.) for final audio

## Performance Tips

1. **Data Preparation**
   - Use high-quality audio (>16kHz)
   - Ensure speaker consistency
   - Remove silence and noise

2. **Training**
   - Monitor losses for convergence
   - Use learning rate scheduling
   - Implement early stopping

3. **Inference**
   - Use neural vocoders for better audio quality
   - Post-process with audio enhancement

## Next Steps

1. Integrate neural vocoder (HiFi-GAN, MelGAN)
2. Add multi-speaker support
3. Implement real-time conversion
4. Add audio quality metrics
5. Create web interface

For more details, check the individual Python files and their documentation.