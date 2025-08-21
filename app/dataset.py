import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StarGANVCDataset(Dataset):
    def __init__(self, source_dir, target_dir, segment_frames=128):
        self.source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".npy")]
        self.target_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".npy")]
        self.segment_frames = segment_frames
        
        # Check if directories exist and have files
        if not self.source_files:
            raise ValueError(f"No .npy files found in source directory: {source_dir}")
        if not self.target_files:
            raise ValueError(f"No .npy files found in target directory: {target_dir}")
        
        # Speaker IDs (0=source, 1=target)
        self.speaker_map = {os.path.basename(source_dir): 0, os.path.basename(target_dir): 1}

    def __len__(self):
        return min(len(self.source_files), len(self.target_files))

    def _pad_or_segment(self, mel_spec, target_frames):
        """Pad or segment mel spectrogram to target frames"""
        if mel_spec.shape[1] < target_frames:
            # Pad if too short
            pad_width = target_frames - mel_spec.shape[1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_width), mode='constant', value=0)
        elif mel_spec.shape[1] > target_frames:
            # Random segment if too long
            start = torch.randint(0, mel_spec.shape[1] - target_frames + 1, (1,)).item()
            mel_spec = mel_spec[:, start:start+target_frames]
        return mel_spec

    def __getitem__(self, idx):
        try:
            source_mel = np.load(self.source_files[idx % len(self.source_files)])
            target_mel = np.load(self.target_files[idx % len(self.target_files)])
        except Exception as e:
            print(f"Error loading files: {e}")
            # Return dummy data if file loading fails
            source_mel = np.zeros((80, self.segment_frames))
            target_mel = np.zeros((80, self.segment_frames))

        # Convert to torch tensors
        source_mel = torch.from_numpy(source_mel).float()
        target_mel = torch.from_numpy(target_mel).float()

        # Ensure consistent dimensions
        source_mel = self._pad_or_segment(source_mel, self.segment_frames)
        target_mel = self._pad_or_segment(target_mel, self.segment_frames)

        # Return mel + speaker ID
        return source_mel, torch.tensor(0, dtype=torch.long), target_mel, torch.tensor(1, dtype=torch.long)

def get_dataloader(source_dir, target_dir, batch_size=8, num_workers=0):
    dataset = StarGANVCDataset(source_dir, target_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)