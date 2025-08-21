import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_mels=80, speaker_dim=2, hidden_dim=256):
        super().__init__()
        self.n_mels = n_mels
        self.speaker_dim = speaker_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim + speaker_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, n_mels, kernel_size=5, padding=2)
        )

    def forward(self, x, speaker_id):
        # x: [B, n_mels, frames]
        batch_size, _, frames = x.shape
        
        # Encode
        h = self.encoder(x)
        
        # Create speaker embedding
        speaker_embed = torch.nn.functional.one_hot(speaker_id, num_classes=self.speaker_dim).float()
        # Expand to match frames dimension: [B, speaker_dim] -> [B, speaker_dim, frames]
        speaker_embed = speaker_embed.unsqueeze(2).expand(batch_size, self.speaker_dim, frames)
        
        # Concatenate encoded features with speaker embedding
        h = torch.cat([h, speaker_embed], dim=1)
        
        # Decode
        out = self.decoder(h)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)