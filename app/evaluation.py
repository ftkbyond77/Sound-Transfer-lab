import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from scipy import linalg
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class VoiceConversionEvaluator:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
    def compute_mel_cepstral_distortion(self, original_mel, converted_mel):
        """
        Compute Mel Cepstral Distortion (MCD) - Lower is better
        MCD measures the spectral distortion between original and converted audio
        """
        # Convert to numpy if torch tensors
        if torch.is_tensor(original_mel):
            original_mel = original_mel.detach().cpu().numpy()
        if torch.is_tensor(converted_mel):
            converted_mel = converted_mel.detach().cpu().numpy()
        
        # Ensure same dimensions
        min_frames = min(original_mel.shape[-1], converted_mel.shape[-1])
        original_mel = original_mel[..., :min_frames]
        converted_mel = converted_mel[..., :min_frames]
        
        # Compute MCD using the standard formula
        # MCD = (10/ln(10)) * sqrt(2 * sum((c1 - c2)^2))
        diff = original_mel - converted_mel
        mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diff ** 2))
        
        return mcd
    
    def compute_reconstruction_loss(self, original, reconstructed):
        """Compute L1 reconstruction loss"""
        if torch.is_tensor(original) and torch.is_tensor(reconstructed):
            return F.l1_loss(original, reconstructed).item()
        else:
            return np.mean(np.abs(original - reconstructed))
    
    def compute_frechet_audio_distance(self, real_features, fake_features):
        """
        Compute Fréchet Audio Distance (FAD) - Lower is better
        Measures the distance between real and generated audio in feature space
        """
        # Convert to numpy if needed
        if torch.is_tensor(real_features):
            real_features = real_features.detach().cpu().numpy()
        if torch.is_tensor(fake_features):
            fake_features = fake_features.detach().cpu().numpy()
        
        # Flatten if needed
        if real_features.ndim > 2:
            real_features = real_features.reshape(real_features.shape[0], -1)
        if fake_features.ndim > 2:
            fake_features = fake_features.reshape(fake_features.shape[0], -1)
        
        # Compute statistics
        mu1 = np.mean(real_features, axis=0)
        mu2 = np.mean(fake_features, axis=0)
        
        sigma1 = np.cov(real_features, rowvar=False)
        sigma2 = np.cov(fake_features, rowvar=False)
        
        # Compute FAD
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fad = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return fad
    
    def extract_mfcc_features(self, mel_spec, n_mfcc=13):
        """Extract MFCC features for speaker verification"""
        if torch.is_tensor(mel_spec):
            mel_spec = mel_spec.detach().cpu().numpy()
        
        # Convert log mel to linear
        if np.min(mel_spec) >= 0:  # log1p format
            mel_linear = np.expm1(mel_spec)
        else:
            mel_linear = np.exp(mel_spec)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_linear), 
                                   n_mfcc=n_mfcc)
        
        # Flatten and return mean across time
        return np.mean(mfcc, axis=1)
    
    def compute_speaker_verification_accuracy(self, original_mels, converted_mels, 
                                           original_labels, target_labels):
        """
        Compute speaker verification accuracy using MFCC features
        Higher accuracy means better speaker conversion
        """
        # Extract features
        original_features = []
        converted_features = []
        
        print("Extracting MFCC features...")
        for mel in tqdm(original_mels):
            features = self.extract_mfcc_features(mel)
            original_features.append(features)
        
        for mel in tqdm(converted_mels):
            features = self.extract_mfcc_features(mel)
            converted_features.append(features)
        
        original_features = np.array(original_features)
        converted_features = np.array(converted_features)
        
        # Prepare data for classification
        # We want to verify if converted audio sounds like target speaker
        X = np.vstack([original_features, converted_features])
        y = np.hstack([original_labels, target_labels])
        
        # Split and train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM classifier
        classifier = SVC(kernel='rbf', random_state=42)
        classifier.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, classifier, scaler
    
    def compute_pitch_similarity(self, original_mel, converted_mel):
        """
        Compute pitch similarity between original and converted audio
        Higher is better (for same-speaker conversion)
        """
        # This is a simplified version - would need actual audio for precise F0
        # Using mel spectrograms as approximation
        if torch.is_tensor(original_mel):
            original_mel = original_mel.detach().cpu().numpy()
        if torch.is_tensor(converted_mel):
            converted_mel = converted_mel.detach().cpu().numpy()
        
        # Compute correlation across frequency dimension
        correlation = np.corrcoef(original_mel.flatten(), converted_mel.flatten())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

class ModelEvaluator:
    """Evaluate the entire StarGAN-VC model"""
    
    def __init__(self, generator, discriminator, device='cuda'):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.evaluator = VoiceConversionEvaluator()
        
    def evaluate_model(self, dataloader, num_speakers=2):
        """
        Comprehensive model evaluation
        Returns dictionary with all metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        results = {
            'mcd_scores': [],
            'reconstruction_losses': [],
            'discriminator_accuracies': [],
            'speaker_similarities': [],
            'conversion_qualities': []
        }
        
        print("Evaluating model performance...")
        
        with torch.no_grad():
            for batch_idx, (real_mels, speaker_ids) in enumerate(tqdm(dataloader)):
                real_mels = real_mels.to(self.device)
                speaker_ids = speaker_ids.to(self.device)
                batch_size = real_mels.size(0)
                
                # Test different speaker conversions
                for target_id in range(num_speakers):
                    target_ids = torch.full((batch_size,), target_id, 
                                          dtype=torch.long, device=self.device)
                    
                    # Generate converted audio
                    fake_mels = self.generator(real_mels, target_ids)
                    
                    # Compute metrics for each sample in batch
                    for i in range(batch_size):
                        original_speaker = speaker_ids[i].item()
                        target_speaker = target_id
                        
                        real_mel = real_mels[i]
                        fake_mel = fake_mels[i]
                        
                        # MCD Score
                        mcd = self.evaluator.compute_mel_cepstral_distortion(
                            real_mel, fake_mel
                        )
                        results['mcd_scores'].append(mcd)
                        
                        # Reconstruction Loss
                        recon_loss = self.evaluator.compute_reconstruction_loss(
                            real_mel, fake_mel
                        )
                        results['reconstruction_losses'].append(recon_loss)
                        
                        # Speaker Similarity (pitch correlation)
                        similarity = self.evaluator.compute_pitch_similarity(
                            real_mel, fake_mel
                        )
                        results['speaker_similarities'].append(similarity)
                        
                        # Discriminator accuracy (how real does it think converted audio is)
                        disc_output = self.discriminator(fake_mel.unsqueeze(0))
                        disc_prob = torch.sigmoid(disc_output).item()
                        results['discriminator_accuracies'].append(disc_prob)
                        
                        # Conversion quality (combined metric)
                        quality_score = (1.0 / (1.0 + mcd)) * similarity * disc_prob
                        results['conversion_qualities'].append(quality_score)
                
                # Limit evaluation to prevent memory issues
                if batch_idx >= 10:  # Evaluate first 10 batches
                    break
        
        # Compute summary statistics
        summary = {}
        for key, values in results.items():
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary, results
    
    def compute_identity_preservation(self, dataloader):
        """
        Test how well the model preserves identity when converting to same speaker
        Higher is better
        """
        self.generator.eval()
        identity_scores = []
        
        with torch.no_grad():
            for batch_idx, (real_mels, speaker_ids) in enumerate(dataloader):
                real_mels = real_mels.to(self.device)
                speaker_ids = speaker_ids.to(self.device)
                
                # Convert to same speaker (identity conversion)
                fake_mels = self.generator(real_mels, speaker_ids)
                
                # Compute similarity
                for i in range(real_mels.size(0)):
                    similarity = self.evaluator.compute_pitch_similarity(
                        real_mels[i], fake_mels[i]
                    )
                    identity_scores.append(similarity)
                
                if batch_idx >= 5:  # Limit for speed
                    break
        
        return np.mean(identity_scores)

def plot_evaluation_results(summary_results, save_path='evaluation_results.png'):
    """Plot evaluation metrics"""
    metrics = list(summary_results.keys())
    means = [summary_results[metric]['mean'] for metric in metrics]
    stds = [summary_results[metric]['std'] for metric in metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, mean, std) in enumerate(zip(metrics, means, stds)):
        if i < len(axes):
            axes[i].bar([metric], [mean], yerr=[std], capsize=5)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            
            # Add value on top of bar
            axes[i].text(0, mean + std, f'{mean:.3f}±{std:.3f}', 
                        ha='center', va='bottom')
    
    # Hide empty subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Evaluation results saved to: {save_path}")

# Usage example
if __name__ == "__main__":
    # Example of how to use the evaluator
    print("Voice Conversion Model Evaluation")
    print("=" * 40)
    
    # This would be integrated into your training script
    # from model import Generator, Discriminator
    # from dataset import VoiceDataset
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # generator = Generator().to(device)
    # discriminator = Discriminator().to(device)
    
    # # Load your trained model
    # checkpoint = torch.load('path/to/checkpoint.pth')
    # generator.load_state_dict(checkpoint['generator_state_dict'])
    # discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # # Create evaluator
    # evaluator = ModelEvaluator(generator, discriminator, device)
    
    # # Evaluate
    # summary, detailed = evaluator.evaluate_model(test_dataloader)
    # identity_score = evaluator.compute_identity_preservation(test_dataloader)
    
    # # Print results
    # print("\nEvaluation Results:")
    # for metric, stats in summary.items():
    #     print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    # print(f"Identity Preservation: {identity_score:.4f}")
    
    # # Plot results
    # plot_evaluation_results(summary)
    
    print("To use this evaluator, integrate it into your training script!")