#!/usr/bin/env python3
"""
Quick script to evaluate an already trained StarGAN-VC model
Usage: python check_model_accuracy.py --model_path /path/to/model.pth
"""

import torch
import numpy as np
import os
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your evaluation modules
from evaluation import ModelEvaluator, VoiceConversionEvaluator

def load_model_for_evaluation(model_path, device):
    """Load a trained model for evaluation"""
    # You'll need to import your actual model classes
    # from model import Generator, Discriminator
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model parameters from checkpoint if available
        n_mels = 80
        speaker_dim = 2
        hidden_dim = 256
        
        # Create models (you'll need to uncomment and import your actual classes)
        """
        generator = Generator(n_mels=n_mels, speaker_dim=speaker_dim, hidden_dim=hidden_dim).to(device)
        discriminator = Discriminator(n_mels=n_mels, hidden_dim=hidden_dim).to(device)
        
        # Load state dicts
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Assume it's just the generator state dict
            generator.load_state_dict(checkpoint)
            print("✓ Loaded generator state dict")
        
        generator.eval()
        discriminator.eval()
        
        return generator, discriminator, checkpoint
        """
        
        print("Please uncomment the model loading code and import your model classes")
        return None, None, checkpoint
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None, None

def create_dummy_dataloader(processed_dir, batch_size=16):
    """Create a simple dataloader from processed .npy files for evaluation"""
    
    source_dir = Path(processed_dir) / "source"
    target_dir = Path(processed_dir) / "target"
    
    if not source_dir.exists() or not target_dir.exists():
        print(f"✗ Processed directories not found: {source_dir}, {target_dir}")
        return None
    
    # Get all .npy files
    source_files = list(source_dir.glob("*.npy"))
    target_files = list(target_dir.glob("*.npy"))
    
    if not source_files and not target_files:
        print("✗ No .npy files found in processed directories")
        return None
    
    print(f"Found {len(source_files)} source files and {len(target_files)} target files")
    
    # Simple dataset class
    class SimpleDataset:
        def __init__(self, files, speaker_id=0):
            self.files = files
            self.speaker_id = speaker_id
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            mel_data = np.load(self.files[idx])
            # Ensure correct shape [n_mels, frames]
            if len(mel_data.shape) == 3:
                mel_data = mel_data.squeeze(0)
            
            return torch.from_numpy(mel_data).float(), torch.tensor(self.speaker_id, dtype=torch.long)
    
    # Create datasets
    datasets = []
    if source_files:
        datasets.append(SimpleDataset(source_files, speaker_id=0))
    if target_files:
        datasets.append(SimpleDataset(target_files, speaker_id=1))
    
    # Combine datasets
    combined_data = []
    for dataset in datasets:
        for i in range(len(dataset)):
            combined_data.append(dataset[i])
    
    # Simple dataloader
    class SimpleDataLoader:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i+self.batch_size]
                
                mels = []
                speakers = []
                for mel, speaker in batch:
                    mels.append(mel)
                    speakers.append(speaker)
                
                # Pad sequences to same length
                max_frames = max(mel.shape[-1] for mel in mels)
                padded_mels = []
                for mel in mels:
                    if mel.shape[-1] < max_frames:
                        padding = torch.zeros(mel.shape[0], max_frames - mel.shape[-1])
                        mel = torch.cat([mel, padding], dim=-1)
                    padded_mels.append(mel)
                
                yield torch.stack(padded_mels), torch.stack(speakers)
        
        def __len__(self):
            return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    return SimpleDataLoader(combined_data, batch_size)

def evaluate_existing_model(model_path, processed_dir, output_dir="./evaluation_output"):
    """Evaluate an existing trained model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    generator, discriminator, checkpoint = load_model_for_evaluation(model_path, device)
    
    if generator is None:
        print("Cannot proceed without loaded model. Please update the model loading code.")
        return None
    
    print("Creating evaluation dataloader...")
    dataloader = create_dummy_dataloader(processed_dir, batch_size=8)
    
    if dataloader is None:
        print("Cannot proceed without data. Please check your processed directory.")
        return None
    
    print("Starting evaluation...")
    
    # Create evaluator
    evaluator = ModelEvaluator(generator, discriminator, device)
    
    # Run evaluation
    try:
        summary, detailed = evaluator.evaluate_model(dataloader, num_speakers=2)
        identity_score = evaluator.compute_identity_preservation(dataloader)
        
        # Create results dictionary
        results = {
            'model_path': model_path,
            'evaluation_date': str(datetime.now()),
            'device': str(device),
            'summary_metrics': summary,
            'identity_preservation': identity_score,
            'training_info': {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_mcd': checkpoint.get('best_mcd', 'unknown'),
                'best_quality': checkpoint.get('best_quality', 'unknown')
            }
        }
        
        # Print results
        print("\n" + "="*60)
        print("            MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nModel: {os.path.basename(model_path)}")
        print(f"Epoch: {results['training_info']['epoch']}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        for metric, stats in summary.items():
            mean_val = stats['mean']
            std_val = stats['std']
            print(f"   • {metric.replace('_', ' ').title():<25}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"   • Identity Preservation{' ':<10}: {identity_score:.4f}")
        
        # Interpret results
        mcd_score = summary['mcd_scores']['mean']
        quality_score = summary['conversion_qualities']['mean']
        
        print(f"\n🎯 PERFORMANCE INTERPRETATION:")
        
        if mcd_score < 5.0:
            print("   • MCD Score: EXCELLENT - Very low spectral distortion")
        elif mcd_score < 8.0:
            print("   • MCD Score: GOOD - Acceptable spectral quality")
        elif mcd_score < 12.0:
            print("   • MCD Score: FAIR - Some spectral artifacts present")
        else:
            print("   • MCD Score: POOR - High spectral distortion")
        
        if quality_score > 0.7:
            print("   • Conversion Quality: EXCELLENT - High-quality conversions")
        elif quality_score > 0.5:
            print("   • Conversion Quality: GOOD - Decent conversion quality")
        elif quality_score > 0.3:
            print("   • Conversion Quality: FAIR - Noticeable artifacts")
        else:
            print("   • Conversion Quality: POOR - Significant quality issues")
        
        if identity_score > 0.8:
            print("   • Identity Preservation: EXCELLENT - Content well preserved")
        elif identity_score > 0.6:
            print("   • Identity Preservation: GOOD - Most content preserved")
        elif identity_score > 0.4:
            print("   • Identity Preservation: FAIR - Some content loss")
        else:
            print("   • Identity Preservation: POOR - Significant content loss")
        
        print("="*60)
        
        # Save results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n✓ Detailed results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained StarGAN-VC model accuracy")
    parser.add_argument('--model_path', '-m', 
                       default='/workspace/checkpoints/stargan_vc_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--processed_dir', '-d',
                       default='/workspace/processed',
                       help='Directory containing processed .npy files')
    parser.add_argument('--output_dir', '-o',
                       default='/workspace/output/evaluation',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print("StarGAN-VC Model Accuracy Checker")
    print("-" * 40)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.processed_dir}")
    print(f"Output: {args.output_dir}")
    print("-" * 40)
    
    if not os.path.exists(args.model_path):
        print(f"✗ Model file not found: {args.model_path}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = os.path.dirname(args.model_path)
        if os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(checkpoints_dir, f)}")
        return
    
    if not os.path.exists(args.processed_dir):
        print(f"✗ Processed data directory not found: {args.processed_dir}")
        return
    
    # Run evaluation
    results = evaluate_existing_model(args.model_path, args.processed_dir, args.output_dir)
    
    if results:
        print(f"\n✓ Evaluation completed successfully!")
    else:
        print(f"\n✗ Evaluation failed!")

if __name__ == "__main__":
    main()