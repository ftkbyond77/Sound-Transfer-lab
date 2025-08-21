import torch
import torch.nn as nn
import os
from dataset import get_dataloader
from model import Generator, Discriminator

def train_stargan_vc():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths (Docker-compatible)
    source_dir = "/workspace/processed/source"
    target_dir = "/workspace/processed/target"
    
    # Check if directories exist
    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        print(f"Error: Make sure {source_dir} and {target_dir} exist and contain .npy files")
        return

    # Create checkpoints directory
    os.makedirs("/workspace/checkpoints", exist_ok=True)

    # Dataloader
    try:
        dataloader = get_dataloader(source_dir, target_dir, batch_size=8)
        print(f"Dataset loaded successfully. Total batches: {len(dataloader)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

    # Optimizers
    g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()
    
    # Training parameters
    num_epochs = 100
    lambda_recon = 10.0  # Reconstruction loss weight
    
    # Training loop
    for epoch in range(num_epochs):
        G.train()
        D.train()
        
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        
        for batch_idx, (source, source_id, target, target_id) in enumerate(dataloader):
            batch_size = source.size(0)
            
            # Move to device
            source = source.to(device)
            target = target.to(device)
            source_id = source_id.to(device)
            target_id = target_id.to(device)
            
            # Labels for adversarial loss
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()
            
            # Real samples
            real_pred = D(target)
            d_real_loss = adversarial_loss(real_pred, real_labels)
            
            # Fake samples
            with torch.no_grad():
                fake_target = G(source, target_id)
            fake_pred = D(fake_target.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # -----------------
            # Train Generator
            # -----------------
            g_optimizer.zero_grad()
            
            # Generate converted speech
            converted = G(source, target_id)
            
            # Adversarial loss
            fake_pred = D(converted)
            g_adv_loss = adversarial_loss(fake_pred, real_labels)
            
            # Reconstruction loss (convert back to source)
            reconstructed = G(converted, source_id)
            g_recon_loss = reconstruction_loss(reconstructed, source)
            
            # Identity mapping loss (source -> source)
            identity = G(source, source_id)
            g_identity_loss = reconstruction_loss(identity, source)
            
            # Total generator loss
            g_loss = g_adv_loss + lambda_recon * (g_recon_loss + g_identity_loss)
            g_loss.backward()
            g_optimizer.step()

            # Accumulate losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")

        # Print epoch statistics
        num_batches = len(dataloader)
        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] completed - "
              f"Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }
            torch.save(checkpoint, f"/workspace/checkpoints/stargan_vc_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save(G.state_dict(), "/workspace/starGAN_G.pth")
    torch.save(D.state_dict(), "/workspace/starGAN_D.pth")
    print("Training finished. Models saved.")

if __name__ == "__main__":
    train_stargan_vc()