import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Configure PyTorch memory management
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better memory efficiency
torch.backends.cudnn.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Local imports
from src.models.perception_system import PerceptionSystem

class EnhancedMultiModalDataset(Dataset):
    def __init__(self, data_dir, towns=['Town01','Town03'], weather_conditions=['ClearNight','ClearNoon']):
        super().__init__()
        self.samples = []
        print(f"Initializing dataset from {data_dir}")

        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")

        for town in towns:
            town_path = os.path.join(data_dir, town)
            if not os.path.exists(town_path):
                continue

            for weather in weather_conditions:
                weather_path = os.path.join(town_path, weather)
                if not os.path.exists(weather_path):
                    continue

                rgb_path = os.path.join(weather_path, 'rgb')
                depth_path = os.path.join(weather_path, 'depth')
                seg_path = os.path.join(weather_path, 'segmentation')

                if not all(os.path.exists(p) for p in [rgb_path, depth_path, seg_path]):
                    continue

                self._process_sequence(rgb_path, depth_path, seg_path)

        if len(self.samples) == 0:
            raise ValueError("No valid samples found")

        # Data augmentation pipeline
        self.rgb_transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        self.seg_transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def _process_sequence(self, rgb_path, depth_path, seg_path):
        try:
            rgb_files = sorted(os.listdir(rgb_path))
            depth_files = sorted(os.listdir(depth_path))
            seg_files = sorted(os.listdir(seg_path))

            def extract_frame_id(filename):
                return int(filename.split('-')[-1].split('_')[0].split('.')[0])

            for rgb_file in rgb_files:
                try:
                    rgb_frame_id = extract_frame_id(rgb_file)
                    closest_depth = self._find_closest_frame(depth_files, rgb_frame_id, extract_frame_id)
                    closest_seg = self._find_closest_frame(seg_files, rgb_frame_id, extract_frame_id)

                    if closest_depth and closest_seg:
                        self.samples.append({
                            'rgb': os.path.join(rgb_path, rgb_file),
                            'depth': os.path.join(depth_path, closest_depth),
                            'seg': os.path.join(seg_path, closest_seg),
                            'frame_diff': abs(extract_frame_id(closest_depth) - rgb_frame_id) +
                                         abs(extract_frame_id(closest_seg) - rgb_frame_id)
                        })
                except Exception as e:
                    continue
        except Exception as e:
            print(f"Error processing sequence: {e}")

    def _find_closest_frame(self, files, target_id, id_extractor):
        closest_file = None
        min_diff = float('inf')
        
        for f in files:
            try:
                current_id = id_extractor(f)
                diff = abs(current_id - target_id)
                if diff < min_diff:
                    closest_file = f
                    min_diff = diff
            except:
                continue
        
        return closest_file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform images
        rgb = self.rgb_transform(Image.open(sample['rgb']).convert('RGB'))
        depth = self.depth_transform(Image.open(sample['depth']).convert('L'))
        seg = self.seg_transform(Image.open(sample['seg']))

        return {
            'rgb': rgb,
            'depth': depth,
            'segmentation': seg,
            'frame_diff': torch.tensor(sample['frame_diff'], dtype=torch.float32)
        }

class EnhancedSSLPerceptionTrainer(nn.Module):
    def __init__(self, perception_system, feature_dim=256, temperature=0.07):
        super().__init__()
        self.perception = perception_system
        self.temperature = temperature
        
        # Enable gradient checkpointing for memory efficiency
        self.perception.rgb_encoder.gradient_checkpointing_enable()
        self.perception.depth_encoder.gradient_checkpointing_enable()
        self.perception.segmentation_parser.gradient_checkpointing_enable()
        self.perception.fusion_transformer.gradient_checkpointing_enable()

        # Projectors for contrastive learning (including fusion)
        self.projectors = nn.ModuleDict({
            'rgb': self._build_projector(feature_dim),
            'depth': self._build_projector(feature_dim),
            'seg': self._build_projector(feature_dim),
            'fused': self._build_projector(feature_dim)
        })

        # Decoders for reconstruction
        self.decoders = nn.ModuleDict({
            'rgb': self._build_decoder(feature_dim, 3),
            'depth': self._build_decoder(feature_dim, 1),
            'seg': self._build_decoder(feature_dim, 1)
        })

    def _build_projector(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 128)
        )

    def _build_decoder(self, dim, out_channels):
        return nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 480 * 640 * out_channels),
            nn.Unflatten(1, (out_channels, 480, 640))
        )

    def forward(self, batch):
        # Clear unnecessary memory
        torch.cuda.empty_cache()
        
        # Extract individual modality features with memory optimization
        with torch.cuda.amp.autocast():
            rgb_feat = self.perception.rgb_encoder(batch['rgb'])
            depth_feat = self.perception.depth_encoder(batch['depth'])
            seg_feat = self.perception.segmentation_parser(batch['segmentation'])
        
        # Get fused features through transformer
        fused_feat = self.perception.fusion_transformer(rgb_feat, depth_feat, seg_feat)
        
        # Update temporal memory and get temporal features
        self.perception.memory.append(fused_feat)
        if len(self.perception.memory) > self.perception.num_frames:
            self.perception.memory.pop(0)
            
        # Get temporal features if we have enough frames
        if len(self.perception.memory) >= 2:  # At least 2 frames for temporal consistency
            temporal_input = torch.stack(self.perception.memory, dim=1)
            temporal_feat, _ = self.perception.temporal_aggregator(temporal_input)
        else:
            temporal_feat = fused_feat
        
        # Collect all features
        features = {
            'rgb': rgb_feat,
            'depth': depth_feat,
            'seg': seg_feat,
            'fused': fused_feat,
            'temporal': temporal_feat
        }
        
        # Project features for contrastive learning
        projections = {}
        for modality in ['rgb', 'depth', 'seg', 'fused']:
            projections[modality] = self.projectors[modality](features[modality])

        # Reconstruct inputs
        reconstructions = {}
        for modality in ['rgb', 'depth', 'seg']:
            reconstructions[modality] = self.decoders[modality](features[modality])

        return features, projections, reconstructions

    def compute_losses(self, batch, features, projections, reconstructions):
        losses = {}
        
        # Contrastive losses between modalities and fusion
        for m1 in ['rgb', 'depth', 'seg', 'fused']:
            for m2 in ['rgb', 'depth', 'seg', 'fused']:
                if m1 < m2:  # Avoid duplicate pairs
                    losses[f'contrastive_{m1}_{m2}'] = self.contrastive_loss(
                        projections[m1], projections[m2], batch['frame_diff']
                    )

        # Temporal consistency loss if we have enough frames
        if len(self.perception.memory) >= 2:
            losses['temporal_consistency'] = F.mse_loss(features['temporal'], features['fused'])

        # Reconstruction losses
        for modality in ['rgb', 'depth', 'seg']:
            losses[f'recon_{modality}'] = F.mse_loss(
                reconstructions[modality], batch[modality]
            )

        # Feature consistency loss
        losses['feature_consistency'] = self.feature_consistency_loss(features)

        # Compute total loss with adaptive weighting
        total_loss = sum(losses.values())
        
        return total_loss, losses

    def contrastive_loss(self, features_1, features_2, frame_diff):
        batch_size = features_1.shape[0]
        features = torch.cat([features_1.unsqueeze(1), features_2.unsqueeze(1)], dim=1)
        features = features.view(-1, features.shape[-1])

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Frame-aware positive pair weighting
        frame_weights = 1.0 / (1.0 + frame_diff.view(-1, 1))
        sim_matrix = sim_matrix * frame_weights

        # Compute InfoNCE loss
        labels = torch.arange(batch_size, device=features.device).repeat_interleave(2)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def feature_consistency_loss(self, features):
        loss = 0
        for m1 in ['rgb', 'depth', 'seg']:
            for m2 in ['rgb', 'depth', 'seg']:
                if m1 < m2:
                    loss += F.mse_loss(features[m1], features[m2])
        return loss / 3  # Average over all pairs

def train_ssl_perception(data_dir, perception_system, num_epochs=100, batch_size=2, learning_rate=1e-4, checkpoint_dir='checkpoints', accumulation_steps=64):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize dataset and dataloader with error handling
    try:
        dataset = EnhancedMultiModalDataset(data_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Optimize data loading with memory-efficient settings
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=2, pin_memory=True, prefetch_factor=2,
                                persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)
    except Exception as e:
        print(f"Error initializing dataset: {str(e)}")
        raise

    # Initialize trainer, optimizer and AMP scaler with memory optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model components separately to manage memory
    perception_system = perception_system.to(device)
    trainer = EnhancedSSLPerceptionTrainer(perception_system)
    
    # Use gradient checkpointing and mixed precision
    trainer.train()
    trainer = trainer.to(device)
    
    # Configure optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = amp.GradScaler()

    # Initialize tensorboard writer
    log_dir = 'tensorboard_logs/ssl'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop with progress tracking and memory management
    best_val_loss = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Starting training for {num_epochs} epochs...")
    try:
        for epoch in range(num_epochs):
            trainer.train()
            train_losses = []
            train_component_losses = defaultdict(list)

            # Training phase with progress bar
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            optimizer.zero_grad()
            for idx, batch in enumerate(train_pbar):
                try:
                    # Clear cache before processing batch
                    torch.cuda.empty_cache()
                    
                    # Move batch to device efficiently
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                    # Forward pass with mixed precision and memory optimization
                    with torch.cuda.amp.autocast():
                        features, projections, reconstructions = trainer(batch)
                        total_loss, component_losses = trainer.compute_losses(
                            batch, features, projections, reconstructions
                        )
                        # Scale loss by accumulation steps
                        total_loss = total_loss / accumulation_steps

                    # Backward pass with gradient scaling
                    scaler.scale(total_loss).backward()
                    
                    # Gradient accumulation
                    if (idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    # Update progress bar
                    train_losses.append(total_loss.item())
                    for name, loss in component_losses.items():
                        train_component_losses[name].append(loss.item())
                    
                    train_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
                except Exception as e:
                    print(f"Error in training batch: {str(e)}")
                    continue

            # Validation phase
            trainer.eval()
            val_losses = []
            val_component_losses = defaultdict(list)
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        features, projections, reconstructions = trainer(batch)
                        total_loss, component_losses = trainer.compute_losses(
                            batch, features, projections, reconstructions
                        )
                        val_losses.append(total_loss.item())
                        for name, loss in component_losses.items():
                            val_component_losses[name].append(loss.item())
                            
                        val_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
                    except Exception as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue

            # Calculate and log metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            # Log all component losses
            for name in train_component_losses.keys():
                train_comp_loss = np.mean(train_component_losses[name])
                val_comp_loss = np.mean(val_component_losses[name])
                writer.add_scalar(f'train/{name}', train_comp_loss, epoch)
                writer.add_scalar(f'val/{name}', val_comp_loss, epoch)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
            
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            # Save checkpoints
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            
            # Save latest checkpoint
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
                # Save perception system weights separately for RL use
                torch.save(trainer.perception.state_dict(), os.path.join(checkpoint_dir, 'pretrained_perception.pth'))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Unexpected error during training: {str(e)}")
        raise
    finally:
        writer.close()

    return trainer

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Training configuration with memory optimization
    config = {
        'data_dir': 'data',  # Data directory relative to src
        'batch_size': 2,  # Reduced batch size to save memory
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'feature_dim': 256,
        'checkpoint_dir': 'checkpoints/ssl',
        'accumulation_steps': 64  # Gradient accumulation steps to maintain effective batch size
    }
    
    try:
        # Initialize perception system
        perception_system = PerceptionSystem(
            feature_dim=config['feature_dim'],
            fusion_dim=config['feature_dim'],
            output_dim=config['feature_dim']
        )
        
        # Train the model
        trainer = train_ssl_perception(
            data_dir=config['data_dir'],
            perception_system=perception_system,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            checkpoint_dir=config['checkpoint_dir']
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()