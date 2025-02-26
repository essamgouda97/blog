import os
import random
import math
import time
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = '/app/playground/data/modelnet/data'
CSV_PATH = '/app/playground/data/modelnet/metadata_modelnet10.csv'


# Utils from https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook
def read_off(file_path):
    with open(file_path, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')
        # Read number of vertices and faces
        n_verts, n_faces, _ = tuple(int(i) for i in f.readline().strip().split(' '))

        # Extract vertices
        verts = [[float(i) for i in f.readline().strip().split(' ')] for i in range(n_verts)]
        
        # Extract faces
        faces = [[int(i) for i in f.readline().strip().split(' ')][1:] for i in range(n_faces)]
        
    return verts, faces

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=50, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode='immediate'
                        )]
                    )
                ])]
        ),
        frames=frames
    )

    return fig

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def visualize_embeddings(model, dataloader, device, output_path='embeddings.png'):
    """
    Visualize embeddings using UMAP
    """
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder

    model.eval()
    embeddings = []
    labels = []
    
    print("Collecting embeddings...")
    with torch.no_grad():
        for anchor, pos, neg, batch_labels in tqdm(dataloader):
            anchor = anchor.to(device)
            # Skip projection computation to save resources
            _, features, _, _ = model(anchor, compute_projection=False)
            embeddings.append(features.cpu().numpy())
            # Get the actual class labels for these samples
            batch_size = len(anchor)
            start_idx = len(embeddings) * batch_size
            indices = range(start_idx, start_idx + batch_size)
            batch_labels = [dataloader.dataset.classes[idx % len(dataloader.dataset)] for idx in indices]
            labels.extend(batch_labels)
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    
    print(f"Collected {len(embeddings)} embeddings with {len(set(labels))} unique labels")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Reduce dimensionality with UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels_encoded, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title('Embeddings Visualization (UMAP)')
    
    # Add legend
    unique_classes = np.unique(labels)
    handles = [plt.scatter([], [], c=plt.cm.tab10(i/len(unique_classes)), label=cls) 
              for i, cls in enumerate(unique_classes)]
    plt.legend(handles=handles, title='Classes')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


## TRANSFORMATIONS
# Inspired by https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook
class PointSampler(object):
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        
        # Sample points from vertices (with replacement if needed)
        if len(verts) >= self.output_size:
            indices = np.random.choice(len(verts), self.output_size, replace=False)
        else:
            indices = np.random.choice(len(verts), self.output_size, replace=True)
            
        points = verts[indices]
        return torch.from_numpy(points).float()

class Normalize(object):
    def __call__(self, pointcloud):
        if isinstance(pointcloud, tuple):
            pointcloud = pointcloud[0]
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - torch.mean(pointcloud, dim=0) 
        norm_pointcloud /= torch.max(torch.norm(norm_pointcloud, dim=1))
        return norm_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        if isinstance(pointcloud, tuple):
            pointcloud = pointcloud[0]
        assert len(pointcloud.shape)==2

        noise = torch.randn_like(pointcloud) * 0.02
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)


class RandomRotation(object):
    def __call__(self, pointcloud):
        if isinstance(pointcloud, tuple):
            pointcloud = pointcloud[0]
            
        # Random rotation around all axes
        angles = torch.rand(3) * 2 * math.pi
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])]
        ], device=pointcloud.device)
        
        Ry = torch.tensor([
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
        ], device=pointcloud.device)
        
        Rz = torch.tensor([
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1]
        ], device=pointcloud.device)
        
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)
        rotated_pc = torch.matmul(pointcloud, R.T)
        return rotated_pc

class RandomJitter(object):
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip
        
    def __call__(self, pointcloud):
        if isinstance(pointcloud, tuple):
            pointcloud = pointcloud[0]
            
        jitter = torch.clamp(self.scale * torch.randn_like(pointcloud), -self.clip, self.clip)
        jittered_pc = pointcloud + jitter
        return jittered_pc


class PointCloudDataset(Dataset):
    def __init__(self, data_path, metatable_path, train=True, transform=None):
        self.metatable = pd.read_csv(metatable_path)
        self.metatable['class'] = self.metatable['class'].apply(lambda x: 'night_stand' if x == 'night' else x)
        self.metatable = self.metatable[self.metatable['class'] != '.DS']
        
        if train:
            self.metatable = self.metatable.loc[self.metatable['split'] == 'train']
        else:
            self.metatable = self.metatable.loc[self.metatable['split'] == 'test']
            
        # Pre-group objects by class for faster access
        self.class_groups = {name: group['object_path'].values for name, group in self.metatable.groupby('class')}
        self.data_path = data_path
        self.train = train
        self.transform = transform
        
        # Cache file paths
        self.paths = self.metatable['object_path'].values
        self.classes = self.metatable['class'].values
        
        # Pre-compute class indices for faster sampling
        self.class_indices = {cls: np.where(self.classes == cls)[0] for cls in self.class_groups.keys()}
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            # Get anchor object
            anchor_path = self.paths[idx]
            anchor_class = self.classes[idx]
            anchor_verts, anchor_faces = read_off(os.path.join(self.data_path, anchor_path))
            if self.transform is not None:
                anchor = self.transform((anchor_verts, anchor_faces))
            
            # Sample positive from same class
            pos_indices = self.class_indices[anchor_class]
            pos_indices = pos_indices[pos_indices != idx]  # Remove anchor
            pos_idx = np.random.choice(pos_indices)
            pos_path = self.paths[pos_idx]
            pos_verts, pos_faces = read_off(os.path.join(self.data_path, pos_path))
            if self.transform is not None:
                positive = self.transform((pos_verts, pos_faces))
            
            # Sample multiple negatives from different classes
            num_negatives = 3  # Sample from 3 different negative classes
            neg_samples = []
            
            neg_classes = [c for c in self.class_groups.keys() if c != anchor_class]
            sampled_neg_classes = np.random.choice(neg_classes, min(num_negatives, len(neg_classes)), replace=False)
            
            for neg_class in sampled_neg_classes:
                neg_idx = np.random.choice(self.class_indices[neg_class])
                neg_path = self.paths[neg_idx]
                neg_verts, neg_faces = read_off(os.path.join(self.data_path, neg_path))
                if self.transform is not None:
                    negative = self.transform((neg_verts, neg_faces))
                neg_samples.append(negative)
            
            # If we need to pad to reach num_negatives
            while len(neg_samples) < num_negatives:
                neg_samples.append(negative)  # Just duplicate the last one
            
            # Return multiple negatives
            return anchor, positive, neg_samples, anchor_class
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            raise


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, z_a, z_p, z_n):
        """
        Compute NT-Xent loss with triplets
        - z_a: Anchor embedding (B, D)
        - z_p: Positive embedding (B, D)
        - z_n: Negative embedding (B, D)
        """
        # Normalize embeddings
        z_a = F.normalize(z_a, dim=-1)
        z_p = F.normalize(z_p, dim=-1)
        z_n = F.normalize(z_n, dim=-1)

        # Compute similarities
        pos_sim = self.cosine_similarity(z_a, z_p) / self.temperature
        neg_sim = self.cosine_similarity(z_a, z_n) / self.temperature

        # Hard negative mining - focus on the most confusing negatives
        # by adding a margin to push negatives further away
        margin = 0.3
        neg_sim = torch.clamp(neg_sim, max=1.0-margin)
        
        # Compute NT-Xent loss
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.exp(neg_sim)
        
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        return loss.mean()


class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(k, 64, 1)  # (B, K, N) -> (B, 64, N)
        self.batchnorm1 = nn.BatchNorm1d(64)        
        self.conv2 = nn.Conv1d(64, 128, 1)  # (B, 64, N) -> (B, 128, N)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)  # (B, 128, N) -> (B, 1024, N)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.maxpool = nn.MaxPool1d(1024)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(1024, 512)  # (B, 1024) -> (B, 512)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)  # (B, 512) -> (B, 256)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)  # (B, 256) -> (B, k*k)


    def forward(self, x):
        bs = x.size(0)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.flatten(x)

        x = self.relu(self.batchnorm4(self.fc1(x)))
        x = self.relu(self.batchnorm5(self.fc2(x)))

        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            init = init.cuda()

        y = self.fc3(x).view(-1, self.k, self.k) + init  # Ensure transformation is close to identity
        return y


class PointNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(PointNet, self).__init__()
        self.tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.tnet2 = TNet(k=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.AdaptiveMaxPool1d(1)  # Always reduces to (B, 1024, 1)
        self.relu = nn.ReLU()
        
        # Add projection MLP for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x, compute_projection=True):
        if x.shape[1:] != (3, 1024):
          x = x.permute(0, 2, 1)

        # Apply Input T-Net
        matrix3x3 = self.tnet(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix3x3).transpose(1, 2)  # Apply transformation

        # Feature Extraction
        x = self.relu(self.bn1(self.conv1(x)))

        # Apply Feature T-Net
        matrix64x64 = self.tnet2(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1, 2)  # Apply feature transformation

        # More feature extraction
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x).view(-1, 1024)  # Flatten to (B, 1024)
        
        features = x  # Store features before projection
        
        # Only compute projection if needed (during training)
        projection = None
        if compute_projection:
            projection = self.projection(x)  # Get projection for contrastive loss
        
        return projection, features, matrix3x3, matrix64x64



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    training_transform = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandomRotation(),  # Full rotation instead of just z-axis
        RandomJitter(scale=0.02, clip=0.05),  # Stronger jitter
        RandomNoise(),
    ])

    test_transform = transforms.Compose([
        PointSampler(1024),
        Normalize(),
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PointCloudDataset(DATA_DIR, CSV_PATH, train=True, transform=training_transform)
    test_dataset = PointCloudDataset(DATA_DIR, CSV_PATH, train=False, transform=test_transform)
    
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    CHECKPOINT_PATH = 'checkpoint.pth'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    model = PointNet().to(device)
    criterion = NTXentLoss().to(device)
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    # Use one-cycle learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=100,
        pct_start=0.1
    )

    # Initialize tracking variables
    start_epoch = 0
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    # Early stopping parameters
    patience = 10
    early_stop_counter = 0
    
    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_loss = checkpoint['best_test_loss']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        # If the scheduler state is in the checkpoint, load it
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from epoch {start_epoch}")

    print("\nStarting training...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Weight decay: {weight_decay}")
    print(f"Early stopping patience: {patience}")
    
    try:
        # Add mixed precision training
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(start_epoch, 100):
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            # Training loop
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
            for idx, (anchor, positive, negatives, labels) in enumerate(pbar):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negatives = [neg.to(device) for neg in negatives]
                
                with torch.amp.autocast('cuda'):
                    z_a, _, _, _ = model(anchor)
                    z_p, _, _, _ = model(positive)
                    z_n, _, _, _ = model(negatives[0])
                    loss = criterion(z_a, z_p, z_n)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Evaluation every epoch for better monitoring
            model.eval()
            test_loss = 0
            num_test_batches = 0
            
            with torch.no_grad():
                for test_anchor, test_pos, test_negs, test_labels in test_loader:
                    test_anchor = test_anchor.to(device)
                    test_pos = test_pos.to(device)
                    test_negs = [neg.to(device) for neg in test_negs]
                    
                    # During evaluation, we need the projection for computing loss
                    z_a, _, _, _ = model(test_anchor)
                    z_p, _, _, _ = model(test_pos)
                    z_n, _, _, _ = model(test_negs[0])
                    
                    loss = criterion(z_a, z_p, z_n)
                    test_loss += loss.item()
                    num_test_batches += 1
            
            avg_test_loss = test_loss / num_test_batches
            test_losses.append(avg_test_loss)
            
            # Update the learning rate based on validation loss
            scheduler.step(avg_test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Visualization every 10 epochs
            if epoch % 10 == 0:
                # Generate visualization
                visualize_embeddings(
                    model, 
                    test_loader, 
                    device, 
                    output_path=f'embeddings_epoch_{epoch}.png'
                )
            
            # Early stopping check
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                early_stop_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'test_loss': avg_test_loss,
                    'best_test_loss': best_test_loss,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                }, 'best_model.pth')
                print(f"✓ New best model saved (test loss: {best_test_loss:.4f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
            
            # Save checkpoint every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'best_test_loss': best_test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, CHECKPOINT_PATH)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                  f"Test Loss = {avg_test_loss:.4f} "
                  f"(best: {best_test_loss:.4f}), LR: {current_lr:.2e}, "
                  f"ES counter: {early_stop_counter}/{patience}, Time: {epoch_time:.1f}s")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    finally:
        # Always try to save final state
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'best_test_loss': best_test_loss,
            }, CHECKPOINT_PATH)
            print("Final model saved")
            
            # Plot the learning curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.axhline(y=best_test_loss, color='r', linestyle='--', label=f'Best Test Loss ({best_test_loss:.4f})')
            plt.title('Training and Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
            print("Loss curves saved to 'loss_curves.png'")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")

