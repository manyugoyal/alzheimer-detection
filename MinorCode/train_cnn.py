import torch #type
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from cnn_model import Alzheimer3DCNN
from data_loader import ADNIDataset

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=r"D:\MinorCode\runs\adni_training")

    csv_path = r"D:\MinorCode\final_training_labels.csv"
    dataset = ADNIDataset(csv_file=csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Total training samples mapped: {len(dataset)}")
    
    model = Alzheimer3DCNN(pretrain_mode=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scaler = GradScaler()

    epochs = 30
    global_step = 0

    model.train()
    
    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
        
        for i, (volumes, labels) in enumerate(dataloader):
            labels = labels.squeeze(-1).to(torch.long) - 1
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            # --- THE SHIELD: Bypass corrupted or 1-slice scans ---
            if volumes.shape[2] < 16:
                print(f"[WARNING] Skipping Batch {i+1} | Reason: Scan only has {volumes.shape[2]} slices.")
                continue
            # -----------------------------------------------------

            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(volumes)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # --- TENSORBOARD UI UPDATES ---
            # 1. Update the live Loss Graph
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            
            # 2. Dynamically grab the exact middle slice of the brain for the UI
            mid_idx = volumes.shape[2] // 2
            mid_slice = volumes[0, 0, mid_idx, :, :].unsqueeze(0).cpu()
            writer.add_image('Live MRI Input', mid_slice, global_step)
            # ------------------------------
            
            print(f"Epoch {epoch+1} | Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            global_step += 1

    # Save the trained model weights permanently
    save_path = r"D:\MinorCode\pretrained_3dcnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights successfully saved to {save_path}")
    
    writer.close()

if __name__ == "__main__":
    train_model()