import os
import torch #type:ignore
import pydicom #type:ignore
import numpy as np #type:ignore
import pandas as pd #type:ignore
import torch.nn.functional as F #type:ignore
from torch.utils.data import Dataset, DataLoader #type:ignore

class ADNIDataset(Dataset):
    # Added target_shape to enforce uniform tensor sizes
    def __init__(self, csv_file, target_shape=(128, 128, 128)):
        self.data = pd.read_csv(csv_file)
        
        self.data = self.data.dropna(subset=['Diagnosis_Label'])
        self.data['Diagnosis_Label'] = self.data['Diagnosis_Label'].astype(int) - 1
        
        self.data = self.data.reset_index(drop=True)
        self.target_shape = target_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder_path = self.data.loc[idx, 'Folder_Path']
        label = self.data.loc[idx, 'Diagnosis_Label']
        
        slices = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".dcm"):
                filepath = os.path.join(folder_path, filename)
                try:
                    slices.append(pydicom.dcmread(filepath))
                except Exception as e:
                    continue 
        
        # --- DATA SHIELD ---
        # Skip severely truncated/corrupted scans
        if len(slices) < 50:
            print(f"Warning: Corrupt or incomplete scan at {folder_path}. Loading fallback.")
            return self.__getitem__(0) if idx != 0 else (torch.zeros(1, *self.target_shape), torch.tensor([label], dtype=torch.float32))

        slices.sort(key=lambda x: int(x.InstanceNumber))
        volume = np.stack([s.pixel_array for s in slices])

        volume = volume.astype(np.float32)
        vol_min, vol_max = np.min(volume), np.max(volume)
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)

        # (1, 1, D, H, W) for interpolation
        tensor_volume = torch.tensor(volume).unsqueeze(0).unsqueeze(0) 
        
        # --- UNIFORM INTERPOLATION ---
        # Forces the volume to 128x128x128 so DataLoader can process it without crashing
        tensor_volume = F.interpolate(
            tensor_volume, 
            size=self.target_shape, 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0) # Remove batch dim -> (1, D, H, W)
        
        tensor_label = torch.tensor([label], dtype=torch.long) # Changed to long for CrossEntropyLoss

        return tensor_volume, tensor_label

if __name__ == "__main__":
    # Quick test block
    csv_path = r"D:\MinorCode\debug_labels.csv"
    
    adni_dataset = ADNIDataset(csv_file=csv_path, target_shape=(128, 128, 128))
    dataloader = DataLoader(adni_dataset, batch_size=1, shuffle=True)
    
    print("Fetching batch...")
    for volume, label in dataloader:
        print(f"Uniform Batch Tensor Shape: {volume.shape}") 
        print(f"Batch Labels: {label}")
        break