import os
import torch
import pydicom
import pandas as pd
import numpy as np
import torch.nn.functional as F

def pack_dataset():
    # 1. Paths
    input_csv = r"D:\MinorCode\final_training_labels.csv"
    output_dir = r"D:\MinorCode\Kaggle_Ready_Data"
    output_csv = os.path.join(output_dir, "kaggle_labels.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['Diagnosis_Label'])
    df['Diagnosis_Label'] = df['Diagnosis_Label'].astype(int)
    
    kaggle_rows = []
    
    print(f"Starting compression for {len(df)} patients...")
    print(f"Saving tiny tensors to: {output_dir}\n")
    
    # 2. Process each patient locally
    for index, row in df.iterrows():
        folder_path = row['Folder_Path']
        label = row['Diagnosis_Label']
        patient_id = row['PTID']
        
        try:
            # Dig for DICOMs
            slices = []
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if filename.endswith(".dcm"):
                        filepath = os.path.join(root, filename)
                        slices.append(pydicom.dcmread(filepath))
            
            if len(slices) < 50:
                continue # Skip bad scans
                
            # Stack and normalize
            slices.sort(key=lambda x: int(x.InstanceNumber))
            volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
            
            vol_min, vol_max = np.min(volume), np.max(volume)
            if vol_max > vol_min:
                volume = (volume - vol_min) / (vol_max - vol_min)
                
            # Resize to 128x128x128
            tensor_volume = torch.tensor(volume).unsqueeze(0).unsqueeze(0)
            tensor_volume = F.interpolate(
                tensor_volume, size=(128, 128, 128), mode='trilinear', align_corners=False
            ).squeeze(0) # Shape: [1, 128, 128, 128]
            
            # 3. Save as a tiny .pt file!
            save_name = f"{patient_id}_{index}.pt"
            save_path = os.path.join(output_dir, save_name)
            torch.save(tensor_volume, save_path)
            
            # Add to the new Kaggle CSV
            kaggle_rows.append({
                'Tensor_File': save_name,
                'Label': label
            })
            
            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1} / {len(df)} scans...")
                
        except Exception as e:
            print(f"Skipping index {index} due to error: {e}")
            continue

    # 4. Save the ultra-clean Kaggle map
    kaggle_df = pd.DataFrame(kaggle_rows)
    kaggle_df.to_csv(output_csv, index=False)
    
    print("\nSUCCESS! Dataset has been crushed down to PyTorch tensors.")
    print(f"Zip the folder '{output_dir}' and upload it to Kaggle!")

if __name__ == "__main__":
    pack_dataset()