import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import your original model class from your local cnn_model.py
from cnn_model import Alzheimer3DCNN

# ---------------------------------------------------------
# 1. LOCAL DATA LOADER
# ---------------------------------------------------------
class LocalADNIDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ Missing Specified CSV File mapping: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Static mapping matching your training setup (1->0, 2->1, 3->2)
        self.label_map = {1: 0, 2: 1, 3: 2} 
        print(f"Dataset loaded successfully. Evaluating exactly {len(self.data)} elements.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_name = self.data.iloc[idx]['Tensor_File']
        raw_label = self.data.iloc[idx]['Label']
        safe_label = self.label_map[raw_label]
        
        tensor_path = os.path.join(self.root_dir, tensor_name)
        volume = torch.load(tensor_path, map_location='cpu') 
        
        return volume, torch.tensor(safe_label, dtype=torch.long)

# ---------------------------------------------------------
# 2. ACCURACY EVALUATION LOGIC
# ---------------------------------------------------------
def evaluate_accuracy():
    print("\n--- RUNNING EVALUATION ON 137 DATA ELEMENTS ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware acquired: {device}")
    
    # Direct explicit paths
    data_dir = r"D:\MinorCode\Kaggle_Ready_Dataa\Kaggle_Ready_Data"
    csv_path = os.path.join(data_dir, "kaggle_labels.csv")
    weights_path = r"D:\alzh\adni_pilot_weights.pth" 
    
    # Initialize data pipeline
    dataset = LocalADNIDataset(csv_file=csv_path, root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    readable_labels = {0: "Cognitive Normal (CN)", 1: "Mild Cognitive Impairment (MCI)", 2: "Alzheimer's Disease (AD)"} 
    
    # Initialize Model and Load Weights
    model = Alzheimer3DCNN(pretrain_mode=True).to(device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded weights from: {weights_path}")
    else:
        raise FileNotFoundError(f"❌ Missing weights file at: {weights_path}")

    model.eval() 
    
    correct_predictions = 0
    total_samples = 0
    
    class_correct = [0] * 3
    class_total = [0] * 3

    # Inference Loop
    print("\nRunning model inference across test elements...")
    with torch.no_grad(): 
        for volumes, labels in tqdm(dataloader, desc="Processing Samples"):
            volumes = volumes.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda'):
                outputs = model(volumes)
            
            _, predicted = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # Display Final Results
    overall_accuracy = (correct_predictions / total_samples) * 100
    
    print("\n" + "="*50)
    print(f"📊 TEST ACCURACY (137 Samples): {overall_accuracy:.2f}%")
    print(f"🔢 Total Scanned: {total_samples}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print("="*50)
    
    print("\n📈 Class Accuracy Breakdown:")
    for idx in range(3):
        c_name = readable_labels[idx]
        if class_total[idx] > 0:
            class_acc = (class_correct[idx] / class_total[idx]) * 100
            print(f" - {c_name}: {class_acc:.2f}% ({class_correct[idx]}/{class_total[idx]} verified)")
        else:
            print(f" - {c_name}: 0 samples in dataset mapping file.")
            
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_accuracy()