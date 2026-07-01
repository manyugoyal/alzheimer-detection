import os
import gradio as gr
import torch
import pydicom
import numpy as np
import torch.nn.functional as F

# Import your model!
from cnn_model import Alzheimer3DCNN

print("Loading Model and Trained Classifier Weights into memory...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX 1: Set pretrain_mode=True so the classifier head is built
model = Alzheimer3DCNN(pretrain_mode=True).to(device)

# FIX 2: Load your trained weights! 
# Make sure 'adni_pilot_weights.pth' is in the same folder as this script.
weights_path = "adni_pilot_weights.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Successfully loaded trained weights!")
else:
    print(f"WARNING: '{weights_path}' not found. Model is using random weights!")

model.eval() # CRITICAL: Sets the model to "Inference" mode

# Define your classification labels based on how your DataLoader mapped them
# Adjust these labels (e.g., Normal, MCI, AD) to match your dataset's actual classes!
CLASS_LABELS = {0: "Cognitive Normal (CN)", 1: "Mild Cognitive Impairment (MCI)", 2: "Alzheimer's Disease (AD)"}

def extract_features_and_predict(folder_path):
    """This function runs when you click 'Submit' in the UI"""
    
    if not os.path.exists(folder_path):
        return "Error: That folder does not exist. Please check the path."
    
    try:
        # 1. Load and process the DICOMs (Recursive search)
        slices = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".dcm"):
                    filepath = os.path.join(root, filename)
                    slices.append(pydicom.dcmread(filepath))
                
        if len(slices) < 50:
            return f"Error: Found only {len(slices)} slices. Make sure you pasted a valid patient path."

        # Sort slices chronologically by their spatial position sequence
        slices.sort(key=lambda x: int(x.InstanceNumber))
        # Stack them into a 3D numpy volume array
        volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
        
        # Normalize
        vol_min, vol_max = np.min(volume), np.max(volume)
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)

        # Convert to Tensor and Interpolate to 128x128x128
        tensor_volume = torch.tensor(volume).unsqueeze(0).unsqueeze(0)
        tensor_volume = F.interpolate(
            tensor_volume, size=(128, 128, 128), mode='trilinear', align_corners=False
        ).to(device)

        # 2. Run the CNN and the Classifier Head!
        with torch.no_grad(): # No gradients needed for UI/inference
            with torch.cuda.amp.autocast(): # Keep memory low
                logits = model(tensor_volume)
                # FIX 3: Convert raw outputs (logits) into percentages/probabilities
                probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
        
        # Find the index with the highest probability
        predicted_class_idx = np.argmax(probabilities)
        predicted_label = CLASS_LABELS.get(predicted_class_idx, f"Class {predicted_class_idx}")
        
        # 3. Format the output elegantly for the screen
        result_text = f"🎉 PREDICTION SEQUENCE COMPLETE 🎉\n\n"
        result_text += f"➡️ FINAL DIAGNOSIS: {predicted_label}\n"
        result_text += "=" * 40 + "\n\n"
        result_text += "Confidence Breakdown:\n"
        
        for idx, prob in enumerate(probabilities):
            label_name = CLASS_LABELS.get(idx, f"Class {idx}")
            result_text += f" - {label_name}: {prob * 100:.2f}%\n"
            
        return result_text

    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- BUILD THE UI ---
demo = gr.Interface(
    fn=extract_features_and_predict,
    inputs=gr.Textbox(
        lines=1, 
        placeholder="Paste patient DICOM folder path here...", 
        label="Patient MRI Folder Path"
    ),
    outputs=gr.Textbox(lines=10, label="Classifier Diagnosis & Confidence Scores"),
    title="Alzheimer's 3D CNN - Classifier Predictor",
    description="Paste a folder path containing DICOM slices to run full CNN extraction and classification prediction."
)

if __name__ == "__main__":
    demo.launch()