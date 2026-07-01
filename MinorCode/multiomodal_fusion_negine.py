import torch
import torch.nn as nn
from transformers import HubertModel, BertModel
from cnn_model import Alzheimer3DCNN

class IntegratedMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. ENCODERS (Your "Eyes" and "Ears")
        self.cnn = Alzheimer3DCNN() # Your existing model
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Freeze encoders so we only train the Imputation and Classification heads
        for param in self.hubert.parameters(): param.requires_grad = False
        for param in self.bert.parameters(): param.requires_grad = False

        # 2. IMPUTATION ENGINE (The "Mind Readers")
        self.mri_to_speech = nn.Sequential(nn.Linear(524288, 2048), nn.ReLU(), nn.Linear(2048, 768))
        self.speech_to_mri = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 524288))
        
        # 3. MASTER CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(524288 + 768, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 3) 
        )

    def forward(self, mri_raw=None, speech_raw=None):
        # A. FEATURE EXTRACTION
        mri_feat = self.cnn(mri_raw) if mri_raw is not None else None
        
        # HuBERT output: take the mean of the sequence to get 768 dim vector
        if speech_raw is not None:
            speech_feat = self.hubert(speech_raw).last_hidden_state.mean(dim=1)
        else:
            speech_feat = None

        # B. IMPUTATION LOGIC
        if mri_feat is not None and speech_feat is None:
            # Predict missing speech
            speech_feat = self.mri_to_speech(mri_feat)
            combined = torch.cat((mri_feat, speech_feat), dim=1)
            
        elif speech_feat is not None and mri_feat is None:
            # Predict missing MRI
            mri_feat = self.speech_to_mri(speech_feat)
            combined = torch.cat((mri_feat, speech_feat), dim=1)
            
        else:
            combined = torch.cat((mri_feat, speech_feat), dim=1)

        # C. CLASSIFICATION
        return self.classifier(combined)

# --- HOW TO RUN IT ---
# model = IntegratedMultimodalModel()
# output = model(mri_raw=some_mri, speech_raw=some_audio)