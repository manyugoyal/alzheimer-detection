import os
import zipfile
import tempfile

import gradio as gr
import torch
import pydicom
import numpy as np
import torch.nn.functional as F

from cnn_model import Alzheimer3DCNN

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading Model and Trained Classifier Weights into memory...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Alzheimer3DCNN(pretrain_mode=True).to(device)

weights_path = "adni_pilot_weights.pth"

if os.path.exists(weights_path):
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    print("Successfully loaded trained weights!")
else:
    print(
        f"WARNING: '{weights_path}' not found. "
        "Model is using random weights!"
    )

model.eval()

# ==========================================================
# CLASS LABELS
# ==========================================================

CLASS_LABELS = {
    0: "Cognitive Normal (CN)",
    1: "Mild Cognitive Impairment (MCI)",
    2: "Alzheimer's Disease (AD)"
}

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def extract_features_and_predict(zip_file):

    if zip_file is None:
        raise gr.Error("Please upload a ZIP file.")

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            # Extract ZIP
            with zipfile.ZipFile(zip_file.name, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            slices = []

            # Search for DICOM files
            for root, dirs, files in os.walk(temp_dir):

                for filename in files:

                    if filename.lower().endswith(".dcm"):

                        filepath = os.path.join(
                            root,
                            filename
                        )

                        slices.append(
                            pydicom.dcmread(filepath)
                        )

            if len(slices) < 50:

                raise gr.Error(
                    f"Only {len(slices)} DICOM slices found.\n"
                    "Please upload a valid MRI scan."
                )

            # ==================================================
            # SAME PREPROCESSING AS TRAINING
            # ==================================================

            slices.sort(
                key=lambda x: int(x.InstanceNumber)
            )

            volume = np.stack(
                [s.pixel_array for s in slices]
            ).astype(np.float32)

            vol_min = np.min(volume)
            vol_max = np.max(volume)

            if vol_max > vol_min:

                volume = (
                    volume - vol_min
                ) / (
                    vol_max - vol_min
                )

            tensor_volume = (
                torch.tensor(volume)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            tensor_volume = F.interpolate(
                tensor_volume,
                size=(128, 128, 128),
                mode="trilinear",
                align_corners=False
            ).to(device)

            # ==================================================
            # INFERENCE
            # ==================================================

            with torch.no_grad():

                if device.type == "cuda":

                    with torch.cuda.amp.autocast():

                        logits = model(
                            tensor_volume
                        )

                else:

                    logits = model(
                        tensor_volume
                    )

                probabilities = (
                    F.softmax(logits, dim=1)
                    .cpu()
                    .numpy()
                    .flatten()
                )

            return {
                CLASS_LABELS[0]:
                    float(probabilities[0]),

                CLASS_LABELS[1]:
                    float(probabilities[1]),

                CLASS_LABELS[2]:
                    float(probabilities[2]),
            }

    except Exception as e:

        raise gr.Error(
            f"Prediction failed:\n{str(e)}"
        )

# ==========================================================
# UI
# ==========================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Alzheimer's MRI Classifier"
) as demo:

    gr.Markdown(
        """
        # 🧠 Alzheimer's Disease MRI Classifier

        Upload a ZIP file containing all DICOM slices
        from a patient's MRI scan.
        """
    )

    with gr.Row():

        upload_file = gr.File(
            file_types=[".zip"],
            label="Upload MRI ZIP"
        )

        prediction_output = gr.Label(
            num_top_classes=3,
            label="Prediction Results"
        )

    predict_btn = gr.Button(
        "Run Analysis",
        variant="primary"
    )

    predict_btn.click(
        fn=extract_features_and_predict,
        inputs=upload_file,
        outputs=prediction_output
    )

    gr.Markdown(
        """
        ---
        ### ⚠ Research Use Only

        This system is intended for educational and
        research purposes only.

        It is **not approved for clinical diagnosis**
        and should not replace professional medical
        evaluation.
        """
    )

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    demo.launch()