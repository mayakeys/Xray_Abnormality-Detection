#make_submission.py
import os
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from model import MultiLabelResNet50, MultiLabelResNet50Reg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_model(model, ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model

model_pe  = load_model(MultiLabelResNet50(512), "models/best_pe_model.pth")
model_cm  = load_model(MultiLabelResNet50(512), "models/best_cm_model.pth")
model_ecm = load_model(MultiLabelResNet50(512), "models/best_ec_model.pth")

model_lo = load_model(MultiLabelResNet50(256), "models/lo_partial/epoch_3.pth")
model_lo_fr = load_model(MultiLabelResNet50(256), "models/lo_partial_frontal/epoch_6.pth")
model_lo_l = load_model(MultiLabelResNet50(256), "models/lo_partial_lateral/epoch_4.pth")
model_lo_fr1 = load_model(MultiLabelResNet50(256), "models/lo_partial_frontal/epoch_7.pth")
model_lo_l1 = load_model(MultiLabelResNet50(256), "models/lo_partial_lateral/epoch_8.pth")

model_fr = load_model(MultiLabelResNet50Reg(), "models/best_fr4_model.pth")
model_nf = load_model(MultiLabelResNet50(256), "models/nf_partial/epoch_3.pth")
model_sd = load_model(MultiLabelResNet50(256), "models/sd_partial/epoch_2.pth")
model_pn = load_model(MultiLabelResNet50(256), "models/pn_partial/epoch_6.pth")
model_po = load_model(MultiLabelResNet50(256), "models/po_final/epoch_5.pth")


# --------------------------------------------
# Submission Step
# --------------------------------------------
columns = [
    "Id", "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Pneumonia", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

test_dirs = {
    "frontal": "input_images/test_frontal",
    "lateral": "input_images/test_lateral"
}

# --------------------------------------------
# Loop Over Frontal and Lateral Test Sets
# --------------------------------------------
batch_size = 64
rows = []

for view, folder in test_dirs.items():
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]

    lo_view_1 = model_lo_fr if view == "frontal" else model_lo_l
    lo_view_2 = model_lo_fr1 if view == "frontal" else model_lo_l1

    batch, ids = [], []

    for f in tqdm(files, desc=f"Processing {view}"):
        # Load preprocessed image tensor (.pt) and track its image ID
        batch.append(torch.load(os.path.join(folder, f)).to(device))
        ids.append(f.replace(".pt", ""))

        # Run inference once we hit batch_size or reach the end of the directory
        if len(batch) == batch_size or f == files[-1]:
            x = torch.stack(batch)

            with torch.no_grad():
                pe  = model_pe(x)    # Pleural Effusion
                cm  = model_cm(x)    # Cardiomegaly
                ecm = model_ecm(x)   # Enlarged Cardiomediastinum

                # Lung Opacity: base model + two view-specific models
                lo0 = model_lo(x)
                lo1 = lo_view_1(x)
                lo2 = lo_view_2(x)

                fr  = model_fr(x)    # Fracture
                nf  = model_nf(x)    # No Finding
                sd  = model_sd(x)    # Support Devices
                pn  = model_pn(x)    # Pneumonia
                po  = model_po(x)    # Pleural Other
                
            # Assemble one CSV row per image in the batch
            for i in range(len(batch)):

                # Convert sigmoid output [0, 1] → competition scale [-1, 1]
                def s(t): 
                    return (t[i, 0].item() * 2 - 1)
                
                # Ensemble Lung Opacity by averaging multiple specialized models
                lo_score = (s(lo0) + s(lo1) + s(lo2)) / 3

                rows.append([
                    ids[i],     # Image ID
                    s(nf),
                    s(ecm),
                    s(cm),
                    lo_score,
                    s(pn),
                    s(pe),
                    s(po),
                    s(fr),
                    s(sd),
                ])

            batch, ids = [], []

# --------------------------------------------
# Save CSV
# --------------------------------------------
df = pd.DataFrame(rows, columns=columns).sort_values("Id")
df.to_csv("submission.csv", index=False)
print("Saved submission.csv")
