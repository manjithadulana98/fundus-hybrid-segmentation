import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_closing, binary_fill_holes
import matplotlib.pyplot as plt

from performance_measure import performance_measure
from models.frnet import FRNet, RRCNNBlock, RecurrentConvNeXtBlock

# Paths
manual_type = "1st_manual"
output_dir = "segmentation_outputs"
manual_dir = "DRIVE/Test/" + manual_type
model_path = "FRNET_Model.pth"
save_csv_path = f"FRNet_Performance_{manual_type}.csv"
image_size = (256, 256)  # Model input size

# Load model
model = FRNet(ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Initialize results
results = []

file_list = sorted([f for f in os.listdir(output_dir) if f.endswith("_segmentation.png")])

for filename in tqdm(file_list, desc="Evaluating FRNet"):
    image_id = filename.replace("_segmentation.png", "")
    num_part = image_id.split("_")[0]
    pred_path = os.path.join(output_dir, filename)
    gt_path = os.path.join(manual_dir, f"{num_part}_manual1.gif")

    if not os.path.exists(gt_path):
        print(f"⚠ Ground truth missing for: {filename}")
        continue

    # Load ground truth (unchanged resolution)
    gt_img = Image.open(gt_path).convert("L")
    gt_np = np.array(gt_img)
    gt_bin = (gt_np > 127).astype(np.uint8)

    # Load prediction and resize to GT size
    pred_img = Image.open(pred_path).convert("L")
    pred_resized = pred_img.resize(gt_img.size, Image.BILINEAR)
    pred_np = np.array(pred_resized) / 255.0
    pred_bin = (pred_np > 0.5).astype(np.uint8)

    # Optional post-processing
    pred_bin = binary_closing(pred_bin, structure=np.ones((3, 3)))
    pred_bin = binary_fill_holes(pred_bin).astype(np.uint8)

    # Performance metrics
    Se, Sp, Acc = performance_measure(pred_bin, gt_bin)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)

    results.append({
        "Image": filename,
        "Accuracy": round(Acc, 4),
        "Sensitivity": round(Se, 4),
        "Specificity": round(Sp, 4),
        "Dice": round(dice, 4)
    })

    print(f"{filename} → Acc: {Acc:.4f}, Se: {Se:.4f}, Sp: {Sp:.4f}, Dice: {dice:.4f}")

# Save to CSV
df = pd.DataFrame(results)
avg_row = {
    "Image": "Average",
    "Accuracy": df["Accuracy"].mean(),
    "Sensitivity": df["Sensitivity"].mean(),
    "Specificity": df["Specificity"].mean(),
    "Dice": df["Dice"].mean()
}
df.loc[len(df)] = avg_row
df.to_csv(save_csv_path, index=False)

print(f"\n✅ Evaluation complete. Results saved to {save_csv_path}")
