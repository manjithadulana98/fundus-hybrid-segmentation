import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_closing
from skimage.morphology import remove_small_objects

# -------------------------------
# CONFIGURATION
# -------------------------------
manual_type = "2nd_manual"
annotation_num = manual_type[0]
PRED_DIR = "segmentation_outputs"
GT_DIR = f"DRIVE/test/{manual_type}"
PLOT_DIR = f"eval_plots_{manual_type}"
CSV_OUT = f"FRNet_Performance_{manual_type}.csv"

RESIZE_DIM = None  # Set to (256, 256) if resizing needed
MIN_VESSEL_SIZE = 80  # Min pixel count for vessels in GT

os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------------
# METRIC FUNCTIONS
# -------------------------------
def compute_metrics(pred, gt):
    TP = np.logical_and(pred == 1, gt == 1).sum()
    FP = np.logical_and(pred == 1, gt == 0).sum()
    FN = np.logical_and(pred == 0, gt == 1).sum()
    TN = np.logical_and(pred == 0, gt == 0).sum()

    Se = TP / (TP + FN + 1e-8)
    Sp = TN / (TN + FP + 1e-8)
    Acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    Dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    return Acc, Se, Sp, Dice

# -------------------------------
# MAIN LOOP
# -------------------------------
results = []

files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith("_segmentation.png")])

for filename in tqdm(files, desc="Evaluating FRNet"):
    base = filename.replace("_segmentation.png", "")
    base_num = base.split("_")[0]
    pred_path = os.path.join(PRED_DIR, filename)
    gt_path = os.path.join(GT_DIR, f"{base_num}_manual{annotation_num}.gif")

    if not os.path.exists(gt_path):
        print(f"⚠ Skipping (GT not found): {gt_path}")
        continue

    # --- Load prediction ---
    pred_img = Image.open(pred_path).convert("L")
    if RESIZE_DIM:
        pred_img = pred_img.resize(RESIZE_DIM, Image.NEAREST)
    pred_np = (np.array(pred_img) > 127).astype(np.uint8)

    # --- Load and prune GT ---
    gt_img = Image.open(gt_path).convert("L")
    if RESIZE_DIM:
        gt_img = gt_img.resize(RESIZE_DIM, Image.NEAREST)
    gt_np = (np.array(gt_img) > 127).astype(np.uint8)
    gt_smooth = binary_closing(gt_np, structure=np.ones((3, 3)))
    gt_pruned = remove_small_objects(gt_smooth.astype(bool), min_size=MIN_VESSEL_SIZE)
    gt_final = gt_pruned.astype(np.uint8)

    # --- Compute metrics ---
    acc, se, sp, dice = compute_metrics(pred_np, gt_final)

    # --- Save result ---
    results.append({
        "Image": filename,
        "Accuracy": acc,
        "Sensitivity": se,
        "Specificity": sp,
        "Dice": dice
    })

    # --- Save visualization ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(gt_final, cmap="gray")
    axs[0].set_title("Ground Truth (Pruned)")
    axs[0].axis("off")

    axs[1].imshow(pred_np, cmap="gray")
    axs[1].set_title("Prediction")
    axs[1].axis("off")

    # Overlay prediction on GT
    overlay = np.zeros((*gt_final.shape, 3), dtype=np.uint8)
    overlay[(gt_final == 1) & (pred_np == 0)] = [255, 0, 0]      # FN - Red
    overlay[(gt_final == 0) & (pred_np == 1)] = [0, 0, 255]      # FP - Blue
    overlay[(gt_final == 1) & (pred_np == 1)] = [0, 255, 0]      # TP - Green

    axs[2].imshow(overlay)
    axs[2].set_title("Overlay (TP:Green, FP:Blue, FN:Red)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{base}_eval_plot.png"))
    plt.close()

# -------------------------------
# SAVE CSV
# -------------------------------
df = pd.DataFrame(results)

mean_row = {
    "Image": "Average",
    "Accuracy": df["Accuracy"].mean(),
    "Sensitivity": df["Sensitivity"].mean(),
    "Specificity": df["Specificity"].mean(),
    "Dice": df["Dice"].mean()
}
df.loc[len(df)] = mean_row

df.to_csv(CSV_OUT, index=False)
print(f"\n✅ Results saved to: {CSV_OUT}")
print(f"📸 Plots saved in: {PLOT_DIR}")
