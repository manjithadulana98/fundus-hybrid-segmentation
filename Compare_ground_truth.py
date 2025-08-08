import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_closing
from skimage.morphology import remove_small_objects

# -------------------------------
# CONFIGURATION
# -------------------------------
manual_type = "2nd_manual"  # or "1st_manual"
annotation_num = manual_type[0]
GT_DIR = f"DRIVE/test/{manual_type}"  # update this if your path differs
GT_VS_PRUNED_DIR = f"gt_vs_pruned_{manual_type}"
RESIZE_DIM = None
MIN_VESSEL_SIZE = 80

os.makedirs(GT_VS_PRUNED_DIR, exist_ok=True)

# -------------------------------
# VISUALIZATION LOOP
# -------------------------------
files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".gif")])

for filename in tqdm(files, desc="Saving GT vs Pruned comparisons"):
    base_num = filename.split("_")[0]
    gt_path = os.path.join(GT_DIR, filename)

    # Load original GT
    gt_img = Image.open(gt_path).convert("L")
    if RESIZE_DIM:
        gt_img = gt_img.resize(RESIZE_DIM, Image.NEAREST)
    gt_np = (np.array(gt_img) > 127).astype(np.uint8)

    # Prune GT
    gt_smooth = binary_closing(gt_np, structure=np.ones((3, 3)))
    gt_pruned = remove_small_objects(gt_smooth.astype(bool), min_size=MIN_VESSEL_SIZE)
    gt_final = gt_pruned.astype(np.uint8)

    # Plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(gt_np, cmap="gray")
    axs[0].set_title("Original Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(gt_final, cmap="gray")
    axs[1].set_title("Pruned Ground Truth")
    axs[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(GT_VS_PRUNED_DIR, f"{base_num}_gt_vs_pruned.png")
    plt.savefig(save_path)
    plt.close()

print(f"\n✅ Ground Truth comparison plots saved in: {GT_VS_PRUNED_DIR}")
