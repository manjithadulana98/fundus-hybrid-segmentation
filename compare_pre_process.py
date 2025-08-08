import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
# Update these paths as needed
image_id = "01"  # Change for other samples

original_img_path = f"DRIVE/test/images/{image_id}_test.tif"
manual_gt_path = f"DRIVE/test/1st_manual/{image_id}_manual1.gif"
rule_based_output_path = f"segmentation_outputs/{image_id}_test_segmentation.png"

# -------------------------------
# LOAD IMAGES
# -------------------------------

# Original fundus image
original_img = Image.open(original_img_path).convert("RGB")

# Manual ground truth (binary mask)
manual_mask = Image.open(manual_gt_path).convert("L")
manual_mask_np = np.array(manual_mask)
manual_mask_bin = (manual_mask_np > 127).astype(np.uint8)

# Rule-based segmentation output (binarized if not already)
rule_img = Image.open(rule_based_output_path).convert("L")
rule_np = np.array(rule_img)
rule_mask_bin = (rule_np > 127).astype(np.uint8)

# Resize masks to match original image if needed
if manual_mask.size != original_img.size:
    manual_mask = manual_mask.resize(original_img.size, Image.NEAREST)
    manual_mask_bin = (np.array(manual_mask) > 127).astype(np.uint8)

if rule_img.size != original_img.size:
    rule_img = rule_img.resize(original_img.size, Image.NEAREST)
    rule_mask_bin = (np.array(rule_img) > 127).astype(np.uint8)

# -------------------------------
# PLOT COMPARISON
# -------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(original_img)
axs[0].set_title("Original Fundus Image")
axs[0].axis("off")

axs[1].imshow(manual_mask_bin, cmap='gray')
axs[1].set_title("Manual Annotation (GT)")
axs[1].axis("off")

axs[2].imshow(rule_mask_bin, cmap='gray')
axs[2].set_title("Rule-Based Segmentation")
axs[2].axis("off")

plt.tight_layout()
# Save the figure
os.makedirs("comparison_plots", exist_ok=True)
plot_path = f"comparison_plots/{image_id}_comparison.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"\n✅ Plot saved to: {plot_path}")
