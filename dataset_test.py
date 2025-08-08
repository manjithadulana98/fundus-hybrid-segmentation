import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from retinal_vessel_seg import retinal_vessel_seg

def dataset_test(file_path_im, file_path_manual, im_postfix, ma_postfix):
    # List and sort files by extension (case-insensitive)
    img_path_list_im = sorted([f for f in os.listdir(file_path_im) if f.lower().endswith(im_postfix.lower())])
    img_path_list_manual = sorted([f for f in os.listdir(file_path_manual) if f.lower().endswith(ma_postfix.lower())])

    if len(img_path_list_im) != len(img_path_list_manual):
        raise ValueError("Mismatch in number of images and manual masks!")

    img_num = len(img_path_list_im)
    results = np.zeros((img_num + 1, 4))  # 20 rows + 1 for average

    sum_Acc = sum_Se = sum_Sp = sum_Dice = 0

    # Create output directory
    output_dir = "segmentation_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(img_num):
        image_path = os.path.join(file_path_im, img_path_list_im[i])
        manual_path = os.path.join(file_path_manual, img_path_list_manual[i])

        # Read test image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Read manual mask using Pillow
        try:
            manual = Image.open(manual_path).convert('L')
            manual = np.array(manual)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load manual mask: {manual_path}\n{e}")

        # Run segmentation
        Se, Sp, Acc, Dice, im_final = retinal_vessel_seg(image, manual)

        # Save segmentation result
        base_name = os.path.splitext(img_path_list_im[i])[0]
        output_path = os.path.join(output_dir, f"{base_name}_segmentation.png")
        cv2.imwrite(output_path, (im_final * 255).astype(np.uint8))

        # Record performance
        results[i] = [Acc, Se, Sp, Dice]
        sum_Acc += Acc
        sum_Se += Se
        sum_Sp += Sp
        sum_Dice += Dice

    # Average row
    results[img_num] = [sum_Acc / img_num, sum_Se / img_num, sum_Sp / img_num, sum_Dice / img_num]

    # Save results to CSV
    df = pd.DataFrame(results, columns=["Accuracy", "Sensitivity", "Specificity", "Dice"])
    df.to_csv("Performance.csv", index=False)
    print("✔ Segmentation results saved to:", output_dir)
    print("✔ Performance metrics saved to: Performance.csv")
