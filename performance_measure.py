import numpy as np

def performance_measure(final_image, target_image):
    """
    Measure segmentation performance: Sensitivity, Specificity, Accuracy.

    Args:
        final_image (ndarray): Binary segmentation result (0 or 1).
        target_image (ndarray): Ground truth (0 or 1), may be grayscale initially.

    Returns:
        Se (float): Sensitivity
        Sp (float): Specificity
        Acc (float): Accuracy
    """
    # Ensure binary format
    final_bin = (final_image > 0).astype(np.uint8)
    target_bin = (target_image > 127).astype(np.uint8)  # handles grayscale

    TP = np.sum((final_bin == 1) & (target_bin == 1))
    FP = np.sum((final_bin == 1) & (target_bin == 0))
    FN = np.sum((final_bin == 0) & (target_bin == 1))
    TN = np.sum((final_bin == 0) & (target_bin == 0))

    Se = TP / (TP + FN + 1e-8)
    Sp = TN / (TN + FP + 1e-8)
    Acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return Se, Sp, Acc
