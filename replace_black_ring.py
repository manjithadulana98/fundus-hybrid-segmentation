import numpy as np

def replace_black_ring(im_enh, im_mask):
    """
    Replace the black background of the fundus image with the mean of
    3 randomly selected (50x50) background regions.
    
    Args:
        im_enh (ndarray): Enhanced grayscale image (2D, float32).
        im_mask (ndarray): Binary mask (2D, float or bool).
    
    Returns:
        im_new (ndarray): Image with black ring replaced.
        mean_val (float): Mean intensity from sampled background regions.
    """

    row, col = im_mask.shape
    area_sum = np.zeros((50, 50), dtype=np.float32)

    # Generate 3 random positions within the image ensuring 50x50 patch fits
    np.random.seed(0)  # Optional: for reproducibility
    max_offset = int(1/3 * min(row, col))
    posit = np.ceil((np.random.rand(3, 2) + 1) * max_offset).astype(int)

    for i in range(3):
        x, y = posit[i]
        # Ensure patch stays in bounds
        x = np.clip(x, 25, row - 25)
        y = np.clip(y, 25, col - 25)
        area_rand = im_enh[x - 25:x + 25, y - 25:y + 25]
        area_sum += area_rand

    area_sum = area_sum / 3.0
    mean_val = np.mean(area_sum)

    mean_mask = (~im_mask.astype(bool)).astype(float) * mean_val
    im_new = mean_mask + im_enh * im_mask

    return im_new, mean_val
