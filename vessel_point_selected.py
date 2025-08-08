import numpy as np
from scipy.ndimage import median_filter

def vessel_point_selected(im_gray, im_thre, mean_val):
    """
    Select vessel points based on intensity proximity to max intensity
    and background mean.
    
    Args:
        im_gray (ndarray): Grayscale image (float32 or float64).
        im_thre (ndarray): Binary thresholded image (0 or 1).
        mean_val (float): Background mean or green channel average.
    
    Returns:
        im_sel (ndarray): Binary image of selected vessels.
    """
    im_gray = np.asarray(im_gray)
    im_thre = np.asarray(im_thre)
    
    row, col = im_gray.shape
    im_sel = np.zeros_like(im_gray, dtype=np.uint8)

    p_max = np.max(im_gray)
    p_min = mean_val

    # Compare each thresholded pixel's intensity to p_max and p_min
    vessel_mask = (im_thre != 0) & (
        np.abs(im_gray - p_max) < np.abs(im_gray - p_min)
    )
    im_sel[vessel_mask] = 1

    # Apply 3x3 median filter to threshold image
    im_med = median_filter(im_thre.astype(np.uint8), size=(3, 3))

    # Combine selected and filtered pixels
    im_sel = np.logical_or(im_sel, im_med).astype(np.uint8)

    return im_sel
