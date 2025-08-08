import numpy as np
from scipy.ndimage import convolve

def combine_thin_vessel(im_thin_vess, im_sel):
    """
    Combines thin and thick vessel candidates.
    
    Args:
        im_thin_vess (ndarray): Thin vessel binary image (0 or 1).
        im_sel (ndarray): Thick vessel candidate binary image (0 or 1).
    
    Returns:
        im_final (ndarray): Final combined vessel map.
    """
    im_thin_vess = im_thin_vess.astype(np.uint8)
    im_sel = im_sel.astype(np.uint8)
    
    # Copy thin vessel map to start with
    im_final = np.copy(im_thin_vess)

    # Define 3x3 kernel excluding center pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Count neighbors around each pixel in thin vessel map
    neighbor_sum = convolve(im_thin_vess, kernel, mode='constant', cval=0)

    # Add thick vessel candidates only if thin vessel is nearby
    condition = (im_sel != 0) & (neighbor_sum > 0)
    im_final[condition] = im_sel[condition]

    return im_final
