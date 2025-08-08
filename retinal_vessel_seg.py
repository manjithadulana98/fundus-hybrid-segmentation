import cv2
import numpy as np
from skimage import exposure, morphology, filters, measure

from replace_black_ring import replace_black_ring
from vessel_point_selected import vessel_point_selected
from MatchFilterWithGaussDerivative import MatchFilterWithGaussDerivative
from combine_thin_vessel import combine_thin_vessel
from performance_measure import performance_measure

def retinal_vessel_seg(image, manual):
    # Convert to float [0,1] range
    im_rgb = image.astype(np.float32) / 255.0
    
    # Create mask based on green channel
    im_mask = im_rgb[:, :, 1] > (20 / 255.0)
    im_mask = morphology.erosion(im_mask, morphology.disk(3)).astype(float)

    # Extract green channel
    im_green = im_rgb[:, :, 1]

    # CLAHE
    im_enh = exposure.equalize_adapthist(im_green, kernel_size=(8, 8), nbins=128)

    # Replace black ring
    im_enh1, mean_val = replace_black_ring(im_enh, im_mask)

    # Negative image
    im_gray = 1.0 - im_enh1

    # Top-hat transform
    selem = morphology.disk(10)
    im_top = im_gray - morphology.opening(im_gray, selem)

    # OTSU Thresholding
    level = filters.threshold_otsu(im_top)
    im_thre = (im_top > level) & (im_mask > 0)

    # Remove small connected components
    im_rmpix = morphology.remove_small_objects(im_thre, min_size=100, connectivity=2)

    # Vessel point selection (for thick vessels)
    im_sel = vessel_point_selected(im_gray, im_rmpix, mean_val)

    # Thin vessel extraction using matched filter and FDoG
    im_thin_vess = MatchFilterWithGaussDerivative(im_enh, 1, 4, 12, im_mask, 2.3, 30)

    # Combine both
    im_final = combine_thin_vessel(im_thin_vess, im_sel)

    # Evaluation
    Se, Sp, Acc = performance_measure(im_final, manual)

    manual_bin = manual > 127  # threshold if grayscale
    Dice = 2 * np.sum(im_final & manual_bin) / (np.sum(im_final) + np.sum(manual_bin) + 1e-8)

    return Se, Sp, Acc, Dice, im_final
