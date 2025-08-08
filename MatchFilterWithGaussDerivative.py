import numpy as np
from scipy.ndimage import convolve, uniform_filter
from skimage.morphology import square, closing
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing

from MatchFilterAndGaussDerKernel import MatchFilterAndGaussDerKernel

def MatchFilterWithGaussDerivative(img, sigma, yLength, numOfDirections, mask, c_value, t):
    """
    Retinal vessel extraction using matched filter + first-order Gaussian derivative.
    """
    if img.dtype != np.float64 and img.dtype != np.float32:
        img = img.astype(np.float64)

    rows, cols = img.shape
    MatchFilterRes = np.zeros((rows, cols, numOfDirections))
    GaussDerivativeRes = np.zeros((rows, cols, numOfDirections))

    # Filter in multiple orientations
    for i in range(numOfDirections):
        theta = np.pi / numOfDirections * i

        match_kernel = MatchFilterAndGaussDerKernel(sigma, yLength, theta, deriv_flag=0)
        gauss_deriv_kernel = MatchFilterAndGaussDerKernel(sigma, yLength, theta, deriv_flag=1)

        MatchFilterRes[:, :, i] = convolve(img, match_kernel, mode='reflect')
        GaussDerivativeRes[:, :, i] = convolve(img, gauss_deriv_kernel, mode='reflect')

    # Max response
    maxMatchFilterRes = np.max(MatchFilterRes, axis=2)
    maxGaussDerivativeRes = np.max(GaussDerivativeRes, axis=2)

    D = maxGaussDerivativeRes

    # Smooth D using 31x31 average filter
    Dm = uniform_filter(D, size=31)
    Dm = Dm - np.min(Dm)
    Dm = Dm / np.max(Dm)

    H = maxMatchFilterRes
    muH = np.mean(H)
    Tc = c_value * muH
    T = (1 + Dm) * Tc

    Mh = H >= T
    vess = Mh & (mask > 0)

    # Morphological cleanup
    vess = binary_closing(vess, square(3))
    
    # Remove small regions
    labeled_vessels, num = label(vess, connectivity=2, return_num=True)
    props = regionprops(labeled_vessels)

    # Keep only regions with area > t
    keep_labels = {prop.label for prop in props if prop.area > t}
    vess_cleaned = np.isin(labeled_vessels, list(keep_labels))

    return vess_cleaned.astype(np.uint8)
