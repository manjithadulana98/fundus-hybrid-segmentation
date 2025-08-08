import numpy as np

def MatchFilterAndGaussDerKernel(sigma, YLength, theta, deriv_flag=0):
    """
    Create matched filter or first derivative of Gaussian (FDOG) kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian.
        YLength (int): Length of the filter in the y-direction.
        theta (float): Orientation in radians.
        deriv_flag (int): 0 = MF, 1 = FDOG.

    Returns:
        kernel (ndarray): 2D kernel for convolution.
    """
    width = int(np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2)))
    if width % 2 == 0:
        width += 1
    halfLength = (width - 1) // 2

    kernel = np.zeros((width, width), dtype=np.float64)

    for row, y in enumerate(range(halfLength, -halfLength - 1, -1)):
        for col, x in enumerate(range(-halfLength, halfLength + 1)):
            xPrime = x * np.cos(theta) + y * np.sin(theta)
            yPrime = y * np.cos(theta) - x * np.sin(theta)

            if abs(xPrime) > 3.5 * np.ceil(sigma) or abs(yPrime) > (YLength - 1) / 2:
                value = 0
            else:
                if deriv_flag == 0:  # Matched filter (negative Gaussian)
                    value = -np.exp(-0.5 * (xPrime / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
                else:  # FDOG (first derivative)
                    value = -np.exp(-0.5 * (xPrime / sigma) ** 2) * xPrime / (np.sqrt(2 * np.pi) * sigma ** 3)
            kernel[row, col] = value

    # Adjust mean for matched filter
    if deriv_flag == 0:
        negative_mask = kernel < 0
        negative_mean = np.sum(kernel) / np.sum(negative_mask)
        kernel[negative_mask] -= negative_mean

    return kernel
