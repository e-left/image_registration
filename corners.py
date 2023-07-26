import numpy as np
from scipy.ndimage import convolve, gaussian_filter

def isCorner(I, p, k, Rthres):
    sigma = 1
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolve(I, sobel_x)
    Iy = convolve(I, sobel_y)

    Ix2 = gaussian_filter(Ix**2, sigma)
    Iy2 = gaussian_filter(Iy**2, sigma)
    Ixy = gaussian_filter(Ix*Iy, sigma)

    Ix2 = Ix2[p[:, 0], p[:, 1]].astype(np.int64)
    Iy2 = Iy2[p[:, 0], p[:, 1]].astype(np.int64)
    Ixy = Ixy[p[:, 0], p[:, 1]].astype(np.int64)

    det = Ix2 * Iy2 - Ixy**2
    trace = Ix2 + Iy2
    response = det - k * trace**2

    return response > Rthres

def myDetectHarrisFeatures(I):
    width = I.shape[0]
    height = I.shape[1]

    k = 0.04
    Rthres = 10000

    corners = []

    x = range(width)
    y = range(height)

    # produce all image points
    p = np.array([[xt, yt] for xt in x for yt in y])

    corner_status = isCorner(I, p, k, Rthres)

    corners = p[corner_status]
    
    return corners