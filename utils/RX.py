import numpy as np

def RX(hsi_img):
    """

    Mahalanobis Distance anomaly detector
    uses global image mean and covariance as background estimates

    Inputs:
     hsi_image - n_row x n_col x n_band hyperspectral image
     mask - binary image limiting detector operation to pixels where mask is true
            if not present or empty, no mask restrictions are used

    Outputs:
      dist_img - detector output image

    8/7/2012 - Taylor C. Glenn
    5/5/2018 - Edited by Alina Zare
    11/2018 - Python Implementation by Yutai Zhou
    """
    hsi_img = hsi_img.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
    hsi_img = (hsi_img-np.min(hsi_img))/(np.max(hsi_img)-np.min(hsi_img))
    n_row, n_col, n_band = hsi_img.shape
    n_pixels = n_row * n_col
    hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order='F').T

    mu = np.mean(hsi_data, 1)
    sigma = np.cov(hsi_data.T, rowvar=False)

    z = hsi_data - mu[:, np.newaxis]
    sig_inv = np.linalg.pinv(sigma)

    dist_data = np.zeros(n_pixels)
    for i in range(n_pixels):
        dist_data[i] = z[:, i].T @ sig_inv @ z[:, i]

    return dist_data.reshape([n_col, n_row]).T