import cv2
import numpy as np


def extract_prnu_residual(img):
    """
    Extract PRNU noise residual using high-pass filtering.
    Input: grayscale image (uint8)
    Output: noise residual (float32)
    """

    img = img.astype(np.float32)

    # Remove image content
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Noise residual
    prnu = img - blur

    # Zero-mean normalization
    prnu = prnu - np.mean(prnu)

    return prnu
