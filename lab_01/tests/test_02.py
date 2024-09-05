import pytest
import numpy as np
import cv2
import sys
from lab_01_basic_image_processing import apply_gaussian_blur

def apply_gaussian_blur_answer(src_img, kernel_size=5, sigma=1):
    blurred_img = cv2.GaussianBlur(src_img, (kernel_size, kernel_size), sigma)

    return blurred_img
 

def check_image_diff_ratio(img1, img2):
    diff_sum =  np.absolute(img1.astype(int) - img2.astype(int)).sum()
    diff_ratio = diff_sum / (img1.shape[0] * img1.shape[1])
    return(diff_ratio)



def test_apply_gaussian_blur_shape_score_25():
    """
    Test that the blurred image has the same shape as the original image.
    """
    src_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    blurred_img = apply_gaussian_blur(src_img, kernel_size=5, sigma=1)
    
    assert src_img.shape == blurred_img.shape, "The shape of the blurred image should match the original image."


def test_apply_gaussian_blur_value_score_25():
    """
    Test that applying different kernel sizes and sigma values works correctly.
    """
    # Create a dummy image (e.g., 100x100 grayscale)
    src_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Test different kernel sizes and sigma values
    kernel_sizes = [3, 5, 7]
    sigmas = [0.5, 1, 2]
    
    for kernel_size in kernel_sizes:
        for sigma in sigmas:
            blurred_img = apply_gaussian_blur(src_img, kernel_size=kernel_size, sigma=sigma)
            blurred_img_ans = apply_gaussian_blur_answer(src_img, kernel_size=kernel_size, sigma=sigma)
            diff_ratio = check_image_diff_ratio(blurred_img, blurred_img_ans)

            assert diff_ratio < 1, f"Blurring result should be correcct"
