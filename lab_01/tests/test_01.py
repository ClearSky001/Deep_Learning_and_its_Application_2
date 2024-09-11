import pytest
import numpy as np
import cv2
import sys
from lab_01_basic_image_processing import cutout_image

def cutout_image_answer(src_img, top_left_corner, bottom_right_corner, fill_color):
    modified_img = src_img.copy()
    cv2.rectangle(modified_img, (top_left_corner[1], top_left_corner[0]), (bottom_right_corner[1], bottom_right_corner[0]), fill_color, thickness=-1)

    return modified_img

def check_image_diff_ratio(img1, img2):
    diff_sum =  np.absolute(img1.astype(int) - img2.astype(int)).sum()
    diff_ratio = diff_sum / (img1.shape[0] * img1.shape[1])
    return(diff_ratio)


def test_apply_cutout_image_1_score_25():
    """
    Test that applying different kernel sizes and sigma values works correctly.
    """
    # Create a dummy image (e.g., 100x100 grayscale)
    src_img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    

    img_test = cutout_image(src_img, (50,100), (300, 250), (0, 255, 0))
    img_ans = cutout_image_answer(src_img, (50,100), (300, 250), (0, 255, 0))
    diff_ratio = check_image_diff_ratio(img_test, img_ans)

    assert diff_ratio < 2, f"Cutout result should be correct"


def test_apply_cutout_image_2_score_25():
    """
    Test that applying different kernel sizes and sigma values works correctly.
    """
    # Create a dummy image (e.g., 100x100 grayscale)
    src_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    

    img_test = cutout_image(src_img, (34,53), (364, 457), (255, 0, 0))
    img_ans = cutout_image_answer(src_img, (34,53), (364, 457), (255, 0, 0))
    diff_ratio = check_image_diff_ratio(img_test, img_ans)

    assert diff_ratio < 2, f"Cutout result should be correct"

