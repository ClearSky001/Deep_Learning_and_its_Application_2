#!/usr/bin/env python
# coding: utf-8

# # OpenCV를 이용한 이미지 처리 기초
# - OpenCV는 컴퓨터로 이미지나 영상을 읽고, 이미지의 사이즈 변환이나 회전, 선분 및 도형 그리
# 기, 채널 분리 등의 연산을 처리할 수 있도록 만들어진 오픈 소스 라이브러리로, 이미지 처리 분야
# 에서 가장 많이 사용된다

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# 이미지 크기는 512x512이고 3개 채널로 이루어져 있다

# 이미지 픽셀당 8bit이다 (24비트 트루컬러)

# In[5]:


def show_image(img, method = "plt"):
    if method == "plt":
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    elif method == "cv2":
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif method == "colab":
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)

def show_image_list(title_to_img_dict, figsize):
    n = len(title_to_img_dict)
    fig, axes = plt.subplots(1, n, figsize=figsize)


    for i, (title, img) in enumerate(title_to_img_dict.items()):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        # axes[i].axis('off')

    # plt.tight_layout()
    plt.show()


# 이미지 크기를 리사이즈 할수 있다

# 이미지를 상하/좌우로 뒤집을 수 있다

# array slicing으로 이미지의 일부 영역을 가져올 수 있다

# <mark>과제</mark> 이미지의 특정 사각 영역을 전달받은 색상값 fill_color로 채워넣는 cutout함수를 작성하라 (cutout augmentation에 쓰임) 

# In[11]:


def cutout_image(src_img, top_left_corner, bottom_right_corner, fill_color):
    """
    이 함수는 이미지의 사각 영역을 fill_color값으로 채운다. 시작점과 끝점은 top_left_corner, bottom_right_corner 변수에 의해 지정된다.
    row -> y축, col -> x축에 대응되며, cv2.imread함수는 dim 0가 y축, dim 1이 x축에 대응됨에 주의할것.
    힌트: array slicing을 이용할것

    Args:
        src_img (numpy.ndarray): The source image to modify.
        top_left_corner (tuple): The (row, col) coordinates of the top-left corner of the cutout region.
        bottom_right_corner (tuple): The (row, col) coordinates of the bottom-right corner of the cutout region.
        fill_color (int or tuple): The color or intensity value to fill the cutout region with. 
                                   For grayscale images, use an integer value. For RGB images, use a tuple of 3 values.

    Returns:
        numpy.ndarray: The modified image with the cutout applied.
    """

    modified_img = src_img.copy() # src_img를 보존하기 위해 deep copy를 수행한다

    ##### YOUR CODE START #####
    
    top, left = top_left_corner
    bottom, right = bottom_right_corner

    if len(modified_img.shape) == 2:
        modified_img[top:bottom, left:right] = fill_color

    else:
        modified_img[top:bottom, left:right] = fill_color
    
    ##### YOUR CODE END #####
    
    return(modified_img)



# 선이나 도형을 그릴 수 있다

# In[13]:


def draw_line_on_image(src_img):
    modified_img = src_img.copy()
    modified_img = cv2.line(modified_img, pt1 = (100,50), pt2 = (300,200), color = (255,0,0), thickness = 3)
    return(modified_img)


# In[15]:


def draw_circle_on_image(src_img):
    modified_img = src_img.copy()
    modified_img = cv2.circle(modified_img, center = (150, 150), radius = 50, color = (0,0,0), thickness = 2)
    return modified_img


# In[17]:


def draw_poly_on_image(src_img):
    modified_img = src_img.copy()

    points = np.array([[256, 100], [100, 200], [150, 400], [362, 400]])
    
    modified_img = cv2.fillPoly(modified_img, [points], color = (255, 255, 255), lineType = 1)
    return modified_img


# ### Affine transformation and perspective transformation
# 1. 어파인 변환 (Affine transformation): 점, 직선, 평면을 보존하는 선형 변환으로 변환 전에 평행이였던 선들은 변환 후에도 평행성이 보존된다.
# 
# $\begin{pmatrix}x'\\y'\\ \end{pmatrix}$ = $\begin{pmatrix}a&b\\c&d\\ \end{pmatrix}$ $\begin{pmatrix}x\\y\\ \end{pmatrix}$ + $\begin{pmatrix}tx\\ty\\ \end{pmatrix}$ 
# 
# 2. 원근 변환 (Perspective transformation): 이미지를 다른 관점에서 보는 것처럼 변환한다.
# 
# $\begin{pmatrix}x'\\y'\\w'\\ \end{pmatrix}$ = $\begin{pmatrix}a&b&c\\d&e&f\\g&h&1\\ \end{pmatrix}$ $\begin{pmatrix}x\\y\\1\\ \end{pmatrix}$ 

# In[19]:


def rotate_image(src_img, degree):
    """
    이미지를 degree만큼 회전시킨다.
    가장자리는 검은색으로 채워 넣는다
    """

    height, width, channel = src_img.shape
    aff_matrix = cv2.getRotationMatrix2D(center = (width/2, height/2), angle = degree, scale = 1)
    print(f"Affine transform matrix is: ")
    print(aff_matrix)
    img_rotated = cv2.warpAffine(src = src_img, M = aff_matrix, dsize = (width, height), 
                                 borderValue=(0,0,0))
    return(img_rotated)


# In[21]:


def perspective_transform_image(src_img):
    ordered_corners = np.array([[57, 630], [936, 330], [1404, 792], [550, 1431]], dtype='float32')

    # 너비와 높이 계산
    ordered_width = int(max(np.linalg.norm(ordered_corners[0] - ordered_corners[1]), 
                            np.linalg.norm(ordered_corners[2] - ordered_corners[3]))) 
    ordered_height = int(max(np.linalg.norm(ordered_corners[0] - ordered_corners[3]), 
                            np.linalg.norm(ordered_corners[1] - ordered_corners[2])))
    # 변환이 될 꼭짓점 좌표 지정
    ordered_rect_corners = np.array([[0, 0], [ordered_width, 0], [ordered_width, ordered_height], [0, ordered_height]], dtype='float32')

    # 호모그래피 행렬 계산
    ordered_scan_matrix = cv2.getPerspectiveTransform(ordered_corners, ordered_rect_corners)
    # 원근 변환 다시 적용
    ordered_scanned_image = cv2.warpPerspective(src_img, ordered_scan_matrix, (ordered_width, ordered_height))
    return(ordered_scanned_image)


# # Image Processing

# ### Gaussian Filter

# In[23]:


def add_gaussian_noise(src_img, mean = 0, sigma = 1):
    gaussian_noise=np.random.normal(mean, sigma, src_img.shape).astype('float32')
    
    noisy_image = src_img.astype('float32') + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype('uint8')
    return(noisy_image)


# In[24]:


def gaussian_kernel(size, sigma=1):
    """
    Generates a Gaussian kernel.
    
    Args:
        size (int): The size of the kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian function.
        
    Returns:
        numpy.ndarray: The Gaussian kernel.
    """
    k = (size - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g / g.sum()


# <mark>과제</mark> 커널 사이즈와 표준편차를 전달받아 전달받은 이미지에 가우시안 블러를 적용하는 함수를 작성하라.
# 
# cv2.filter2D 함수를 이용할것.

# Function: `cv2.filter2D(src, ddepth, kernel, ...)`
# 
# **Arguments:**
# 
# - **`src`**: 필터를 적용할 소스 이미지.
# - **`ddepth`**: 출력 이미지의 깊이 (-1일 경우 입력이미지와 동일하게 만듬)
# - **`kernel`**: 커널로 사용할 2차원 행렬

# In[28]:


def apply_gaussian_blur(src_img, kernel_size=5, sigma=1):
    """
    가우시안 커널을 계산하여 이미지에 적용한다

    Args:
        src_img (numpy.ndarray): source 이미지
        kernel_size (int): 가우시안 커널의 크기 (홀수여야 함).
        sigma (float): 가우시안 커널을 계산할때 사용할 표준편자 값
    
    Returns:
        numpy.ndarray: The blurred image.
    """
    
    ##### YOUR CODE START #####

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    gaussian_kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

    blurred_img = cv2.filter2D(src_img, -1, gaussian_kernel_2d)

    return blurred_img

    ##### YOUR CODE END #####
    


# Median filter를 사용하면 salt & pepper noise를 효율적으로 제거할 수 있다

# In[ ]:


def add_salt_noise(image):
    num_salt = np.ceil(0.05 * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]

    salted_image = image.copy()
    salted_image[coords[0], coords[1]] = 255
    
    return salted_image


# In[ ]:


def add_pepper_noise(image):
    num_pepper = np.ceil(0.05 * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]

    peppered_image = image.copy()
    peppered_image[coords[0], coords[1]] = 0
    return peppered_image


# 소벨 필터를 이용한 edge 검출

# 그 외에도 다양한 엣지검출기들이 있다
