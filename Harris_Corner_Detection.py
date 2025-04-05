import matplotlib.pyplot as plt
import numpy as np
import cv2
# %matplotlib inline

from scipy import ndimage
import os

img_path = os.path.join('./original.jpg')
img = cv2.imread(img_path)
img_copy = np.copy(img)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_smooth(size, sigma=1):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Gaussian Smoothing                                     #
    #   Input: window size, sigma                                          #
    #   Output: smoothing image                                            #
    ########################################################################
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    img = kernel / np.sum(kernel)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return img

from scipy.ndimage.filters import convolve
img_filtered_K5 = convolve(img_Gray, gaussian_smooth(size=5,sigma=5))
img_filtered_K10 = convolve(img_Gray, gaussian_smooth(size=10,sigma=5))

def sobel_edge_detection(im,sigma):
    ########################################################################
    # TODO:                                                                #
    #   Perform the sobel edge detection                                   #
    #   Input: image after smoothing                                       #
    #   Output: the magnitude and direction of gradient                    #
    ########################################################################
    
    # Define Sobel filters
    Kx = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]])
    
    Ix = ndimage.convolve(im, Kx)
    Iy = ndimage.convolve(im, Ky)
    
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    gradient_direction = np.arctan2(Iy, Ix)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return  (gradient_magnitude, gradient_direction)


def structure_tensor(gradient_magnitude, gradient_direction, k, sigma):
    ########################################################################
    # TODO:                                                                #
    #   Perform the cornermess response                                    #
    #   Input: gradient_magnitude, gradient_direction                      #
    #   Output: second-moment matrix of Structure Tensor                   #
    ########################################################################
    
    Ix2 = gradient_magnitude ** 2 * np.cos(gradient_direction) ** 2
    Iy2 = gradient_magnitude ** 2 * np.sin(gradient_direction) ** 2
    Ixy = gradient_magnitude ** 2 * np.cos(gradient_direction) * np.sin(gradient_direction)
    
    Sxx = ndimage.gaussian_filter(Ix2, sigma)
    Syy = ndimage.gaussian_filter(Iy2, sigma)
    Sxy = ndimage.gaussian_filter(Ixy, sigma)
    
    det_M = Sxx * Syy - Sxy ** 2
    trace_M = Sxx + Syy
    
    StructureTensor = det_M / (trace_M + 1e-6)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return  StructureTensor

def NMS(harrisim,window_size=30,threshold=0.1):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Non-Maximum Suppression                                #
    #   Input: Structure Tensor, window size, threshold                    #
    #   Output: filtered coordinators                                      #
    ########################################################################
    # 以圖像最大值與閾值係數定義閾值
    harrisim_max = harrisim.max()
    harrisim_threshold = harrisim_max * threshold

    # 使用 maximum_filter 取得每個局部窗口內的最大值
    max_filtered = ndimage.maximum_filter(harrisim, size=window_size)

    # 建立 mask：只有響應值大於閾值且等於局部最大值的像素才被保留
    nms_mask = (harrisim == max_filtered) & (harrisim > harrisim_threshold)

    # 取得符合條件的角點座標
    filtered_coords = np.argwhere(nms_mask)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    plt.figure()
    plt.gray()
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0]for p in filtered_coords],'+')
    plt.axis('off')
    plt.show()
    
def rotate(image, angle, center = None, scale = 1.0):

    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated