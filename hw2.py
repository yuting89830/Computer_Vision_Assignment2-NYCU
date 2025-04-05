import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage import filters

from Harris_Corner_Detection import gaussian_smooth, sobel_edge_detection, structure_tensor, NMS, rotate

sigma=1
threshold=0.3
k=0.04
angle=30

if __name__ == '__main__':
    img_path = os.path.join('./original.jpg')
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# A. Harris corner detection with original image
# i. Gaussian smooth results: 𝜎=5 and kernel size=5 and 10 (2 images)

    img_filtered_K5 = convolve(img_Gray, gaussian_smooth(size=5,sigma=5))
    img_filtered_K10 = convolve(img_Gray, gaussian_smooth(size=10,sigma=5))
    img_filtered_K5 =  img_filtered_K5 / np.amax(img_filtered_K5) * 255
    img_filtered_K10 =  img_filtered_K10 / np.amax(img_filtered_K10) * 255
    save_img_path1 = os.path.join('./results/Gaussian smooth results', 'gaussian_smooth_of_sigma_and_kernal_size_5.jpg')
    save_img_path2 = os.path.join('./results/Gaussian smooth results', 'gaussian_smooth_of_sigma_and_kernal_size_10.jpg')
    cv2.imwrite(save_img_path1, img_filtered_K5)
    cv2.imwrite(save_img_path2, img_filtered_K10)

# ii. Sobel edge detection results
# (1) magnitude of gradient (Gaussian kernel size=5 and 10) (2 images)
# (2) direction of gradient (Gaussian kernel size=5 and 10) (2 images)    
    
    
    gradient_magnitude_K5, gradient_direction_K5 = sobel_edge_detection(img_filtered_K5,sigma)
    gradient_magnitude_K10, gradient_direction_K10 = sobel_edge_detection(img_filtered_K10,sigma)
    gradient_magnitude_K5 =  gradient_magnitude_K5 / np.amax(gradient_magnitude_K5) * 255
    gradient_magnitude_K10 =  gradient_magnitude_K10 / np.amax(gradient_magnitude_K10) * 255
    gradient_direction_K5 =  gradient_direction_K5 / np.amax(gradient_direction_K5) * 255
    gradient_direction_K10 =  gradient_direction_K10 / np.amax(gradient_direction_K10) * 255
    save_img_path3 = os.path.join('./results/Sobel edge detection results', 'magnitude_of_gradient_kernel_size_5.jpg')
    save_img_path4 = os.path.join('./results/Sobel edge detection results', 'magnitude_of_gradient_kernel_size_10.jpg')
    save_img_path5 = os.path.join('./results/Sobel edge detection results', 'direction_of_gradient_kernel_size_5.jpg')
    save_img_path6 = os.path.join('./results/Sobel edge detection results', 'derection_of_gradient_kernel_size_10.jpg')
    cv2.imwrite(save_img_path3, gradient_magnitude_K5)
    cv2.imwrite(save_img_path4, gradient_magnitude_K10)
    cv2.imwrite(save_img_path5, gradient_direction_K5)
    cv2.imwrite(save_img_path6, gradient_direction_K10)
    
# iii. Structure tensor + NMS results (Gaussian kernel size=10)
# (1) window size = 3x3 (1 image)
# (2) window size = 30x30 (1 image)
    
    
    harrisim=structure_tensor(gradient_magnitude_K10, gradient_direction_K10, k, sigma)
    # # ========== 顯示原始 Structure Tensor （不使用 NMS）==========
    # plt.figure(); plt.gray(); plt.figure(figsize=(20, 10))
    # plt.imshow(harrisim, cmap='hot')
    # plt.title('Structure Tensor Response (No NMS)')
    # plt.axis('off')
    # plt.savefig("./results/Structure_tensor/harrisim_raw.jpg")

    # # ========== 加入 Threshold 過濾，但不使用 NMS ==========
    # threshold_value = 0.01 * harrisim.max()
    # points_raw = np.argwhere(harrisim > threshold_value)

    # plt.figure(); plt.gray(); plt.figure(figsize=(20, 10))
    # plt.imshow(img_copy)
    # plt.plot(points_raw[:, 1], points_raw[:, 0], '+')
    # plt.title('Structure Tensor Threshold Only (No NMS)')
    # plt.axis('off')
    # plt.savefig("./results/Structure_tensor/harrisim_threshold_only.jpg")

    window_size=3
    NMS_W3=NMS(harrisim,window_size,threshold)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    # black_bg = np.zeros_like(harrisim)
    # plt.imshow(black_bg, cmap='gray')
    plt.imshow(img_copy)
    plt.plot([p[1] for p in NMS_W3],[p[0]for p in NMS_W3],'+')
    plt.axis('off')
    plt.savefig("./results/Structure tensor + NMS results/NMS_window_size_3.jpg")
    
    window_size=30
    NMS_W30=NMS(harrisim,window_size,threshold)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    black_bg = np.zeros_like(harrisim)
    plt.imshow(black_bg, cmap='gray')
    plt.imshow(img_copy)
    plt.plot([p[1] for p in NMS_W30],[p[0]for p in NMS_W30],'+')
    plt.axis('off')
    plt.savefig("./results/Structure tensor + NMS results/NMS_window_size_30.jpg")
    
# B. Final results of rotating (by 30°) original images (1 image)    
    
    img_Gray_30 = rotate(img_Gray, angle)
    img_filtered_K10_R30 = convolve(img_Gray_30, gaussian_smooth(size=10,sigma=5))
    img_filtered_K10_R30 =  img_filtered_K10_R30 / np.amax(img_filtered_K10_R30) * 255
    gradient_magnitude_K10_R30, gradient_direction_K10_R30 = sobel_edge_detection(img_filtered_K10_R30,sigma)
    gradient_magnitude_K10_R30 =  gradient_magnitude_K10_R30 / np.amax(gradient_magnitude_K10_R30) * 255
    gradient_direction_K10_R30 =  gradient_direction_K10_R30 / np.amax(gradient_magnitude_K10_R30) * 255
    harrisim_R30=structure_tensor(gradient_magnitude_K10_R30, gradient_direction_K10_R30, k, sigma)
    where_are_nan = np.isnan(harrisim_R30)
    harrisim_R30[where_are_nan] = 0
    window_size=3
    NMS_W3_R30=NMS(harrisim_R30,window_size,threshold)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    # black_bg = np.zeros_like(harrisim_R30)
    # plt.imshow(black_bg, cmap='gray')
    plt.imshow(img_Gray_30)
    plt.plot([p[1] for p in NMS_W3_R30],[p[0]for p in NMS_W3_R30],'+')
    plt.axis('off')
    plt.savefig("./results/Final results of rotating/Rotate_30.jpg")
    
# C. Final results of scaling (to 0.5x) original images (1 image)
    
    img_Gray_scaled = cv2.resize(img_Gray,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    img_filtered_K10_scaled = convolve(img_Gray_scaled, gaussian_smooth(size=10,sigma=5))
    img_filtered_K10_scaled =  img_filtered_K10_scaled / np.amax(img_filtered_K10_scaled) * 255
    gradient_magnitude_K10_scaled, gradient_direction_K10_scaled = sobel_edge_detection(img_filtered_K10_scaled,sigma)
    gradient_magnitude_K10_scaled =  gradient_magnitude_K10_scaled / np.amax(gradient_magnitude_K10_scaled) * 255
    gradient_direction_K10_scaled =  gradient_direction_K10_scaled / np.amax(gradient_direction_K10_scaled) * 255
    harrisim_scaled=structure_tensor(gradient_magnitude_K10_scaled, gradient_direction_K10_scaled, k, sigma)
    NMS_W3_scaled=NMS(harrisim_scaled,window_size,threshold)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    # black_bg = np.zeros_like(harrisim)
    # plt.imshow(black_bg, cmap='gray')
    plt.imshow(img_Gray_scaled)
    plt.plot([p[1] for p in NMS_W3_scaled],[p[0]for p in NMS_W3_scaled],'+')
    plt.axis('off')
    plt.savefig("./results/Final results of scaling/Scaling.jpg")
    
    