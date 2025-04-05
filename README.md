
# Homework 2 說明

## 1. 本次作業用到的Library: `os`, `cv2`, `numpy`, `matplotlib`, `scipy`
## 2. py 檔: `hw2.py` (執行檔), `Harris_Corner_Detection.py` (Function)
## 3. 若要執行程式, 產生圖片, 請輸入: `python hw2.py`
## 4. Function 包含:  `gaussian_smooth`, `sobel_edge_detection`, `structure_tensor`, `NMS`, `rotate`
## 5. results 內容包含5個子Folder, 分別為:
> ### (1) `Gaussian smooth results`: 2張圖片, 分別是 Gaussian smooth results: 𝜎=5 and kernel size=5 與 Gaussian smooth results: 𝜎=5 and kernel size=10 images)
> ### (2) `Sobel edge detection results`: 4張圖片, 分別是 magnitude of gradient (Gaussian kernel size=5 and 10) (2 images) 與 direction of gradient (Gaussian kernel size=5 and 10) (2 images)
> ### (3) `Structure tensor + NMS results`: 2張圖片, 分別是 window size = 3x3 與 window size = 30x30
> ### (4) `Final results of rotating`: 1張圖片, 內容為 Final results of rotating (by 30°) original images 
> ### (5) `Final results of scaling`: 1張圖片, 內容為 Final results of scaling (to 0.5x) original images
