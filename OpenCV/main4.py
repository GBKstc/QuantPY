import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'images', 'cv8.jpeg')

img = cv2.imread(img_path,0) # 读取灰度图像

img_float32 = np.float32(img) # 将图像转换为32位浮点型，为DFT做准备
# 进行离散傅里叶变换(DFT)
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT) # 得到复数输出
dft_shift = np.fft.fftshift(dft) # 将频谱移动到中心位置，低频部分在中心，高频部分在四周

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()

# 得到灰度图能表示的形式
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()