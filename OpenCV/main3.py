import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'images', 'cv8.jpeg')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
def imShowFN(img,name='image'):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#储存图片
def save_img(img,img_path):
    #判断img_path是地址路径 还是文件名
    if os.path.isdir(img_path):
        name = os.path.join(img_path,img)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        name = os.path.join(current_dir, 'images', img_path)
    cv2.imwrite(name,img)

img = cv2.imread(img_path) #读取图片
img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:,:] #灰度图
ret1, binaryInv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV) #二值化


equ = cv2.equalizeHist(img_gray)
plt.subplot(141)
plt.hist(img_gray.ravel(),256) # 均衡化前
plt.subplot(142)
plt.hist(equ.ravel(),256) # 均衡化后
plt.subplot(143)
plt.imshow(equ,'gray')
plt.subplot(144)
plt.imshow(img_gray,'gray')
plt.show()

# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255
# ## 显示一下这个掩码
# masked_img = cv2.bitwise_and(img, img, mask=mask)#与操作
# cv_show(masked_img,'masked_img')


# hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])


# # 创建两个子图
# plt.figure(figsize=(10,5))

# # 显示 cv2.calcHist 计算的直方图
# plt.subplot(121)
# plt.imshow(img_gray)
# plt.title('原图')
# plt.xlabel('像素值')
# plt.ylabel('像素数量')

# # 显示 plt.hist 计算的直方图
# plt.subplot(122)
# plt.hist(img.ravel(), 256, [0,256])
# plt.title('plt.hist 直方图')
# plt.xlabel('像素值')
# plt.ylabel('像素数量')

# plt.tight_layout()
# plt.show() #显示图像




