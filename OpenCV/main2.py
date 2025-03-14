import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'images', 'cv6.png')

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



img = cv2.imread(img_path)
img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:,:]
#处理过后的图片
ret1, binaryInv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV) #二值化
# cv2.imshow('thresh1',thresh1)

# kernel = np.ones((3, 3), np.uint8)
# erode = cv2.erode(binaryInv,kernel,iterations = 1) #迭代次数表示做几次腐蚀操作
# dilate = cv2.dilate(binaryInv,kernel,iterations=1) #迭代次数表示做几次膨胀操作
# opening = cv2.morphologyEx(binaryInv, cv2.MORPH_OPEN, kernel) #先腐蚀后膨胀
# closing = cv2.morphologyEx(binaryInv, cv2.MORPH_CLOSE, kernel) #先膨胀后腐蚀
# gradient = cv2.morphologyEx(binaryInv, cv2.MORPH_GRADIENT, kernel) #膨胀与腐蚀的差别
# tophat  = cv2.morphologyEx(binaryInv, cv2.MORPH_TOPHAT, kernel) #原图与开运算的差
# blackhat = cv2.morphologyEx(binaryInv, cv2.MORPH_BLACKHAT, kernel) #闭运算与原图的差

# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3) #x方向梯度
# # sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
# # sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # 合并两个方向

# scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
# scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
# # scharrx = cv2.convertScaleAbs(scharrx)
# # scharry = cv2.convertScaleAbs(scharry)
# scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# laplacian = cv2.convertScaleAbs(laplacian)

# down = cv2.pyrDown(img) #下采样
# up = cv2.pyrUp(img) #上采样
# # l_1=img-down_up # 查看变换前后之间的差别

# Canny1=cv2.Canny(img_gray,80,150) #边缘检测
# Canny2=cv2.Canny(img_gray,100,300) #边缘检测

# #检测轮廓
# contours, _ = cv2.findContours(binaryInv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# cnt = max(contours, key=cv2.contourArea) #找到最大的轮廓
# contour_image = img.copy()  # 复制原图

# x,y,w,h = cv2.boundingRect(cnt)
# contour_image = cv2.rectangle(contour_image,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.drawContours(contour_image,[cnt],-1,(0,0,255),2)

# area = cv2.contourArea(cnt) #计算面积
# rect_area = w*h  #矩形面积
# extent = float(area)/rect_area #轮廓面积与边界矩形比
# 在计算面积前，先检查轮廓
# print("所有轮廓数量:", len(contours))
# print("当前选择的轮廓索引:", 2)
# print("当前轮廓点数:", len(cnt))
# print("轮廓实际面积:", area)
# print("外接矩形面积:", rect_area)
# print('轮廓面积与边界矩形比',extent)

#使用plt展示 2行3列
# plt.subplot(131),plt.imshow(img,'gray'),plt.title('img')
# plt.subplot(132),plt.imshow(Canny1,'gray'),plt.title('Canny1')
# plt.subplot(133),plt.imshow(contour_image,'gray'),plt.title('contour_image')
# save_img(contour_image,'contour_image.png')

# plt.subplot(344),plt.imshow(l_1,'gray'),plt.title('l_1')

# plt.subplot(345),plt.imshow(scharrx,'gray'),plt.title('scharrx')
# plt.subplot(346),plt.imshow(scharry,'gray'),plt.title('scharry')
# plt.subplot(347),plt.imshow(scharrxy,'gray'),plt.title('scharrxy')
# plt.subplot(348),plt.imshow(laplacian,'gray'),plt.title('laplacian')


#默认全屏展示
# plt.get_current_fig_manager().window.state('zoomed')
# plt.show()





