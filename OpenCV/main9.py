import cv2
import numpy as np
import os
import time

def cv_show(img, name):
 cv2.imshow(name, img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

current_dir = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(current_dir,'images','lena_std.tif')
# 使用 cv2 读取图片
original_lena = cv2.imread(img_path)
# 使用 cv2 进行旋转
rows, cols = original_lena.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
lena_rot45 = cv2.warpAffine(original_lena, M, (cols, rows))

sift = cv2.SIFT_create()
# 获取各个图像的特征点及sift特征向量
# 返回值kp包含sift特征的方向、位置、大小等信息；des的shape为（sift_num， 128）， sift_num表示图像检测到的sift特征数量
(kp1, des1) = sift.detectAndCompute(original_lena, None)
(kp2, des2) = sift.detectAndCompute(lena_rot45, None)
print("=========================================")
print("=========================================")
print('lena 原图  特征点数目：', des1.shape[0])
print('lena 旋转图 特征点数目：', des2.shape[0])
print("=========================================")
print("=========================================")

for i in range(2):
  print("关键点", i)
  print("数据类型:", type(kp1[i]))
  print("关键点坐标:", kp1[i].pt)
  print("邻域直径:", kp1[i].size)
  print("方向:", kp1[i].angle)
  print("所在的图像金字塔的组:", kp1[i].octave)
print("=========================================")
print("=========================================")
"""
首先对原图和旋转图进行特征匹配，即图original_lena和图lena_rot45
"""
# 绘制特征点，并显示为红色圆圈
sift_original_lena = cv2.drawKeypoints(original_lena, kp1, original_lena, color=(255, 0, 255))
sift_lena_rot45 = cv2.drawKeypoints(lena_rot45, kp2, lena_rot45, color=(255, 0, 255))
sift_cat1 = np.hstack((sift_original_lena, sift_lena_rot45))
cv2.imwrite(os.path.join(current_dir,'images','sift_cat1.png'), sift_cat1)
# 特征点匹配
# K近邻算法求取在空间中距离最近的K个数据店，并将这些数据点归为一类
start = time.time()
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(des1, des2, k=2)
print('用于 原图和旋转图 图像匹配的所有特征点数目：', len(matches1))
# 调整ratio
# ratio = 0.4: 对于准确度要求高的匹配，一般选取0.4
# ratio = 0.6: 对于匹配点数目要求比较多的匹配，一般选取0.6
# ratio = 0.5: 一般情况下，一般选取0.5
ration1 = 0.5
good1 = []
for m, n in matches1:
 if m.distance < ration1 * n.distance:
  good1.append([m])
end = time.time()
print('匹配点匹配运行时间：%.4f秒' % (end - start))
#通过对good值进行索引，可以指定固定数目的特征点进行匹配，如good[:20]表示对前20个特征点进行匹配
match_result1 = cv2.drawMatchesKnn(original_lena, kp1, lena_rot45, kp2, good1[:20], None, flags=2)
cv2.imwrite(os.path.join(current_dir,'images','match_result1.png'), match_result1)
print('原图与旋转图 特征点匹配图像已保存')
print("=========================================")
print("=========================================")
print("原图与旋转图匹配对的数码：", len(good1))
for i in range(2):
 print("匹配对", i)
 print("数据类型：", type(good1[i][0]))
 print("描述符之间的距离", good1[i][0].distance)
 print("查询图像中描述符的索引值", good1[i][0].queryIdx)
 print("目标图像中描述符的索引值", good1[i][0].trainIdx)
print("=========================================")
print("=========================================")
#将待匹配图像通过旋转，变换等方式将其与目标图像对齐，这里使用单应性矩阵
# 单应性矩阵有八个参数，如果要解这八个参数的话，需要八个方程。
# 由于每一个对应的像素点可以产生2个方程（x一个，y一个），那么总共需只需要四个像素点就能解出这个单应性矩阵。
if(len(good1) > 4):
 ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good1]).reshape(-1, 1, 2)
 ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good1]).reshape(-1, 1, 2)
 ransacReprojThreshold = 4
 # RANSAC算法选择其中最优的四个点
 H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
 imgout = cv2.warpPerspective(original_lena, H, (lena_rot45.shape[1], lena_rot45.shape[0]),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
 cv2.imwrite(os.path.join(current_dir,'images','imgout.png'), imgout)
 print('单应性矩阵已保存')
 print("=========================================")

