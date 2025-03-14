from imutils import contours
import numpy as np
import argparse
import cv2
import os
from matplotlib import pyplot as plt

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#    help="path to input image")
# ap.add_argument("-t", "--template", required=True,
#    help="path to template OCR-A image")
# args = vars(ap.parse_args())

FIRST_NUMBER = {
   "3": "American Express",
   "4": "Visa",
   "5": "MasterCard",
   "6": "Discover Card"
}

def cv_show(name,img):
   cv2.imshow(name, img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, 'images', 'template.png')  # 模板图片路径
template_img = cv2.imread(template_path)  # 读取模板图片
template_img_ray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
template_img_ray = cv2.threshold(template_img_ray, 10, 255, cv2.THRESH_BINARY_INV)[1]  # 二值化处理
#获取轮廓
refCnts, hierarchy = cv2.findContours(template_img_ray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#将其从左到右排列，从左到右，从上到下，一句坐标信息就能知道其标签信息
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}
# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
   # 计算外接矩形并且resize成合适大小
   (x, y, w, h) = cv2.boundingRect(c)
   roi = template_img_ray[y:y + h, x:x + w]
   roi = cv2.resize(roi, (57, 88))
   # 每一个数字对应每一个模板
   digits[i] = roi
###模板处理结束

#处理目标图片
img_path = os.path.join(current_dir, 'images', 'cv9.png')  # 待检测图片路径
image= cv2.imread(img_path)  # 读取待检测图片  
# 调整图片大小 300宽 
image = cv2.resize(image, (300, int(image.shape[0] * 300 / image.shape[1])))  
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
# 创建矩形核和方形核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # 矩形核
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 方形核

# 礼帽操作
tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rectKernel)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # x方向梯度
gradX = np.absolute(gradX)  # 取绝对值
(minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 取最小值和最大值
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化
gradX = gradX.astype("uint8")  # 转换为uint8类型
print(np.array(gradX).shape)  # (300, 300)  
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)  # 闭操作
thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化处理
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
threshCnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)  # 绘制轮廓

locs = []
# 遍历轮廓
for (i, c) in enumerate(cnts):
   # 计算矩形
   (x, y, w, h) = cv2.boundingRect(c)
   ar = w / float(h)  # 计算宽高比
   if ar > 2.5 and ar < 4.0:  # 选择合适的区域，根据实际任务来，这里只是示例
      if (w > 40 and w < 55) and (h > 10 and h < 20):
         #符合的留下来
         locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])  # 按照x坐标排序

# 遍历 locs 把矩形绘制在 cur_img 上
for  (i,(x, y, w, h)) in enumerate(locs):
   groupOutput = []
   # 根据坐标提取每一个组
   group = image_gray[y - 5:y + h + 5, x - 5:x + w + 5]
   #预处理
   group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  #  cv_show('group',group)
   # 计算每一组的轮廓
   digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
   # 计算每一组的每一个数值
   for c in digitCnts:
      # 找到当前数值的轮廓，resize成合适的的大小
      (x, y, w, h) = cv2.boundingRect(c)
      roi = group[y:y + h, x:x + w]
      roi = cv2.resize(roi, (57, 88))
      # cv_show('roi',roi)
      # 计算匹配得分
      scores = []
      # 在模板中计算每一个得分
      for (digit, digitROI) in digits.items():
         # 模板匹配
         result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
         (_, score, _, _) = cv2.minMaxLoc(result)
         scores.append(score)
      # 得到最合适的数字
      groupOutput.append(str(np.argmax(scores)))
      print(groupOutput)
     
   #画出来
   image_copy = image.copy()
   cv2.rectangle(image_copy,(x - 5, y - 5),(x + w + 5, y + h + 5),(0, 0, 255), 1)
   cv_show('image_copy',image_copy)


  

   

  

   











