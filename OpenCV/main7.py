import os
import cv2
from imutils import contours
import numpy as np

def cv_show(name,img):
   cv2.imshow(name, img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

#获取绝对地址
current_dir = os.path.abspath(os.path.dirname(__file__))
#模板图片地址
template_img_path = os.path.join(current_dir, 'images', 'template.png')
# 读取模板图片
template_img = cv2.imread(template_img_path)
# 转换为灰度图
template_img_ray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
# 转换为二值图
template_img_ray = cv2.threshold(template_img_ray, 10, 255, cv2.THRESH_BINARY_INV)[1]
# 获取轮廓 
template_img_ray_copy = template_img_ray.copy()
refCnts, hierarchy = cv2.findContours(template_img_ray_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#将其从左到右排列，从左到右，从上到下，一句坐标信息就能知道其标签信息
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
# 遍历每一个轮廓 创建一个字典，每一个数字对应每一个模板
digits = {}
for (i, c) in enumerate(refCnts):
   # 计算外接矩形并且resize成合适大小
   (x, y, w, h) = cv2.boundingRect(c)
   roi = template_img_ray[y:y + h, x:x + w]
   roi = cv2.resize(roi, (57, 88))
   # 每一个数字对应每一个模板
   digits[i] = roi
#模板处理结束

#处理银行卡图片
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)) #创建一个矩形核
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #创建一个方形核
card_img_path = os.path.join(current_dir, 'images', 'card.png')  # 待检测图片路径
card_img = cv2.imread(card_img_path)  # 读取待检测图片
card_img = cv2.resize(card_img, (300, int(card_img.shape[0] * 300 / card_img.shape[1])))   # 调整图片大小 300宽
card_img_ray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
tophat = cv2.morphologyEx(card_img_ray, cv2.MORPH_TOPHAT, rectKernel) #礼帽操作
# 计算x方向的梯度
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # x方向梯度
gradX = np.absolute(gradX)  # 取绝对值
(minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 取最小值和最大值
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化
gradX = gradX.astype("uint8")  # 转换为uint8类型
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel) #闭操作
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #二值化处理
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
thresh_cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #查找轮廓

locs = []
for (i,c) in enumerate(thresh_cnts):
    # 计算矩形
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h) # 计算宽高比
    if ar > 2.5 and ar < 4.0: # 选择合适的区域，根据实际任务来，这里只是示例
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            #符合的留下来
            locs.append((x,y,w,h))
            
# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
final_image = card_img.copy()
for(i,(gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []
    group = card_img_ray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5] #根据坐标 获取每一组数据
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #二值化处理
    dithCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #查找轮廓
    dithCnts = contours.sort_contours(dithCnts, method="left-to-right")[0] #将其从左到右排列
    for c in dithCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)          
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    print(groupOutput)   
    # 画出来
    # 在最终图像上绘制
    cv2.rectangle(final_image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(final_image, ''.join(groupOutput), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
     
cv_show('final_image',final_image)