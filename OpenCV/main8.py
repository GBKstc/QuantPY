import os
import cv2

def cv_show(name,img):
   cv2.imshow(name,img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

#获取父目录绝对地址
current_dir = os.path.abspath(os.path.dirname(__file__))
#获取角识别图片地址
horn_img_path = os.path.join(current_dir,'images','horn_img.png')
# 读取图片
horn_img = cv2.imread(horn_img_path)
# 转换为灰度图
horn_img_ray = cv2.cvtColor(horn_img,cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(horn_img_ray,2,3,0.04)
horn_img_copy = horn_img.copy()
# 阈值
horn_img_copy[dst>0.01*dst.max()] = [0,0,255]
cv_show('dst',horn_img_copy)
