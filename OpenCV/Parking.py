import cv2
import numpy as np
import operator

class Parking:
  #class描述
  class_description = '停车场车位识别'
  def __init__(self,image):
        self.image = image

  def cv_show(self,img_name,img):
        cv2.imshow(img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

  def select_rgb_white_yellow(self,image):
        #过滤掉背景
        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])
        # lower_red 和高于upper_red的部分分别变成0，lower_red~upper_red之间的值变成255,相当于过滤背景
        while_mask = cv2.inRange(image, lower, upper)
        # self.cv_show('white mask', while_mask)
        masked = cv2.bitwise_and(image, image, mask=while_mask)
        # self.cv_show('After white mask', masked)
        return masked  

  def convert_gray_scale(self,image):
        # self.cv_show('convert_gray_scale image', image)
        #灰度化
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     
  #边缘检测
  def detect_edges(self,image, low_threshold, high_threshold):
        # self.cv_show('detect_edges image', image)
        return cv2.Canny(image, low_threshold, high_threshold)    

  def select_region(self,image):
        """
          手动选择区域
        """
        rows, cols = image.shape[:2]
        pt_1 = [cols*0.1, rows*0.90]
        pt_2 = [cols*0.1, rows*0.70]
        pt_3 = [cols*0.30, rows*0.55]
        pt_4 = [cols*0.60, rows*0.15]
        pt_5 = [cols*0.90, rows*0.15]
        pt_6 = [cols*0.90, rows*0.90]
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        point_img = image.copy()
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
        for point in vertices[0]:
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        return self.filter_region(image, vertices)    

  def filter_region(self,image, vertices):
        """
          过滤掉区域外的部分
        """
        mask = np.zeros_like(image)
        if len(mask.shape)==2:
            cv2.fillPoly(mask, vertices, 255)
    
        return cv2.bitwise_and(image, mask)      

  def hough_lines(self,image):
        """
          霍夫变换
        """
        return cv2.HoughLinesP(image,rho=0.1,theta=np.pi/10,threshold=15,minLineLength=9,maxLineGap=4)
  
  def draw_lines(self,image,lines,color=[255,0,0],thickness=2,make_copy=True):
        # 过滤霍夫变换检测到直线
        if make_copy:
            image = np.copy(image)
        cleaned = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(y2-y1) < 1 and abs(x2-x1)>=25 and abs(x2-x1)<=55:
                    cleaned.append((x1,y1,x2,y2))
                    cv2.line(image,(x1,y1),(x2,y2),color,thickness)

        print("No lines detected：",len(cleaned))
        return image            
  def identify_blocks(self,image,lines,make_copy=True):  
        """
          识别停车位
        """
        if make_copy:
            image = np.copy(image)
        #Step 1: 过滤部分直线
        cleaned = []
        for line in lines:
          for x1,y1,x2,y2 in line:
            if abs(y2-y1) < 1 and abs(x2-x1)>=25 and abs(x2-x1)<=55:
              cleaned.append((x1,y1,x2,y2))    

        #Step 2: 对直线按照x1进行排序
        list1 = sorted(cleaned, key=operator.itemgetter(0,1))

        #Step 3: 找到多个列,相当于每列是一排车
        clusters = {}
        dIndex = 0
        clus_dist = 10

        for i in range(len(list1)-1):
          distance = abs(list1[i+1][0] - list1[i][0])
          if distance <= clus_dist:
            if dIndex not in clusters.keys():
              clusters[dIndex] = []
            clusters[dIndex].append(list1[i])
            clusters[dIndex].append(list1[i+1])  
          else:
            dIndex += 1

        #Step 4:得到坐标
        rects = {}
        i = 0
        for key in clusters:
          all_list = clusters[key]
          cleaned = list(set(all_list))
          if len(cleaned) > 5:
            cleaned = sorted(cleaned,key=lambda tup:tup[1])
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            avg_x1 = 0
            avg_x2 = 0
            for tup in cleaned:
              avg_x1 += tup[0]
              avg_x2 += tup[2]
            avg_x1 = avg_x1/len(cleaned)
            avg_x2 = avg_x2/len(cleaned)
            rects[i] = (avg_x1,avg_y1,avg_x2,avg_y2)
            i += 1  
        
        print("Num Parking Lanes：",len(rects))

        #Step 5: 把列矩形画出来
        buff = 7
        for key in rects:
          tup_topLeft = (int(rects[key][0] - buff),int(rects[key][1]))
          tup_botRight = (int(rects[key][2] + buff),int(rects[key][3]))
          cv2.rectangle(image,tup_topLeft,tup_botRight,(0,255,0),3)

        return image,rects