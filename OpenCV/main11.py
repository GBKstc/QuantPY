import Parking
import os
import cv2


current_path = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(current_path,"images", "parkingLot.webp")
image = cv2.imread(image_path)
parking = Parking.Parking(image)
print(parking.class_description)

# 图像预处理
white_img = parking.select_rgb_white_yellow(image)
gray_img = parking.convert_gray_scale(white_img)
canny_img = parking.detect_edges(gray_img,80,150)

# 选择感兴趣区域
region_img = parking.select_region(canny_img)

# 霍夫变换检测直线
lines = parking.hough_lines(region_img)

# 绘制检测到的直线
line_img = parking.draw_lines(image, lines)
parking.cv_show('line image', line_img)

# 识别停车位
final_img, rects = parking.identify_blocks(image, lines)
parking.cv_show('final image', final_img)



