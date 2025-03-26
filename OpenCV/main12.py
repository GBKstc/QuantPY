import glob
import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np


def show(img):
    img[0] = cv2.resize(img[0], (900, 600))
    img[1] = cv2.resize(img[1], (900, 600))
    pic = np.hstack((img[0], img[1]))
    cv2.imshow('pic', pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_rgb_white_yellow(img):
    # 过滤掉背景
    lower = np.float32([0.45, 0.45, 0.45])
    upper = np.float32([1.0, 1.0, 1.0])
    # 低于lower和高于upper都变成0，其余变成255
    white_mask = img.copy()
    masked = img.copy()
    for i in range(len(img)):
        white_mask[i] = cv2.inRange(img[i], lower, upper)
        masked[i] = cv2.bitwise_and(img[i], img[i], mask=white_mask[i])
    # show(white_mask)
    # show(masked)
    return masked


def convert_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def int8(img):
    img = img * 255
    return img.astype(np.uint8)


def detect_edges(img, low_th=80, high_th=200):
    img = int8(img)
    return cv2.Canny(img, low_th, high_th)


def select_region(img):
    rows, cols = img.shape[:2]
    pt_1 = [cols * 0.05, rows * 0.90]
    pt_2 = [cols * 0.05, rows * 0.70]
    pt_3 = [cols * 0.30, rows * 0.52]
    pt_4 = [cols * 0.60, rows * 0.15]
    pt_5 = [cols * 0.90, rows * 0.15]
    pt_6 = [cols * 0.90, rows * 0.90]

    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]],dtype=np.int32)
    point_img = img.copy()
    point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
    for point in vertices[0]:
        cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
    return point_img


def filter_region(img):
    rows, cols = img.shape[:2]
    pt_1 = [cols * 0.05, rows * 0.90]
    pt_2 = [cols * 0.05, rows * 0.70]
    pt_3 = [cols * 0.30, rows * 0.52]
    pt_4 = [cols * 0.60, rows * 0.15]
    pt_5 = [cols * 0.90, rows * 0.15]
    pt_6 = [cols * 0.90, rows * 0.90]
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
    mask = np.zeros_like(img)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)  # 填充
    return cv2.bitwise_and(img, mask)


def hough_lines(img):
    # 输入图像是边缘检测后的图像
    return cv2.HoughLinesP(img, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    if make_copy:
        img = np.copy(img)
    # 添加空值检查
    if lines is None:
        return img
        
    cleaned = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) <=1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                cleaned.append((x1, y1, x2, y2))
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def identify_blocks(img, lines, make_copy=True):
    if make_copy:
        img = np.copy(img)
    # Step1:过滤直线
    cleaned =[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) <=1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55 :
                cleaned.append((x1, y1, x2, y2))
    # Step2:过滤后的直线按x1进行排序
    import operator
    list1 = sorted(cleaned, key=operator.itemgetter(0, 1))
    # Step3:找到多个列
    clusters = {}
    dIndex = 0
    clus_dist = 15
    for i in range(len(list1) - 1):
        distance = abs(list1[i+1][0] - list1[i][0])
        if distance <= clus_dist:
            if not dIndex in clusters.keys(): clusters[dIndex] = []
            clusters[dIndex].append(list1[i])
            clusters[dIndex].append(list1[i + 1])  # 会重复加入字典
        else:
            dIndex += 1
    # Step4:得到坐标
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        if len(cleaned) > 5:
            cleaned = sorted(cleaned, key=lambda tup: tup[1])  # 按y值进行排序
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            avg_x1 = 0
            avg_x2 = 0
            for tup in cleaned:
                avg_x1 += tup[0]
                avg_x2 += tup[2]
            avg_x1 = avg_x1 / len(cleaned)
            avg_x2 = avg_x2 / len(cleaned)
            rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
            i += 1
    # Step5:画出簇
    buff = 7
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(img, tup_topLeft, tup_botRight, (0, 255, 0), 3)
    return img, rects


def optimization_rect_coords(rects):
    for key in rects:
        min_y = 5000
        max_y = 0
        for rect in key:
            if min_y > key[rect][1]:
                min_y = key[rect][1]
            if max_y < key[rect][3]:
                max_y = key[rect][3]
        for rect in key:
            key[rect] = (key[rect][0], key[rect][1], key[rect][2], max_y - 12)
            if rect >= 7:
                key[rect] = (key[rect][0], min_y, key[rect][2], key[rect][3])
    return rects


def drew(img, optimization_rects):
    buff = 7
    img = np.copy(img)
    for key in optimization_rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(img, tup_topLeft, tup_botRight, (0, 255, 0), 2)
    return img


def draw_parking(image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
    if make_copy:
        new_image = np.copy(image)
    gap = 16.0
    spot_dict = {}  # 车位对应字典
    tot_spots = 0
    # 微调
    adj_y1 = {0: -12, 1: -5, 2: 0, 3: 0, 4: 10, 5: 12, 6: -15, 7: -32, 8: -32, 9: -25, 10: 50, 11: 0}
    adj_y2 = {0: 0, 1: -5, 2: -3, 3: 0, 4: -1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 5, 11: 53}
    adj_x1 = {0: -8, 1: -10, 2: -10, 3: -10, 4: -15, 5: -15, 6: -15, 7: -15, 8: -15, 9: -10, 10: -10, 11: -2}
    adj_x2 = {0: 0, 1: 18, 2: 18, 3: 16, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 15, 10: 12, 11: 0}
    for key in rects:
        tup = rects[key]
        x1 = int(tup[0] + adj_x1[key])
        x2 = int(tup[2] + adj_x2[key])
        y1 = int(tup[1] + adj_y1[key])
        y2 = int(tup[3] + adj_y2[key])
        cv2.rectangle(new_image, (x1, y1), (x2, y2), color, 2)
        num_splits = int(abs(y2 - y1)//gap)  # 分割
        for i in range(0, num_splits + 1):
            y = int(y1 + i * gap)
            cv2.line(new_image, (x1, y), (x2, y), (0, 0, 255), thickness)
        if key > 0 and key < len(rects) - 1:
            # 竖直线
            x = int((x1 + x2) / 2)
            cv2.line(new_image, (x, y1), (x, y2), (0, 0, 255), thickness)
            tot_spots += (num_splits + 1) * 2
        else:
            tot_spots += num_splits + 1
        # 字典对应
        if key == 0 and key == (len(rects) - 1):
            for i in range(0, num_splits + 1):
                cur_len = len(spot_dict)
                y = int(y1 + i * gap)
                spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
        else:
            for i in range(0, num_splits + 1):
                cur_len = len(spot_dict)
                y = int(y1 + i * gap)
                x = int((x1 + x2) / 2)
                spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                spot_dict[(x, y, x2, y + gap)] = cur_len + 2

    return new_image, spot_dict


import os
def data(img, spot_dict):
    # img = np.copy(img)
    for spot in spot_dict.keys():
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        # 裁剪
        spot_img = img[y1:y2, x1:x2] * 255
        spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
        spot_id = spot_dict[spot]
        filename = 'spot' + str(spot_id) + '.png'
        cv2.imwrite(os.path.join('./data', filename), spot_img)

current_path = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(current_path,"images", "parkingLot.webp")

test_images = [plt.imread(image_path)]
class_dictionary = {}
class_dictionary[0] = 'empty'
class_dictionary[1] = 'occupied'
# show(test_images)
# 过滤背景
images = select_rgb_white_yellow(test_images)
# show(images)
# 转灰度图
gray_images = list(map(convert_gray_scale, images))
# show(gray_images)
# 边缘检测
edge_images = list(map(detect_edges, gray_images))
# show(edge_images)
# 定位区域
roi_images = list(map(select_region, edge_images))
# show(roi_images)
# 剔除多余区域
region_images = list(map(filter_region, edge_images))
# show(region_images)
# 检测直线
list_lines = list(map(hough_lines, region_images))
# print(len(list_lines[0]))
# 画线
line_images = []
for img, lines in zip(test_images, list_lines):
    line_images.append(draw_lines(img, lines))
# show(line_images)
# 定位列
rect_images = []
rect_coords = []
for img, lines in zip(test_images, list_lines):
    new_image, rects = identify_blocks(img, lines)
    rect_images.append(new_image)
    rect_coords.append(rects)
# show(rect_images)
# 优化定位框
optimization_rects = optimization_rect_coords(rect_coords)
optimization_imgs = list(map(drew, test_images, optimization_rects))
# show(optimization_imgs)
# 切分停车位
delineated = []
spot_pos = []
for image, rects in zip(test_images, optimization_rects):
    new_image, spot_dict = draw_parking(image, rects)
    data(image, spot_dict)  # 数据集构建
    delineated.append(new_image)
    spot_pos.append(spot_dict)
# show(delineated)
# 保存
import pickle
with open('spot_dict.pickle', 'wb') as handle:
    pickle.dump(spot_pos[0], handle, protocol=pickle.HIGHEST_PROTOCOL)  # spot_pos[0]定位更精确
