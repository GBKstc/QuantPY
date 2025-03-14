import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def imShowFN(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

top_size, bottom_size, left_size, right_size = (50, 50, 50, 50) 

current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'images', 'cv1.jpeg')
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret1, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
# ret3, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
# ret4, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret5, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
blur = cv2.blur(img, (5,5))
box = cv2.boxFilter(img, -1, (5,5))
aussian = cv2.GaussianBlur(img, (5,5), 1)
medium = cv2.medianBlur(img, 5)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, blur, box, aussian, medium, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
