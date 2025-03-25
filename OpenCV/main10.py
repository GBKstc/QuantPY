import Stitcher
import imutils
import cv2
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True, help="第一张图片地址")
# ap.add_argument("-s", "--second", required=True, help="第二张图片地址")
# args = vars(ap.parse_args())
current_dir = os.path.dirname(os.path.abspath(__file__))
first_dir = os.path.join(current_dir, "images",'montage_L.png')
second_dir = os.path.join(current_dir, "images",'montage_R.png')
args = {"first": first_dir, "second": second_dir}

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

stitcher = Stitcher.Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()