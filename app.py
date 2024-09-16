import cv2
from PIL import ImageGrab
import numpy as np
import time
import keyboard
import pyautogui
import math

#! Game link: https://www.safekidgames.com/zig-zag/ and minimize window then make the game fullscreen
#! put the window at the middle of the screen then make the game dark mode for better performance.

def nothing(x):
    pass

# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("L - H", "Trackbars", 153, 179, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 96, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 175, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 156, 179, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("Thresh1", "Trackbars", 190, 1000, nothing)
# cv2.createTrackbar("Thresh2", "Trackbars", 135, 1000, nothing)
# cv2.createTrackbar("hThresh", "Trackbars", 19, 400, nothing)
# cv2.createTrackbar("hMinLine", "Trackbars", 20, 100, nothing)
# cv2.createTrackbar("hMaxGap", "Trackbars", 1, 100, nothing)
kernel = np.ones((3,3), np.uint8) 
ballX = 0
ballY = 0
r = 0
direction = 0
prevBallX = 0
we = 0
a = 0
dividor = 2
num = 13
threshHold = 0.55
prevRects = []
# diamond = cv2.imread("./diamond-3.PNG", cv2.IMREAD_GRAYSCALE)
diamonds = [cv2.imread("./diamond-2.PNG", cv2.IMREAD_GRAYSCALE)]

while True:
    if keyboard.is_pressed("u"):
        break
    elif keyboard.is_pressed("q"):
        direction = 0
    elif keyboard.is_pressed("w"):
        direction = 1
    # l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    # l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    # l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    # u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    # u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    # u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    # Thresh1 = cv2.getTrackbarPos("Thresh1", "Trackbars")
    # Thresh2 = cv2.getTrackbarPos("Thresh2", "Trackbars")
    # hThresh = cv2.getTrackbarPos("hThresh", "Trackbars")
    # hMinLine = cv2.getTrackbarPos("hMinLine", "Trackbars")
    # hMaxGap = cv2.getTrackbarPos("hMaxGap", "Trackbars")

    # lower_blue = np.array([l_h, l_s, l_v])
    # upper_blue = np.array([u_h, u_s, u_v])

    # lower_blue = np.array([85, 11, 11])
    # upper_blue = np.array([98, 255, 255])
    Thresh1 = 90
    Thresh2 = 108
    hThresh = 0
    we = 0
    img = np.array(ImageGrab.grab(bbox=(811, 173, 1135, 500)))
    # btn = cv2.imread("./btn.PNG", cv2.IMREAD_GRAYSCALE)
    # img = mss.mss().grab({
    #     'left': 600,
    #     'top': 360,
    #     'width': 330,
    #     'height': 80
    # })
    # img = np.array(img)
    # img = cv2.imread("./test3.PNG", cv2.IMREAD_COLOR)
    finalImg = np.zeros_like(img)
    finalImg = cv2.cvtColor(finalImg, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    lines = cv2.Canny(img, threshold1=Thresh1, threshold2=Thresh2)
    img_dilation = cv2.dilate(lines, kernel, iterations=1) 
    HoughLines = cv2.HoughLinesP(lines, 1, np.pi/180, threshold = hThresh, minLineLength = 0, maxLineGap = 0)
    if HoughLines is not None:
        for line in HoughLines:
            coords = line[0]
            cv2.line(finalImg, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 1)
    # lines[mask>0]=0
    for diamond in diamonds:
        found = cv2.matchTemplate(gray, diamond, cv2.TM_CCOEFF_NORMED)
        locs = np.where(found >= threshHold)
        locs = list(zip(*locs[::-1]))
        rects = []
        for loc in locs:
            rect = [int(loc[0]), int(loc[1]), diamond.shape[1], diamond.shape[0]]
            # if rect[1] + (rect[3] / 2) >= ballY + 10:
            #     continue
            rects.append(rect)
        rects, weights = cv2.groupRectangles(rects, 1, 1)
        for (x, y, w, h) in rects:
            top_left  = (x - num, y - num)
            bottom_right = (x + w + num, y + h + num)
            center = [math.floor((top_left[0] + bottom_right[0]) / 2), math.floor((top_left[1] + bottom_right[1]) / 2)]
            # w += num
            # h += num
            # points = np.array([top_left, (top_left[0] + w, top_left[1]), bottom_right, (bottom_right[0] - w, bottom_right[1])])
            # points = np.array([(center[0], center[1] - math.floor(h / 2)), (center[0] + math.floor(w / 2), center[1]), (center[0], center[1] + math.floor(h / 2)), (center[0] - math.floor(w / 2), center[1])])
            # cv2.rectangle(finalImg, (math.floor(top_left[0]),math.floor(top_left[1])), (math.floor(bottom_right[0]), math.floor(bottom_right[1])), (0, 0, 0), -1)
            cv2.circle(finalImg, center, num, (0, 0, 0), -1)
        # for (x, y, w, h) in prevRects:
        #     top_left  = (x - num, y - num)
        #     bottom_right = (x + w + num, y + h + num)
        #     center = [math.floor((top_left[0] + bottom_right[0]) / 2), math.floor((top_left[1] + bottom_right[1]) / 2)]
        #     # w += num
        #     # h += num
        #     # points = np.array([top_left, (top_left[0] + w, top_left[1]), bottom_right, (bottom_right[0] - w, bottom_right[1])])
        #     # points = np.array([(center[0], center[1] - math.floor(h / 2)), (center[0] + math.floor(w / 2), center[1]), (center[0], center[1] + math.floor(h / 2)), (center[0] - math.floor(w / 2), center[1])])
        #     # cv2.rectangle(finalImg, (math.floor(top_left[0]),math.floor(top_left[1])), (math.floor(bottom_right[0]), math.floor(bottom_right[1])), (0, 0, 0), -1)
        #     cv2.circle(finalImg, center, num, (0, 0, 0), -1)
        # prevRects = rects
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=17, minRadius=8, maxRadius=14)
    if circles is not None:
        circles = np.uint16(circles)
        for pt in circles[0, :]:
            x, y, r = pt[0], pt[1], pt[2]
            ballX, ballY, r = pt[0], pt[1], pt[2]
            cv2.circle(finalImg, (x,y), r, (255, 0, 0), 1)
            cv2.circle(lines, (x,y), r, (255, 0, 0), 1)
    if prevBallX > ballX:
        direction = 1
    elif prevBallX < ballX:
        direction = 0
    if a < 51:
        a += 1
    prevBallX = ballX
    for i in range(8, 14):
        try:
            if direction == 0 and finalImg[ballY - math.floor(i / dividor)][ballX + r + i] == 255 or direction == 0 and finalImg[ballY - math.floor(i / .9)][ballX + r + i] == 255:
                we = (ballX + i + r, ballY - math.floor(i / dividor))
                direction = 1
                pyautogui.click()
            elif direction == 1 and finalImg[ballY - math.floor(i / dividor)][ballX - i - r] == 255 or direction == 1 and finalImg[ballY - math.floor(i / .9)][ballX - i - r] == 255:
                we = (ballX - i - r, ballY - math.floor(i / dividor))
                direction = 0
                pyautogui.click()
        except:
            continue
    if we != 0:
        cv2.circle(finalImg, we, 10, (255, 255, 255), -1)
    cv2.imshow("image", finalImg)
    # cv2.imshow(i, lines)
    cv2.waitKey(1)
