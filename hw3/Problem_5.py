import cv2
import numpy as np

img_mix = cv2.imread("origin.png", 0)
img_rgb = cv2.imread("origin.png")
img_mix = cv2.resize(img_mix, (1080, 720))
img_rgb = cv2.resize(img_rgb, (1080, 720))

# Guassion blur to eliminate the noise
img_blur=cv2.GaussianBlur(img_mix, (5, 5), 5)

# adaptive threshold to get binary image to emphasize characters
#img_thresh=cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
ret, img_thresh = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('img_thresh', img_thresh)

# dilation
kernel_1 = np.ones((3, 3), np.uint8)
dilation_1 = cv2.dilate(img_thresh, kernel_1, iterations=2) # dilate characters to be detected more easily
#cv2.imshow('dilation_1', dilation_1)

col = dilation_1.shape[1]
horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (col, 1))  # to get rectangle kernel
dilation_2 = cv2.dilate(dilation_1, horizontalKernel, iterations=1)     # dilate each line of characters to a whole rectangle
#cv2.imshow('dilation_2', dilation_2)

# erosion: to make rectangle thinner
kernel_2 = np.ones((4, 4),np.uint8)
erosion = cv2.erode(dilation_2, kernel_2, iterations=8)
#cv2.imshow('erosion', erosion)

# Hough line detection
lines = cv2.HoughLines(erosion, 0.9, np.pi/2, 300) # theta = pi/2 for horizontal lines only
black_img = np.zeros([720, 1080, 3], dtype=np.uint8)
for line in lines:
    rho, theta = line[0]
    y0 = int(rho)
    img_line = img_thresh[y0,:].tolist()
    x1 = 0
    x2 = 1080
    y1 = y2 = y0
    for i in range(1080):
        if img_line[i] == 0:
            continue
        else:
            x1 = i
            break
    for j in range(1080):
        if img_line[1079-j] == 0:
            continue
        else:
            x2 = 1080-j
            break
    cv2.line(black_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# merge several delete lines on the same line of words
verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
white_lines = cv2.dilate(black_img, verticalKernel, iterations=3)

# erode vertical delete lines
kernel_3 = np.ones((3, 1),np.uint8)
white_lines = cv2.erode(white_lines, kernel_3, iterations=5)

black_lines = ~ white_lines
#cv2.imshow("black_lines", black_lines)

result = cv2.bitwise_and(img_rgb, black_lines)
cv2.imshow("result", result)
cv2.imwrite('result.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()