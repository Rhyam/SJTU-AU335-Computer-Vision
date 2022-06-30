import cv2
import numpy as np

def line_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray.jpg", gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #cv2.imwrite("edges.jpg", edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print(len(lines))
    cv2.imshow("Hough Line Detection", image)
    #cv2.imwrite(f"1_1_200_{len(lines)}.jpg", image)

if __name__ == '__main__':
    img = cv2.imread("test.jpg")
    line_detect(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
