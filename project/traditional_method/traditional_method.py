import cv2
import numpy as np
import glob

def license_location(img):
    # hsv and Gaussian blur
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_blur = cv2.GaussianBlur(img_hsv, [5, 5], 0)
    
    # hvs color segmentation to determine car licence plate area according to its blue color
    img_mask = cv2.inRange(img_hsv_blur, np.array([100, 115, 115]), np.array([124, 255, 255]))
    #cv2.imshow('img_mask', img_mask)

    # morphology open to eliminate noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 2)

    # morphology close to get whole licence area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 2)
    #cv2.imshow('img_lcs', img_lcs)
    
    # cut car license plate out
    lcs_area = np.where(img_lcs==255)
    x1, y1, x2, y2 = min(lcs_area[1]), min(lcs_area[0]), max(lcs_area[1]), max(lcs_area[0])
    lcs = img[y1 + 50:y2 - 50, x1 + 20:x2 - 20]
    cv2.imshow('lcs', lcs)

    return lcs

def license_locate_perspectiveTransform(img):
    # hsv and Gaussian blur
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_blur = cv2.GaussianBlur(img_hsv, [5, 5], 0)

    # hvs color segmentation to determine car licence plate area according to its blue color
    img_mask = cv2.inRange(img_hsv_blur, np.array([100, 115, 115]), np.array([124, 255, 255]))
    #cv2.imshow('img_mask_blue', img_mask)

    # morphology open to eliminate noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 1)

    # morphology close to get whole licence area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 2)
    #cv2.imshow('img_lcs', img_lcs)

    # find contours of areas which possibly are license plate
    contours, hierarchy = cv2.findContours(img_lcs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # if the license plate is of green grounding, repeat the above process
    if len(contours) == 0:
        img_mask = cv2.inRange(img_hsv_blur, np.array([35, 10, 160]), np.array([70, 100, 200]))
        #cv2.imshow('img_mask_green', img_mask)

        # morphology open to eliminate noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 1)

        # morphology close to get whole licence area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 1)
        #cv2.imshow('img_lcs_green', img_lcs)

        # find contours of areas which possibly are license plate 
        contours, hierarchy = cv2.findContours(img_lcs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cut the oblique licence plate out and get the oblique angle
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w > h * 1.2 and w < h * 2.2 and w > 410:
            lcs_oblique = img[y:y + h - 25, x:x + w - 5]
            lcs_oblique_bin = img_lcs[y:y + h - 25, x:x + w - 5]
            cv2.imshow('lcs_oblique', lcs_oblique)
            angle = cv2.minAreaRect(i)[2]
            break
    
    # compute the vertex of the license plate
    lcs_area = np.where(lcs_oblique_bin==255)
    x1, y1, x2, y2 = min(lcs_area[1]), min(lcs_area[0]), max(lcs_area[1]), max(lcs_area[0])
    dx = x2 - x1
    dy = lcs_area[0].shape[0] // dx
    
    if angle < 45:
        src_vertex = np.array([[x1, y1], [x2, y2 - dy], [x1, y1 + dy], [x2, y2]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1 + int(1.5 * dx), y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)
    elif angle < 70:
        src_vertex = np.array([[x1, y2 - dy + 25], [x1, y2], [x2, y1 + 20], [x2 - 5, y1 + dy - 30]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)
    else:
        src_vertex = np.array([[x1 + 20, y2 - dy + 27], [x1, y2], [x2, y1 + 20], [x2 - 25, y1 + dy - 20]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)

    # perspective transform to get the orthogonal license plate
    M = cv2.getPerspectiveTransform(src_vertex, dst_vertex)
    lcs_orth = cv2.warpPerspective(lcs_oblique, M, (int(1.5 * dx), dy))
    cv2.imshow('lcs', lcs_orth)
    
    return lcs_orth

def grounding_identify(img):
    # identify the grounding color of license plate is blue or green
    h, w = img.shape[0], img.shape[1]
    B, G = 0, 0
    for i in range(h):
        for j in range(w):
            B += img[i, j, 0]
            G += img[i, j, 1]
    
    if B > G:
        return True
    else:
        return False

def preprocess(img):
    # resize
    img_bgr = cv2.resize(img, (int(200*img.shape[1]/img.shape[0]), 200))
    
    # identify the grounding color is blue or green
    isBlue = grounding_identify(img_bgr)

    # gray
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to eliminate noise
    img_blur = cv2.GaussianBlur(img_gray, [5, 5], 5)

    # threshold to get binary image to emphasize characters
    if isBlue == False:
        ret, img_thresh = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
    #cv2.imshow('img_thresh', img_thresh)

    # morphology open
    if isBlue == False:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    #cv2.imshow('img_open', img_open)

    # dilate every character for easier split 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 70))
    img_dilated = cv2.dilate(img_open, kernel, iterations=1)
    #cv2.imshow('img_dilated', img_dilated)

    return img_open, img_dilated

def split_character(lcs_char, lcs_char_shape):
    # find contours of every characters
    contours, hierarchy = cv2.findContours(lcs_char_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for i in contours:
        # use rectangle to bond the contour of character
        rect = cv2.boundingRect(i) # rect = [x, y, w, h]
        chars.append(rect)

    # sort characters according to the left-to-right order
    chars = sorted(chars, key=lambda x:x[0], reverse=False)
    char_imgs = []
    for char in chars:
        # judge the characters by the length-width ratio
        if char[3] > char[2] * 1.5 and char[3] < char[2] * 2.2:
            splited_char = lcs_char[char[1]:char[1] + char[3], char[0] + 8:char[0] + char[2] - 8]
            char_imgs.append(splited_char)
    
    for i, char_img in enumerate(char_imgs):
        cv2.imshow('char {}'.format(i), char_img)

    return char_imgs

def template_matching(char_imgs):

    path = 'template_data/'

    nums = ['{}'.format(i) for i in range(10)]
    caps = [chr(i) for i in range(65, 91)]
    caps = caps[0:8] + caps[9:14] + caps[15:25] # 'I' and 'O' are not used in car licence
    prvs = ['藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁',\
            '蒙','闽','宁','青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']
    srns = nums + caps

    prv_temps = [glob.glob(path + prv + '/*') for prv in prvs]
    cap_temps = [glob.glob(path + cap + '/*') for cap in caps]
    srn_temps = [glob.glob(path + srn + '/*') for srn in srns]

    result = ''
    for i, char_img in enumerate(char_imgs):
        if i == 0:
            chars = prvs
            temps = prv_temps
        elif i == 1:
            chars = caps
            temps = cap_temps
        else:
            chars = srns
            temps = srn_temps
        
        scores = []
        for temp in temps:
            score = []
            for sample in temp:
                template = cv2.imdecode(np.fromfile(sample, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                template = cv2.resize(template, (char_img.shape[1], char_img.shape[0]))
                score.append(cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF)[0][0])
            scores.append(max(score))

        index = np.argmax(np.array(scores))
        result = result + chars[index]
        if i == 1:
            result = result + '·'

    return result

if __name__ == '__main__':
    img_name = input('Please input the image name: ')
    if img_name[0] == '1':
        img_bgr = cv2.imread('images/easy/' + img_name)
        char, char_shape = preprocess(img_bgr)
    elif img_name[0] == '2':
        img_bgr = cv2.imread('images/medium/' + img_name)
        lcs = license_location(img_bgr)
        char, char_shape = preprocess(lcs)
    elif img_name[0] == '3':
        img_bgr = cv2.imread('images/difficult/' + img_name)
        lcs = license_locate_perspectiveTransform(img_bgr)
        char, char_shape = preprocess(lcs)
    char_imgs = split_character(char, char_shape)
    print(template_matching(char_imgs))
    cv2.waitKey(0)
    cv2.destroyAllWindows()