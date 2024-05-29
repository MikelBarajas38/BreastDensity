import cv2
import numpy as np

thresh = 20 

def get_side(img):
    N = round(img.shape[1] / 2)
    img_left = img[:,0:N]
    img_right = img[:,N:]
    
    if img_left.mean() > img_right.mean():
        return 'LEFT'
    else:
        return 'RIGHT'   

def get_mask(img):
    norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(norm_img, thresh, norm_img.max(), cv2.THRESH_BINARY)

    #TODO: apply morphological operations
    
    return binary_mask

def get_clean_mask(img):
    
    h, w = img.shape

    # normalize image
    norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # apply otsu thresholding
    thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU )[1] 

    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw largest contour as white filled on black background as mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask

def clean_img(img, size=256):
    mask = get_clean_mask(img)

    img = cv2.bitwise_and(img, img, mask=mask)

    h, w = img.shape
    l = max((h, w))
    canvas = np.zeros((l, l), np.uint8)

    side = get_side(img)

    if side == 'RIGHT':
        canvas[0:h, l-w:l] = img
    else:
        canvas[0:h, 0:w] = img
    
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    if side == 'RIGHT':
        canvas = cv2.flip(canvas, 1)

    return canvas

def get_max_inscribed_circle(img):
    mask = get_mask(img)
    
    #make borders black
    mask[0] = 0
    mask[::,0] = 0
    mask[-1] = 0
    mask[::, -1] = 0

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    
    return center, round(radius)

def apply_clahe(img):

    clahe = cv2.createCLAHE(clipLimit = 40)  # crete clahe parameters

    img_umat = cv2.UMat(img)  # send img to gpu

    img_umat = clahe.apply(img_umat)

    # Normalize image [0, 255]
    img_umat = cv2.normalize(img_umat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_umat.get()  # recover img from gpu

def get_ROI(img, size=256):
    center, radius = get_max_inscribed_circle(img)
    side = radius * np.sqrt(2)

    x1 = int(center[0] - side / 2)
    x2 = int(center[0] + side / 2)
    y1 = int(center[1] - side / 2)
    y2 = int(center[1] + side / 2)

    ROI = img[y1:y2, x1:x2]
    ROI = apply_clahe(ROI)

    ROI = cv2.resize(ROI, (size, size), interpolation=cv2.INTER_AREA)

    return ROI