import cv2
import numpy as np
thresh = 20

def get_mask(img):
    norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(norm_img, thresh, norm_img.max(), cv2.THRESH_BINARY)

    #TODO: apply morphological operations
    
    return binary_mask

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