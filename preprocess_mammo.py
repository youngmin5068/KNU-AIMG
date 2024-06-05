import numpy as np
import cv2


def globalBinarise(img, thresh=0.1, maxval=1.0):
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    return binarised_img

def sortContoursByArea(contours, reverse=True):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    return sorted_contours, bounding_boxes

def xLargestBlobs(mask, top_x=None, reverse=True):

    X_largest_blobs = np.zeros(mask.shape, np.uint8)
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    n_contours = len(contours)
    if n_contours > 0:
        if n_contours < top_x or top_x is None:
            top_x = n_contours

        sorted_contours, bounding_boxes = sortContoursByArea(contours=contours, reverse=reverse)

        X_largest_contours = sorted_contours[0:top_x]

        to_draw_on = np.zeros(mask.shape, np.uint8)
        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,
            contours=X_largest_contours,
            contourIdx=-1,
            color=1,
            thickness=-1,
        )
    return n_contours, X_largest_blobs
    
def applyMask(img, mask):
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img

def crop_mammogram(image, num_iter=5, buffer_size=10):
    mask = image > 0
    eroded_mask = cv2.erode(np.float32(mask), None, iterations=num_iter)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(eroded_mask), 4, cv2.CV_32S)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background label 0
    largest_mask = (labels == largest_component)
    dilated_mask = cv2.dilate(np.float32(largest_mask), None, iterations=num_iter)
    
    x, y, w, h, _ = stats[largest_component]
    x = max(0, x - buffer_size)
    y = max(0, y - buffer_size)
    w = min(image.shape[1] - x, w + 2 * buffer_size)
    h = min(image.shape[0] - y, h + 2 * buffer_size)
    
    cropped_image = image[y:y+h, x:x+w]

    original_height, original_width = image.shape[:2]
    padding_top = y
    padding_bottom = original_height - (y + h)
    padding_left = x
    padding_right = original_width - (x + w)

    return cropped_image, padding_top, padding_bottom, padding_left, padding_right