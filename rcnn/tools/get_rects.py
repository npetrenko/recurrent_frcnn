import os
from multiprocessing import Pool
import cv2
import numpy as np

mask_source_dir = './crowdgen_data/Color/'
annotations_output_dir = './annotations'
n_jobs = 4

prefix = 'Screen_'

def remove_prefix(line):
    if line.startswith(prefix):
        return line[len(prefix):]

n_colors = 15

videos = os.listdir(mask_source_dir)
frames = {v: list(map(lambda x: os.path.join(mask_source_dir, v, x), os.listdir(os.path.join(mask_source_dir, v)))) for v in videos}

def get_rects(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    not_zero_index = np.nonzero(Z)
    not_zero = Z[not_zero_index[0]]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    K = n_colors
    
    ret,label,center=cv2.kmeans(not_zero,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]

    Z[not_zero_index[0]] = res
    res2 = Z.reshape((img.shape))
    
    rects = []
    for color in center:
        mask = cv2.inRange(res2, color, color)
        masked_data = np.uint8(cv2.bitwise_and(res2, res2, mask=mask))
        im_bw = cv2.cvtColor(masked_data, cv2.COLOR_RGB2GRAY)

        (thresh, im_bw) = cv2.threshold(im_bw, 10, 255, 0)

        im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            x,y,w,h = cv2.boundingRect(contour)
            if (w > 10 and h >10):
                rects.append([x,y,x+w,y+h])
    return rects

pool = Pool(n_jobs)

for video in videos:
    cframes = frames[video]

    frame_ims = list(map(cv2.imread, cframes))
    frame_nums = [v.split('/')[-1].split('.')[0] for v in cframes]

    frame_bboxes = pool.map(get_rects, frame_ims)

    output_name = os.path.join(annotations_output_dir, video)
    
    with open(output_name, 'w') as f:
        for bboxes, ix in zip(frame_bboxes, frame_nums):
            for bbox in bboxes:
                f.write(','.join(map(str, [remove_prefix(ix), 1] + bbox + [1])) + '\n') #extra info for MOT file format
