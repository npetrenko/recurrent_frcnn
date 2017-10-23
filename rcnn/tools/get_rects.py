import os
from multiprocessing import Pool
import cv2
import numpy as np

base_dir = '/u01/tmp/newCam/newCam/'
mask_source_dir = os.path.join(base_dir, 'Color/')
annotations_output_dir = os.path.join(base_dir, 'gt')
image_path_dir = os.path.join(base_dir, 'img1')
n_jobs = 40

try:
    os.mkdir(annotations_output_dir)
except:
    pass

prefix = 'Screen_'

def remove_prefix(line):
    if line.startswith(prefix):
        return line[len(prefix):]
    else:
        return line

for frame in os.listdir(image_path_dir):
    ix, form = frame.split('.')
    ix = remove_prefix(ix).zfill(6)
    os.rename(os.path.join(image_path_dir, frame), os.path.join(image_path_dir, '.'.join([ix, form])))

n_colors = 32

videos = os.listdir(mask_source_dir)

def get_rects(img_path):
    img = cv2.imread(img_path)

    try:
        Z = img.reshape((-1,3))
    except:
        print(img_path)
        raise

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

frames = list(map(lambda x: os.path.join(mask_source_dir, x), os.listdir(mask_source_dir)))
frames = [x for x in frames if not 'DS' in x]

frame_nums = [v.split('/')[-1].split('.')[0] for v in frames]

frame_bboxes = pool.map(get_rects, frames)

output_name = os.path.join(annotations_output_dir, 'gt.txt')

with open(output_name, 'w') as f:
    for bboxes, ix in zip(frame_bboxes, frame_nums):
        for bbox in bboxes:
            f.write(','.join(map(str, [remove_prefix(ix), 1] + bbox + [1])) + '\n') #extra info for MOT file format
