import cv2
import numpy as np
import os

def get_data(mot_path, part='train', form='png'):
    path = os.path.join(mot_path, part)

    found_bg = False
    all_videos = []

    classes_count = {}

    class_mapping = {}

    visualise = False
    datasets = [x for x in map(lambda x: os.path.join(path, x), os.listdir(path)) if not ('/.' in x)]
    
    print('Parsing annotation files')
    
    for dataset in datasets:
        frame_path = lambda x: os.path.join(dataset, 'img1', str(x).zfill(6) + '.' + form)
        #print(frame_path)
        frames = {}
        last_frame = -1
        first_frame = 1e8
        with open(os.path.join(dataset, 'gt/gt.txt'),'r') as f:
            for line in f:
                line_split = line.strip().split(',')
                frameix,x1,y1,x2,y2 = map(int, line_split[0:1] + line_split[2:6])

                class_name = 'bbox'

                last_frame = max(frameix, last_frame)
                first_frame = min(first_frame, frameix)

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                if not frameix in frames:
                    frames[frameix] = {}

                    #print(frame_path.format(frameix))
                    img = cv2.imread(frame_path(frameix))
                    try:
                        (rows,cols) = img.shape[:2]
                    except:
                        print(frame_path(frameix), frameix)
                    frames[frameix]['filepath'] = frame_path(frameix)
                    frames[frameix]['width'] = cols
                    frames[frameix]['height'] = rows
                    frames[frameix]['bboxes'] = []
                    
                frames[frameix]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
        video = []
        for frameix in range(first_frame, last_frame+1):
            video.append(frames[frameix])
        all_videos.append(video)


    return all_videos, classes_count, class_mapping


