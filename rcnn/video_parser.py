import cv2
import numpy as np
import os

def get_data(videos_path, annotations_path):
    found_bg = False
    all_videos = []

    classes_count = {}

    class_mapping = {}

    visualise = False
    annots = map(lambda x: os.path.join(annotations_path, x), os.listdir(annotations_path))
    
    print('Parsing annotation files')
    
    for input_path in annots:
        frame_path = os.path.join(videos_path, input_path.split('/')[-1].split('.')[0], '{}.jpg')
        frames = {}
        last_frame = -1
        with open(input_path,'r') as f:

            for line in f:
                line_split = line.strip().split(',')
                frameix,x1,y1,x2,y2 = map(int, line_split)
                class_name = 'bbox'


                last_frame = max(frameix, last_frame)

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                if not frameix in frames:
                    frames[frameix] = {}

                    #print(frame_path.format(frameix))
                    img = cv2.imread(frame_path.format(frameix))
                    (rows,cols) = img.shape[:2]
                    frames[frameix]['filename'] = frame_path.format(frameix)
                    frames[frameix]['width'] = cols
                    frames[frameix]['height'] = rows
                    frames[frameix]['bboxes'] = []
                    
                frames[frameix]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
        video = []
        for frameix in range(last_frame):
            video.append(frames[frameix])
        all_videos.append(video)


    return all_videos, classes_count, class_mapping


