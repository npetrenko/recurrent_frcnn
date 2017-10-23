import cv2
import numpy as np
import os

def read_ini(fpath):
    ret = {}
    with open(fpath, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=')
                v = v.rstrip()
                try:
                    v = int(v)
                except:
                    pass
                ret[k] = v
    return ret


def get_data(mot_pathes, part='train'):

    found_bg = False
    all_videos = []

    classes_count = {}

    class_mapping = {}

    visualise = False

    datasets = []
    for mot_path in mot_pathes:
        path = os.path.join(mot_path, part)
        datasets += [x for x in map(lambda x: os.path.join(path, x), os.listdir(path)) if not ('/.' in x)]
    
    print('Parsing annotation files')
    
    for dataset in datasets:
        try:
            form = read_ini(os.path.join(dataset, 'seqinfo.ini'))['imExt'][1:]
        except:
            form = 'jpg'

        try:
            sprob = read_ini(os.path.join(dataset, 'seqinfo.ini'))['sampleprob']
        except:
            sprob = 1

        coord_form = 'xywh'

        try:
            coord_form = read_ini(os.path.join(dataset, 'seqinfo.ini'))['coordform']
        except:
            pass

        frame_path = lambda x: os.path.join(dataset, 'img1', str(x).zfill(6) + '.' + form)
        #print(frame_path)
        frames = {}
        last_frame = -1
        first_frame = 1e8

        if part == 'train':
            bfile = 'gt/gt.txt'
        else:
            bfile = 'det/det.txt'
        with open(os.path.join(dataset, bfile),'r') as f:
            for line in f:
                line_split = line.strip().split(',')

                if part == 'train':
                    try:
                        cls = int(line_split[6])
                    except:
                        print(line)
                        print(dataset)
                        raise

                    if cls not in [1, 2, 7]:
                        continue

                try:
                    frameix,x1,y1,w,h = map(lambda x: int(float(x)), line_split[0:1] + line_split[2:6])
                except:
                    print(dataset, line)
                    raise

                if coord_form == 'xywh':
                    x2 = x1 + w
                    y2 = y1 + h
                else:
                    x2 = w
                    y2 = h

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

        break_flag = False
        for frameix in range(first_frame, last_frame+1):
            try:
                video.append(frames[frameix])
            except:
                print('Unable to fetch frames in {}, passing'.format(dataset))
                break_flag = True
                break

        if break_flag:
            continue
                
        all_videos.append({'video': video, 'sampleprob': sprob})


    return all_videos, classes_count, class_mapping


