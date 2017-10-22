import os
import numpy as np
from shutil import copyfile, rmtree
from functools import reduce

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
        

def get_absolute(root, leafs):
    return list(map(lambda x: os.path.join(root, x), leafs))

mot_dirs = ['/tmp/MOT16', '/tmp/2DMOT2015/', '/tmp/MOT17/']
#mot_dirs = ['/tmp/valid_dataset/']
info_file = 'seqinfo.ini'
target_framerate = 5
pad_zero_only=True

for mot_dir in mot_dirs:
    targets = get_absolute(mot_dir, ['train'])

    subtargets = {t: [x for x in os.listdir(t) if not '.' in x] for t in targets}

    targets = {}
    for k,v in subtargets.items():
        absp = get_absolute(k, v)
        data = list(map(lambda x: read_ini(os.path.join(x, info_file)), absp))
        targets.update({k:v for k,v in zip(absp, data)})

    def reindex(images):
        imgs = sorted(images)
        ix = map(lambda x: str(int(x.split('.')[0])//stride).zfill(6)+ext, imgs)
        return list(zip(imgs, ix))

    for target, data in targets.items():
        rate = data['frameRate']
        ext = data['imExt']

        root = '/'.join(target.split('/')[:-1])
        last = target.split('/')[-1]

        stride = rate// target_framerate

        if pad_zero_only:
            pads = [0]
        else:
            pads = range(stride)

        for pad in pads:
            new_data = os.path.join(root, last + '_' + str(pad) + 'pad')
            os.mkdir(new_data)
            [os.mkdir(os.path.join(new_data, d)) for d in ['img1', 'gt']]
            new_imgs = reindex([x for x in os.listdir(os.path.join(target, data['imDir'])) if int(x.split('.')[0]) % stride == pad])

            for old, new in new_imgs:
                os.rename(os.path.join(target, data['imDir'], old), os.path.join(new_data, data['imDir'], new))

            lines = []
            with open(os.path.join(target, 'gt/gt.txt'), 'r') as gt:
                with open(os.path.join(new_data, 'gt/gt.txt'), 'w') as gt_new:
                    for line in gt:
                        ix = line.split(',')[0]
                        rest = ','.join(line.split(',')[1:])
                        if int(ix) % stride == pad:
                            ix = int(ix) // stride
                            l = ','.join([str(ix), rest])
                            gt_new.writelines(l)
        rmtree(target)
                

