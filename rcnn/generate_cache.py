from .data_generators import pack, get_anchor
from multiprocessing import Pool
import os

def pack_frame(data):
    frame, class_count, C, img_length_calc_function = data

    x, y, d = get_anchor(frame, class_count, C, img_length_calc_function, 'tf', mode='test')
    pack(C, frame['filepath'], x, y, d)

def create_cache(videos, class_count, C, img_length_calc_function, n_jobs):
    # create directory structure
    tmp_dir = os.path.join(C.tmp_dir, 'rpn_tmp')
    nlev = 2
    prev_lev = [tmp_dir]

    for _ in range(nlev):
        clev = []
        for d in prev_lev:
            for dnum in range(100):
                path = os.path.join(d, str(dnum))
                os.mkdir(path)
                clev.append(path)
        prev_lev = clev

    # process frames

    frames = [(x, class_count, C, img_length_calc_function) for video in videos for x in video]
    pool = Pool(processes=n_jobs)
    pool.map(pack_frame, frames)
