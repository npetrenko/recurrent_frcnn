import os

tmp_dir = './tmp/rpn_tmp/'

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
