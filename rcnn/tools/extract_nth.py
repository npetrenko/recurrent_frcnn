import os
import numpy as np
from shutil import copyfile

sd = './sparsified/0'
md = './selected/'

annot = './fake_annots/0'

d = os.listdir(md)

form = d[0].split('.')[-1]

nums = [int(x.split('_')[-1].split('.')[0]) for x in d]

sort = np.argsort(nums)

new = []

for s in sort:
    new.append(d[s])

fake_annots = []
for i, s in enumerate(new):
    if i % 7 == 0:
        copyfile(os.path.join(md, s), os.path.join(sd,'Screen_{}.{}'.format(i//7, form)))
        fake_annots.append(str(i//7) +',0,1,0,1')

fake_annots = [x + '\n' for x in fake_annots]
with open(annot, 'w') as f:
    f.writelines(fake_annots)
