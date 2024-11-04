import os
import openpyxl
import shutil
import openslide
from pathlib import Path


workbook = openpyxl.load_workbook('NDPI_labels.xlsx')
sheet = workbook.active
rows = sheet.iter_rows()

ori_names = []

for i, row in enumerate(rows):
    id = str(row[2].value)
    ori_names.append(id)

'''
os.makedirs('WSI/features3/gigapath_features', exist_ok=True)
os.makedirs('WSI/features3/uni_features', exist_ok=True)

for name in names:
    src = 'WSI/features/gigapath_features/' + name + '.pkl'
    dst = 'WSI/features3/gigapath_features/' + name + '.pkl'
    shutil.move(src, dst)

path = Path('WSI/')
ndpi = []
for p in path.rglob('*.ndpi'):
    ndpi.append(p.as_posix().split('/')[-1].split('.')[0])
'''

workbook = openpyxl.load_workbook('NDPI_labels.xlsx')
sheet = workbook.active
rows = sheet.iter_rows()

names = []

for i, row in enumerate(rows):
    if i > 0:
        id = str(row[2].value)
        names.append(id)

cnt = 0

for name in names:
    if name not in set(ori_names):
        print(name)