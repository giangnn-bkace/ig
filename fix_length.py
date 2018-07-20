# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:13:52 2018

@author: NGN
"""

import csv
import os

length = 20

file_path = os.path.join('data', 'clipped_data', 'flow_origin.csv')
dest_path = os.path.join('data', 'clipped_data', 'flow.csv')
rgb_file = []

with open(file_path, 'r') as fin:
    reader = csv.reader(fin)
    for row in reader:
        if int(row[2]) >= length:
            row[2] = int(row[2]) - 1
            rgb_file.append(row)

with open(dest_path, 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(rgb_file)