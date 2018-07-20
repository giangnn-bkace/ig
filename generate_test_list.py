# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:45:10 2018

@author: NGN
"""

import os
import csv
import glob

data_path = os.path.join('data', 'clipped')
dest_path = os.path.join('data', 'clipped_data')
test_folders = os.listdir(data_path)

def main():
    for i, folder in enumerate(test_folders):
        test_file = []
        videos = glob.glob(os.path.join(data_path, folder, '*.mpg'))
        for video in videos:
            video_name = video.split(os.path.sep)[-1]
            test_file.append([os.path.join(folder, video_name)])

        with open(os.path.join(dest_path, 'testlist%02d.txt' %(i+1)), 'w') as fout:
            writer = csv.writer(fout)
            writer.writerows(test_file)

if __name__ == '__main__':
    main()