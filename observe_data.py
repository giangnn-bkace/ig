# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:49:07 2018

@author: NGN
"""
import os
import glob
import csv

class_name_alias = {"drink": "drink", "d": "drink", 
                    "eat": "eat", "e": "eat",
                    "groomback": "groom", "groom": "groom", "gb": "groom", "g": "groom",
                    "hang": "hang", "ha": "hang",
                    "head": "micromovement", "he": "micromovement",
                    "rear": "rear", "r": "rear",
                    "rest": "rest", "rs": "rest",
                    "walk": "walk", "w": "walk"}

data_path = os.path.join('data', 'clipped_flow+images')

report_file = []

folders = os.listdir(data_path)

for folder in folders:
    videos = glob.glob(os.path.join(data_path, folder, '*.mpg'))
    for video in videos:
        video_name = video.split(os.path.sep)[-1]
        video_name_no_ext = video_name.split('.')[0]
        video_class_alias = video_name_no_ext.split('_')[1]
        video_class = class_name_alias[video_class_alias]
        frames = glob.glob(os.path.join(data_path, folder, video_name_no_ext, 'images', '*.jpg'))
        num_frame = len(frames)
        report_file.append([folder, video_name, num_frame, video_class])
        print(video_name_no_ext)

with open('report.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(report_file)