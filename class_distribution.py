# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:22:34 2018

@author: NGN
"""

import pandas as pd

data = pd.read_csv('report.csv', header=None)

num_video_per_folder = data.groupby([0])[1].count()
folders = num_video_per_folder.index.values

num_video_per_class = data.groupby([3])[1].count()
classes = num_video_per_class.index.values

class_distribution = data.pivot_table(index=[0], columns=[3], values=1, aggfunc='count', fill_value=0)
frame_distribution = data.pivot_table(index=[0], columns=[2], values=1, aggfunc='count', fill_value=0)
frame_per_class_distribution = data.pivot_table(index=[0], columns=[3], values=[2], aggfunc='sum', fill_value=0)

class_distribution.to_csv('class_dis.csv')
frame_distribution.to_csv('frame_dis.csv')
frame_per_class_distribution.to_csv('frame_per_class_dis.csv')