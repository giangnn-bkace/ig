# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:06:43 2018

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
class_label = ["drink", "eat", "groom", "hang", "micromovement", "rear", "rest", "walk"]

def get_class_id(video_name):
    class_alias = video_name.split('_')[1]
    class_name = class_name_alias[class_alias]
    return class_label.index(class_name)


def main():
    data_path = os.path.join('data', 'clipped')
    dest_path = os.path.join('data', 'clipped_data')

    folders = os.listdir(data_path)
    rgb_file = []
    flow_file = []
    for folder in folders:
        print(folder)
        videos = glob.glob(os.path.join(data_path, folder, '*.mpg'))
        for video in videos:
            video_name = video.split(os.path.sep)[-1].split('.')[0]
            print(video_name)
            image_path = os.path.join(dest_path, 'rgb', video_name)
            flow_u_path = os.path.join(dest_path, 'tvl1', 'flow-u', video_name)
            flow_v_path = os.path.join(dest_path, 'tvl1', 'flow-v', video_name)
            flow_save_path = os.path.join(dest_path, 'tvl1', 'flow-{:s}', video_name)
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            if not os.path.exists(flow_u_path):
                os.mkdir(flow_u_path)
            if not os.path.exists(flow_v_path):
                os.mkdir(flow_v_path)
            images = glob.glob(os.path.join(data_path, folder, video_name, 'images', '*.jpg'))
            flow_u = glob.glob(os.path.join(data_path, folder, video_name, 'flow-u', '*.jpg'))
            flow_v = glob.glob(os.path.join(data_path, folder, video_name, 'flow-v', '*.jpg'))
            label = get_class_id(video_name)
            num_img = len(images)
            num_flow = len(flow_u) - 1
            for image in images:
                image_name = image.split(os.path.sep)[-1]
                os.rename(image, os.path.join(image_path, image_name))
            rgb_file.append([video_name, image_path, num_img, label])
            for flow_img in flow_u:
                flow_img_name = flow_img.split(os.path.sep)[-1]
                os.rename(flow_img, os.path.join(flow_u_path, flow_img_name))
            for flow_img in flow_v:
                flow_img_name = flow_img.split(os.path.sep)[-1]
                os.rename(flow_img, os.path.join(flow_v_path, flow_img_name))
            flow_file.append([video_name, flow_save_path, num_flow, label])

    with open(os.path.join(dest_path, 'rgb.csv'), 'w') as rgb_out:
        writer = csv.writer(rgb_out)
        writer.writerows(rgb_file)

    with open(os.path.join(dest_path, 'flow.csv'), 'w') as flow_out:
        writer = csv.writer(flow_out)
        writer.writerows(flow_file)
        
if __name__ == '__main__':
    main()