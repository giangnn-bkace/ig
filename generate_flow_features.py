# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:17:50 2018

@author: NGN
"""

import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool


_IMAGE_SIZE = 256


def cal_for_frames(video_path):
    print(video_path)
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "frame{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "frame{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def gen_video_path():
    path = []
    flow_path = []
    length = []
        
    data_path = os.path.join('data','clipped')
    #folders = ['20080322161942', '20080324124437', '20080420124815', '20080421163754', '20080421175111', '20080422174518', '20080423115530', '20080321154304']
    folders = os.listdir(data_path)
    for folder in folders:
        videos = glob(os.path.join(data_path, folder, '*.mpg'))
        for video in videos:
            print(video)
            video_name = video.split(os.path.sep)[-1]
            video_name_no_ext = video_name.split('.')[0]
            tmp_path = os.path.join(data_path, folder, video_name_no_ext, 'images')
            tmp_flow = os.path.join(data_path, folder, video_name_no_ext, 'flow-{:s}')
            tmp_len = len(glob(os.path.join(tmp_path, '*.jpg')))
            u = False
            v = False
            if os.path.exists(tmp_flow.format('u')):
                if len(glob(os.path.join(tmp_flow.format('u'), '*.jpg'))) == tmp_len:
                    u = True
            else:
                os.makedirs(tmp_flow.format('u'))
            if os.path.exists(tmp_flow.format('v')):
                if len(glob(os.path.join(tmp_flow.format('v'), '*.jpg'))) == tmp_len:
                    v = True
            else:
                os.makedirs(tmp_flow.format('v'))
            if u and v:
                #print('skip:' + tmp_flow)
                continue

            path.append(tmp_path)
            flow_path.append(tmp_flow)
            length.append(tmp_len)
    return path, flow_path, length


def extract_flow(args):
    video_path, flow_path, video_length = args
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    print(video_length)
    return


if __name__ =='__main__':
    pool = Pool(2)   # multi-processing

    video_paths, flow_paths, video_lengths = gen_video_path()

    pool.map(extract_flow, zip(video_paths, flow_paths, video_lengths))