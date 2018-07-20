# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:01:56 2018

@author: NGN
"""

"""
After moving all the files using the move_file.py, we run this one
to extract the images from the videos and also create a data file
we can use for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

#sequence_length = 20

def get_video_parts(video_path):
    # Given video path, return it parts
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]
    classname = parts[1]
    train_or_test = parts[0]
    
    return train_or_test, classname, filename_no_ext, filename

def get_nb_frames_for_video(video_parts):
    # Given video parts af an already extracted video 
    # return the number of frames that were extracted
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(train_or_test, classname, 
                                             filename_no_ext + '*.jpg'))
    
    return len(generated_files)

def check_already_extracted(video_parts):
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname, 
                                            filename_no_ext + '-0001.jpg')))
    
def extract_files():
    #data_file = []
    data_path = os.path.join('data','clipped')
    
    folders = os.listdir(data_path)
    #folders = ['20080322161942', '20080324124437', '20080420124815', '20080421163754', '20080421175111', '20080422174518', '20080423115530', '20080321154304']
    for folder in folders:
        videos = os.listdir(os.path.join(data_path, folder))
        
        for video in videos:
            src = os.path.join(data_path, folder, video)
            video_name = video.split('.')[0]
            dest_folder = os.path.join(data_path, folder, video_name, 'images')
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            dest = os.path.join(dest_folder, 'frame%06d.jpg')
            call(['ffmpeg', '-i', src, dest])
                    
                #nb_frames = get_nb_frames_for_video(video_parts)
                
    '''
                if (nb_frames >= sequence_length):
                    nb_item = nb_frames - sequence_length + 1
                    for i in range(nb_item):
                        data_file.append([train_or_test, classname, filename_no_ext, sequence_length, i+1])
                
                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))
                
                
    with open('new_data_file.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)
        
    print("Extracted and wrote %d video files." % (len(data_file)))
    '''
    
def main():
    extract_files()
    
if __name__ == '__main__':
    main()
