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

def extract_files():
    #data_file = []
    data_path = os.path.join('data','full')
    
    videos = ['20080421153635.mpg', '20080422170445.mpg', '20080423191834A.mpg', '20080424092813.mpg', '20080428161640.mpg', 'agoutivideo320080229A.mpg', 'agoutivideo320080229F.mpg']
    for video in videos:
        src = os.path.join(data_path, video)
        video_name = video.split('.')[0]
        dest_folder = os.path.join(data_path, video_name, 'images')
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
