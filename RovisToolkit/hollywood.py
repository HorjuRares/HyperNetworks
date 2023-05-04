"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

import numpy as np
import cv2
import os
import moviepy.video.io.ImageSequenceClip


image_folder = 'C:/data/tmp/viz/movie_sensing'
fps = 20 # 7
image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('C:/data/tmp/viz/sensing.mp4')


"""
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video = cv2.VideoWriter("C:/data/tmp/viz/panoptic.avi", fourcc, 20, (1920, 1080))

for i in range(787):
    file_semseg = r'C:/data/tmp/viz/semseg/{0:06d}.png'.format(i)
    file_keypoints = r'C:/data/tmp/viz/keypoints/{0:06d}.png'.format(i)
    img_semseg = cv2.imread(file_semseg)
    img_keypoints = cv2.imread(file_keypoints)
    img_viz = cv2.hconcat([img_semseg, img_keypoints])

    file_dst = r'C:/data/tmp/viz/movie_sensing/{0:06d}.png'.format(i)
    cv2.imwrite(file_dst, img_viz)
    print(file_dst)

    #video.write(img_viz)
    #print(i)

    cv2.imshow("viz", img_viz)
    cv2.waitKey(10)

#video.release()
"""
