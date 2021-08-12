import numpy as np
import json
import os
from PIL import Image

import data_converter



# write data
root_dir = "/mnt/sdc/GW/KITTI/data_tracking_image_2/testing/image_02"
det_dir = "/mnt/sdc/GW/KITTI/KITTICarTestDet"
save_path = "TrackAnnos/KITTI_test.json"


global_cnt = 0
track_data = {}
video_list = os.listdir(root_dir)
det_list = os.listdir(det_dir)
for n in range(len(det_list)): 

    video_name = det_list[n][:8]
    det_name = det_list[n][:-4]
    video_dir = root_dir+"/"+det_name

    det_path = det_dir+"/"+det_list[n]
    track_data, global_cnt = data_converter.convert_MOT(track_data, det_name, video_dir, det_path, None, "test", None, global_cnt)

track_data = data_converter.index_data(track_data)
print(global_cnt)

# write to json
with open(save_path, 'w') as outfile:
    json.dump(track_data, outfile)
