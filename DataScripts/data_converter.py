import numpy as np
import json
import os
from PIL import Image
from scipy.io import loadmat

def index_data(track_data):
    final_data = {}
    final_data['data'] = track_data
    final_data['data_index'] = {}
    cnt = 0
    for vid_id in track_data.keys():
        #import pdb; pdb.set_trace()
        for fr_id in track_data[vid_id]['images'].keys():
            final_data['data_index'][cnt] = {'video_name':vid_id, 
                'frame_id':fr_id}
            cnt += 1
    return final_data

def convert_MOT(track_data, video_name, video_dir, gt_path, convert_labels, gt_type="train", det_thresh=0.3, global_cnt=0):

    # construct video info
    if video_name in track_data.keys():
        return track_data
    video_info = {}
    video_info['video_dir'] = video_dir

    

    # load img
    video_info['images'] = {}
    img_list = os.listdir(video_dir)
    for n in range(len(img_list)):
        tmp_idx = img_list[n].rfind(".")
        tmp_idx2 = img_list[n].find("0")
        fr_idx = int(img_list[n][tmp_idx2:tmp_idx])
        video_info['images'][fr_idx] = {}
        video_info['images'][fr_idx]['image_name'] = img_list[n]
        video_info['images'][fr_idx]['annotations'] = []
        if 'width' not in video_info.keys():
            img_path = video_dir+"/"+img_list[n]
            img = Image.open(img_path)
            width, height = img.size
            video_info['width'] = width
            video_info['height'] = height
    frs = np.array(list(video_info['images'].keys()))
    video_info['start_frame'] = int(np.min(frs))
    video_info['end_frame'] = int(np.max(frs))

    # load gt
    track_gt = np.loadtxt(gt_path, delimiter=',', dtype=str)
    for n in range(len(track_gt)):
        if gt_type=="train":
            class_id = int(track_gt[n][7])
            ignored_flag = int(track_gt[n][6])
            if (class_id not in convert_labels.keys()):# or ignored_flag==0:
                continue
        else:
            # remove det with low detections scores
            if det_thresh is not None:
                if float(track_gt[n][6])<det_thresh:
                    continue

        # write to the dict
        tmp_ann = {}
        fr_id = int(track_gt[n][0])
        tmp_ann['object_id'] = int(track_gt[n][1])
        if gt_type=="train":
            tmp_ann['category_id'] = convert_labels[class_id]
        else:
            tmp_ann['category_id'] = 0
        xmin = float(track_gt[n][2])
        ymin = float(track_gt[n][3])
        xmax = float(track_gt[n][2])+float(track_gt[n][4])
        ymax = float(track_gt[n][3])+float(track_gt[n][5])
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(video_info['width']-1, xmax))
        ymax = int(min(video_info['height']-1, ymax))
        tmp_ann['bbox'] = [xmin, ymin, xmax, ymax]
        if xmax<=xmin+10 or ymax<=ymin+10:
            continue
        if gt_type=="train":
            tmp_ann['visibility_score'] = float(track_gt[n][8])
        else:
            tmp_ann['visibility_score'] = -1
        video_info['images'][fr_id]['annotations'].append(tmp_ann)
        global_cnt += 1

    track_data[video_name] = video_info

    return track_data, global_cnt

