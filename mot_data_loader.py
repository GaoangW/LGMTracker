import numpy as np
import json

import torch.utils.data as data
import torch
import random

class TrackletSplit(object):
    def __init__(self, p0=0.9, p1=0.95, p2=0.9):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def __call__(self, sample):
        uniq_ids = np.unique(sample['obj_ids'])
        uniq_ids = uniq_ids[uniq_ids!=-1]

        remove_idx = []
        for n in range(len(uniq_ids)):
            tmp_idx = np.where(sample['obj_ids']==uniq_ids[n])[0]
            rand_tmp = random.random()
            if rand_tmp<self.p0:
                v = True
            else:
                v = False

            for t in range(len(tmp_idx)):
                rand_tmp = random.random()
                if v:
                    if rand_tmp>self.p1:
                        remove_idx.append(tmp_idx[t])
                        v = False
                else:
                    if rand_tmp<self.p2:
                        remove_idx.append(tmp_idx[t])
                    else:
                        v = True

        sample['boxes'] = np.delete(sample['boxes'], remove_idx, 0)
        sample['gt_boxes'] = np.delete(sample['gt_boxes'], remove_idx, 0)
        sample['classes'] = np.delete(sample['classes'], remove_idx, 0)
        sample['obj_ids'] = np.delete(sample['obj_ids'], remove_idx, 0)
        sample['fr_ids'] = np.delete(sample['fr_ids'], remove_idx, 0)
        return sample


class AddFP(object):
    def __init__(self, temporal_len, fpr=0.1):
        self.fpr = fpr
        self.temporal_len = temporal_len

    def __call__(self, sample):
        h = sample['height']
        w = sample['width']
        
        for t in range(self.temporal_len):
            box_w = (sample['boxes'][:, 2]-sample['boxes'][:, 0]).copy()
            box_h = (sample['boxes'][:, 3]-sample['boxes'][:, 1]).copy()

            cand_ids = np.where(sample['fr_ids']==t)[0]
            if len(cand_ids)<2:
                continue

            fp_num = 0
            for k in range(len(cand_ids)):
                if np.random.rand()<self.fpr:
                    fp_num += 1

            if fp_num==0:
                continue

            mean_x = np.mean(sample['boxes'][cand_ids, 0])
            mean_y = np.mean(sample['boxes'][cand_ids, 1])
            mean_w = np.mean(box_w[cand_ids])
            mean_h = np.mean(box_h[cand_ids])
            std_x = np.std(sample['boxes'][cand_ids, 0])
            std_y = np.std(sample['boxes'][cand_ids, 1])
            std_w = np.std(box_w[cand_ids])
            std_h = np.std(box_h[cand_ids])

            xx = np.random.normal(mean_x, std_x, fp_num)
            yy = np.random.normal(mean_y, std_y, fp_num)
            ww = np.clip(np.random.normal(mean_w, std_w, fp_num), 0, w)
            hh = np.clip(np.random.normal(mean_h, std_h, fp_num), 0, h)

            fp_box = np.stack([xx, yy, xx+ww, yy+hh], 1).astype(np.int32)

            # concatenate according to fr order
            last_fr_idx = cand_ids[-1]+1
            sample['boxes'] = np.concatenate([sample['boxes'][:last_fr_idx], fp_box, sample['boxes'][last_fr_idx:]], 0)
            sample['classes'] = np.concatenate([sample['classes'][:last_fr_idx], np.zeros((fp_num), dtype=np.int32), sample['classes'][last_fr_idx:]], 0)
            sample['obj_ids'] = np.concatenate([sample['obj_ids'][:last_fr_idx], -np.ones((fp_num), dtype=np.int32), sample['obj_ids'][last_fr_idx:]], 0)
            sample['fr_ids'] = np.concatenate([sample['fr_ids'][:last_fr_idx], (t*np.ones((fp_num), dtype=np.int32)).astype(np.int32), sample['fr_ids'][last_fr_idx:]], 0)

        return sample


class BoxJitter(object):
    def __init__(self, jitter_ratio=0.2):
        self.jitter_ratio = jitter_ratio

    def __call__(self, sample):
        sample['gt_boxes'] = sample['boxes'].copy()
        h = sample['height']
        w = sample['width']
        box_w = (sample['boxes'][:, 2]-sample['boxes'][:, 0]).copy()
        box_h = (sample['boxes'][:, 3]-sample['boxes'][:, 1]).copy()
        sample['boxes'][:, 0::2] = sample['boxes'][:, 0::2]+np.expand_dims(box_w, 1)*self.jitter_ratio*(np.random.rand(len(sample['boxes']), 2)-0.5)*2
        sample['boxes'][:, 1::2] = sample['boxes'][:, 1::2]+np.expand_dims(box_h, 1)*self.jitter_ratio*(np.random.rand(len(sample['boxes']), 2)-0.5)*2
        return sample


class BoxShift(object):
    def __init__(self, shift_ratio=0.2):
        self.shift_ratio = shift_ratio

    def __call__(self, sample):
        h = sample['height']
        w = sample['width']
        h_shift = self.shift_ratio*(random.random()-0.5)*2*h
        w_shift = self.shift_ratio*(random.random()-0.5)*2*w
        sample['boxes'][:, 0::2] = sample['boxes'][:, 0::2]+w_shift
        sample['boxes'][:, 1::2] = sample['boxes'][:, 1::2]+h_shift
        return sample


class BoxClip(object):
    def __call__(self, sample):
        h = sample['height']
        w = sample['width']
        sample['boxes'][:, 0::2] = np.clip(sample['boxes'][:, 0::2], 0, w)
        sample['boxes'][:, 1::2] = np.clip(sample['boxes'][:, 1::2], 0, h)
        remove_idx = np.where((sample['boxes'][:, 0]>=w)+(sample['boxes'][:, 1]>=h)+(sample['boxes'][:, 2]<=0)+(sample['boxes'][:, 3]<=0)>0)[0]
        sample['boxes'] = np.delete(sample['boxes'], remove_idx, 0)
        if 'gt_boxes' in sample.keys():
            sample['gt_boxes'] = np.delete(sample['gt_boxes'], remove_idx, 0)
        sample['classes'] = np.delete(sample['classes'], remove_idx, 0)
        sample['obj_ids'] = np.delete(sample['obj_ids'], remove_idx, 0)
        sample['fr_ids'] = np.delete(sample['fr_ids'], remove_idx, 0)
        return sample


class HFlip(object):

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, sample):
        h = sample['height']
        w = sample['width']
        flip_prob = random.random()
        if flip_prob<self.flip_ratio:
            return sample
        else:
            tmp_x1 = w-sample['boxes'][:, 0].copy()
            tmp_x2 = w-sample['boxes'][:, 2].copy()
            sample['boxes'][:, 0] = tmp_x2
            sample['boxes'][:, 2] = tmp_x1
            return sample

class RandomDelete(object):

    def __init__(self, delete_ratio=0.15, max_bbox=2000):
        self.delete_ratio = delete_ratio
        self.max_bbox = max_bbox

    def __call__(self, sample):
        N = len(sample['obj_ids'])
        idx_arr = np.linspace(0, N-1, N, dtype=np.int32)
        np.random.shuffle(idx_arr)
        if N>self.max_bbox:
            remove_idx = idx_arr[:N-self.max_bbox]
        else:
            remove_idx = idx_arr[:int(N*self.delete_ratio)]
        sample['boxes'] = np.delete(sample['boxes'], remove_idx, 0)
        sample['classes'] = np.delete(sample['classes'], remove_idx, 0)
        sample['obj_ids'] = np.delete(sample['obj_ids'], remove_idx, 0)
        sample['fr_ids'] = np.delete(sample['fr_ids'], remove_idx, 0)
        
        return sample


class CreateMOTDataset(data.Dataset):
    def __init__(self, gt_path, temporal_len=64, transform=None, max_id=300):
        self.temporal_len = temporal_len
        with open(gt_path) as json_file:
            self.track_data = json.load(json_file)
        self.transform = transform
        seqs = self.track_data['data'].keys()
        self.vids = []
        for key in seqs:
            self.vids.append(key)
        self.max_id = max_id


    def __len__(self):
        if self.temporal_len==-1:
            return len(self.track_data['data'].keys())
        else:
            return len(self.track_data['data_index'].keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        seq_boxes = []
        seq_fr_ids = []
        seq_classes = []
        seq_obj_ids = []
        seq_img_paths = []
        if self.temporal_len!=-1:
            vid = self.track_data['data_index'][str(idx)]['video_name']
            fr_id = self.track_data['data_index'][str(idx)]['frame_id']
            video_st_fr = self.track_data['data'][str(vid)]['start_frame']
            video_end_fr = self.track_data['data'][str(vid)]['end_frame']
            st_fr = fr_id-self.temporal_len//2
            end_fr = fr_id+self.temporal_len//2
        else:
            vid = self.vids[idx]
            video_st_fr = self.track_data['data'][str(vid)]['start_frame']
            video_end_fr = self.track_data['data'][str(vid)]['end_frame']
            st_fr = video_st_fr
            end_fr = video_end_fr

        fr_cnt = 0
        for n in range(st_fr, end_fr+1):

            cur_fr = n
            if n<video_st_fr:
                cur_fr = video_st_fr
            if n>video_end_fr:
                cur_fr = video_end_fr
            seq_img_paths.append(self.track_data['data'][str(vid)]['video_dir']+\
                "/"+self.track_data['data'][vid]['images'][str(cur_fr)]['image_name'])
            if 'height' not in sample.keys():
                sample['height'] = self.track_data['data'][str(vid)]['height']
                sample['width'] = self.track_data['data'][str(vid)]['width']


            boxes = []
            classes = []
            obj_ids = []
            img_names = []
            for m in range(len(self.track_data['data'][str(vid)]['images'][str(cur_fr)]['annotations'])):
                boxes.append(self.track_data['data'][str(vid)]['images'][str(cur_fr)]['annotations'][m]['bbox'])
                classes.append(self.track_data['data'][str(vid)]['images'][str(cur_fr)]['annotations'][m]['category_id']-1)
                obj_ids.append(self.track_data['data'][str(vid)]['images'][str(cur_fr)]['annotations'][m]['object_id'])
            
            if len(boxes)==0:
                fr_cnt += 1
                continue
            
            seq_boxes.append(np.array(boxes))
            seq_classes.append(np.array(classes))
            seq_obj_ids.append(np.array(obj_ids))
            seq_fr_ids.append(fr_cnt*np.ones((len(obj_ids))))
            fr_cnt += 1

        if len(seq_boxes)>0:
            seq_boxes = np.concatenate(seq_boxes, 0)
            seq_classes = np.concatenate(seq_classes, 0)
            seq_obj_ids = np.concatenate(seq_obj_ids, 0)
            seq_fr_ids = np.concatenate(seq_fr_ids, 0)
        sample['boxes'] = seq_boxes
        sample['classes'] = seq_classes
        sample['obj_ids'] = seq_obj_ids
        if len(seq_fr_ids)>0:
            sample['fr_ids'] = seq_fr_ids.astype(np.int32)
        sample['img_paths'] = seq_img_paths

        if self.transform and len(seq_boxes)>0:
            sample = self.transform(sample)
        
        return sample