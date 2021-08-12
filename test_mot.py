import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import cv2
import os
import time

from mot_data_loader import CreateMOTDataset, HFlip, BoxShift, BoxClip, BoxJitter, AddFP
import build_det_graph
import head_utils, head_gnn

from config import *

def det_graph_predictor(node_input, model):

	# build graph
	edge_idx, A = build_det_graph.build_adj_graph(node_input['fr_ids'], node_input['det_embs'])

	# inference
	with torch.no_grad():
		graph_emb, _ = model(node_input['det_embs'], A, edge_idx, None)

		# get dist of embs
		tmp_idx = torch.nonzero(edge_idx[0, :]<edge_idx[1, :])
		transf_edge_idx = edge_idx[:, tmp_idx[:, 0]]
		dist = torch.norm(graph_emb[transf_edge_idx[0, :], :]-graph_emb[transf_edge_idx[1, :], :], dim=1)

	return dist

def batch_test_tracklet_graph(batch_data, det_graph_model, tracklet_graph_model, post_precessing, remove_N):

	batch_pred_tracklet_label = []

	for b in range(len(batch_data)):

		# convert to cuda
		if 'fr_ids' not in batch_data[b].keys():
			return []
		fr_ids = batch_data[b]['fr_ids'][0].to(device).float()
		bbox = batch_data[b]['boxes'][0].to(device).float()
		bbox = bbox.float()
		bbox[:, 0::2] = bbox[:, 0::2]/float(batch_data[b]['width'].item())
		bbox[:, 1::2] = bbox[:, 1::2]/float(batch_data[b]['height'].item())
		scores = torch.ones(len(bbox), device=device)

		# graph stat
		N_node = len(bbox)
		fr_ids1 = torch.unsqueeze(fr_ids, 1)
		fr_ids2 = torch.unsqueeze(fr_ids, 0)
		delta_fr_ids = fr_ids1.cpu()-fr_ids2.cpu()
		det_edge_idx = torch.nonzero(delta_fr_ids==-1).to(device)
		agg_dist = torch.zeros(len(det_edge_idx), device=device)
		agg_cnt = torch.zeros(len(det_edge_idx), device=device)

		max_fr = torch.max(fr_ids).item()
		t_fr = 0
		end_flag = 0

		while True:
			if t_fr+T_det_window<=max_fr+1:
				st_fr = t_fr
				end_fr = t_fr+T_det_window
			else:
				st_fr = max_fr+1-T_det_window
				end_fr = max_fr+1
				end_flag = 1

			cand_ids = torch.nonzero((fr_ids>=st_fr)*(fr_ids<end_fr))
			tmp_det_embs = bbox[cand_ids[:, 0]]
			tmp_fr_ids = fr_ids[cand_ids[:, 0]]-st_fr

			node_input = {"det_embs": tmp_det_embs, "fr_ids": tmp_fr_ids}

			tmp_dist = det_graph_predictor(node_input, det_graph_model)

			try:
				tmp_edge_st_idx = torch.nonzero((det_edge_idx[:, 0]>=cand_ids[0, 0])*(det_edge_idx[:, 0]<=cand_ids[-1, 0]))[0, 0]
			except:
				t_fr += T_det_stride
				if end_flag==1:
					break
				continue

			agg_dist[tmp_edge_st_idx:tmp_edge_st_idx+len(tmp_dist)] += tmp_dist
			agg_cnt[tmp_edge_st_idx:tmp_edge_st_idx+len(tmp_dist)] += 1

			t_fr += T_det_stride
			if end_flag==1:
				break

		avg_dist = agg_dist/agg_cnt

		if len(avg_dist)==0:
			N_track = len(fr_ids)
			final_bbox_labels = np.linspace(0, N_track-1, N_track, dtype=np.int32)
			batch_pred_tracklet_label.append(final_bbox_labels)
			return batch_pred_tracklet_label
		
		try:
			pred_edge_label = head_utils.get_pred_edge_label(len(fr_ids), avg_dist, det_edge_idx, emb_dist_thresh, device)
		except:
			import pdb; pdb.set_trace()
		tracklet_label = head_utils.get_tracklet_label(len(fr_ids), pred_edge_label, det_edge_idx)


		###################
		uniq_tracklet_labels = np.unique(tracklet_label)
		N_tracklet = len(uniq_tracklet_labels)
		total_D = np.zeros((N_tracklet, N_tracklet))
		total_cnt = np.zeros((N_tracklet, N_tracklet))
		non_A = np.zeros((N_tracklet, N_tracklet))

		###################
		# split data into sliding windows
		max_fr = torch.max(fr_ids).item()
		t_fr = 0
		end_flag = 0
		T_tracklet_window = tracklet_temporal_len+1
		debug_cnt = 0
		while True:
			if t_fr+T_tracklet_window<=max_fr+1:
				st_fr = t_fr
				end_fr = t_fr+T_tracklet_window
			else:
				st_fr = max_fr+1-T_tracklet_window
				end_fr = max_fr+1
				end_flag = 1


			# select tracklets in the window
			tmp_fr_ids = fr_ids.cpu()
			select_idx = np.where((tmp_fr_ids>=st_fr)*(tmp_fr_ids<end_fr))[0]
			select_tracklet_label = tracklet_label[select_idx]
			select_fr_ids = fr_ids[select_idx]
			select_bbox = bbox[select_idx]
			select_scores = scores[select_idx]


			if len(select_tracklet_label)==0:
				t_fr += T_tracklet_stride
				if end_flag==1:
					break
				continue

			try:
				tracklet_data = head_utils.get_tracklet_info(select_tracklet_label, None, select_fr_ids-st_fr, select_bbox, None, select_scores, tracklet_temporal_len, device, "test")
			except:
				import pdb; pdb.set_trace()

			tmp_uniq_tracklet_label = tracklet_data['uniq_tracklet_label']

			if tracklet_data['A'].shape[0]<2 or torch.max(tracklet_data['A']).item()<1:
				t_fr += T_tracklet_stride
				if end_flag==1:
					break
				continue

			# get the dist
			try:
				with torch.no_grad():
					tmp_Dist = tracklet_graph_model(tracklet_data, "test")
			except:
				import pdb; pdb.set_trace()

			tmp_Dist = tmp_Dist.cpu().detach().numpy()
			A = tracklet_data['A'].cpu().detach().numpy()
			

			# map the tmp uniq id to the total id idx
			tmp_map_idx = []
			for n in range(len(tmp_Dist)):
				tmp_map_idx.append(np.where(uniq_tracklet_labels==tmp_uniq_tracklet_label[n])[0][0])
			for n in range(len(tmp_Dist)):
				total_D[tmp_map_idx[n], tmp_map_idx] += tmp_Dist[n,:]
				total_cnt[tmp_map_idx[n], tmp_map_idx] += A[n,:]
				non_A[tmp_map_idx[n], tmp_map_idx] += 1-A[n,:]
			

			t_fr += T_tracklet_stride
			if end_flag==1:
				break
			debug_cnt += 1

		total_D[total_cnt==0] = 100
		total_cnt[total_cnt==0] = 1
		avg_Dist = total_D/total_cnt


		final_tracklet_labels = head_utils.associate_tracklet(avg_Dist, non_A, uniq_tracklet_labels, thresh=tracklet_associate_thresh)
		
		final_bbox_labels = tracklet_label.copy()
		for n in range(len(final_bbox_labels)):
			final_bbox_labels[n] = final_tracklet_labels[int(final_bbox_labels[n])]

		if post_precessing:
			new_fr_ids, final_bbox_labels, new_bbox = head_utils.interp(fr_ids.cpu().detach().numpy().astype(int), final_bbox_labels.astype(int), bbox.cpu().detach().numpy(), remove_N)
			batch_pred_tracklet_label.append(final_bbox_labels)
			new_bbox[:, 0::2] = new_bbox[:, 0::2]*float(batch_data[b]['width'].item())
			new_bbox[:, 1::2] = new_bbox[:, 1::2]*float(batch_data[b]['height'].item())
			return batch_pred_tracklet_label, new_fr_ids, new_bbox
		else:
			batch_pred_tracklet_label.append(final_bbox_labels)
			return batch_pred_tracklet_label, [], []

def batch_test_det_graph(batch_data, det_graph_model):

	batch_pred_tracklet_label = []

	for b in range(len(batch_data)):

		# convert to cuda
		fr_ids = batch_data[b]['fr_ids'][0].to(device).float()
		bbox = batch_data[b]['boxes'][0].to(device).float()
		bbox = bbox.float()
		bbox[:, 0::2] = bbox[:, 0::2]/float(batch_data[b]['width'].item())
		bbox[:, 1::2] = bbox[:, 1::2]/float(batch_data[b]['height'].item())
		scores = torch.ones(len(bbox), device=device)

		# graph stat
		N_node = len(bbox)
		fr_ids1 = torch.unsqueeze(fr_ids, 1)
		fr_ids2 = torch.unsqueeze(fr_ids, 0)
		delta_fr_ids = fr_ids1.cpu()-fr_ids2.cpu()
		det_edge_idx = torch.nonzero(delta_fr_ids==-1).to(device)
		agg_dist = torch.zeros(len(det_edge_idx), device=device)
		agg_cnt = torch.zeros(len(det_edge_idx), device=device)

		max_fr = torch.max(fr_ids).item()
		t_fr = 0
		end_flag = 0

		while True:
			if t_fr+T_det_window<=max_fr+1:
				st_fr = t_fr
				end_fr = t_fr+T_det_window
			else:
				st_fr = max_fr+1-T_det_window
				end_fr = max_fr+1
				end_flag = 1

			cand_ids = torch.nonzero((fr_ids>=st_fr)*(fr_ids<end_fr))
			tmp_det_embs = bbox[cand_ids[:, 0]]
			tmp_fr_ids = fr_ids[cand_ids[:, 0]]-st_fr

			node_input = {"det_embs": tmp_det_embs, "fr_ids": tmp_fr_ids}

			tmp_dist = det_graph_predictor(node_input, det_graph_model)

			tmp_edge_st_idx = torch.nonzero((det_edge_idx[:, 0]>=cand_ids[0, 0])*(det_edge_idx[:, 0]<=cand_ids[-1, 0]))[0, 0]
			agg_dist[tmp_edge_st_idx:tmp_edge_st_idx+len(tmp_dist)] += tmp_dist
			agg_cnt[tmp_edge_st_idx:tmp_edge_st_idx+len(tmp_dist)] += 1

			t_fr += T_det_stride
			if end_flag==1:
				break

		avg_dist = agg_dist/agg_cnt
		pred_edge_label = head_utils.get_pred_edge_label(len(fr_ids), avg_dist, det_edge_idx, emb_dist_thresh, device)
		tracklet_label = head_utils.get_tracklet_label(len(fr_ids), pred_edge_label, det_edge_idx)
		batch_pred_tracklet_label.append(tracklet_label)
		
	return batch_pred_tracklet_label
		

def test_tracklet_graph():
	# color table
	color_table = np.random.rand(5000, 3)

	# det graph model
	det_graph_model = build_det_graph.DetGraphEmb(device)
	det_graph_model.to(device)
	det_graph_checkpoint = torch.load(det_graph_model_load_path, map_location=device)
	det_graph_model.load_state_dict(det_graph_checkpoint['model_state_dict'])
	det_graph_model.eval()

	# tracklet graph model
	tracklet_graph_model = head_gnn.BoxEmb(tracklet_temporal_len+1, device)
	tracklet_graph_model = tracklet_graph_model.to(device)
	tracklet_graph_checkpoint = torch.load(tracklet_graph_model_load_path, map_location=device)
	tracklet_graph_model.load_state_dict(tracklet_graph_checkpoint['model_state_dict'])
	tracklet_graph_model.eval()

	# data loader
	comp_transforms = transforms.Compose([BoxClip()])
	mot_data = CreateMOTDataset(data_path, -1, transform=comp_transforms)
	dataloader = DataLoader(mot_data, batch_size=1,
                        shuffle=False, num_workers=4)

	# prediction
	seq_cnt = 0
	batch_data = []
	for i_batch, sample in enumerate(dataloader):
		seq_cnt += 1
		
		batch_data.append(sample)
		st_time = time.time()
		batch_pred_tracklet_label, new_fr_ids, new_bbox = batch_test_tracklet_graph(batch_data, det_graph_model, tracklet_graph_model, post_precessing, remove_N)
		end_time = time.time()
		batch_data = []
		if len(batch_pred_tracklet_label)==0:
			continue
		tmp_label = batch_pred_tracklet_label[0].copy()

		if post_precessing:
			sample['fr_ids'] = torch.from_numpy(new_fr_ids).unsqueeze(0)
			sample['boxes'] = torch.from_numpy(new_bbox).unsqueeze(0)

		# assign new label
		uniq_label = np.unique(tmp_label)
		for n in range(len(uniq_label)):
			batch_pred_tracklet_label[0][tmp_label==uniq_label[n]] = n
		final_N = len(uniq_label)

		# save tracklet graph results
		tmp_str = sample['img_paths'][0][0].split("/")
		if dataset=="KITTI":
			if not os.path.exists(save_tracklet_graph_img_dir):
				os.mkdir(save_tracklet_graph_img_dir)
			if sub_class=="car":
				seq_path = save_tracklet_graph_img_dir+"/car"
			elif sub_class=="person":
				seq_path = save_tracklet_graph_img_dir+"/person"
			if not os.path.exists(seq_path):
				os.mkdir(seq_path)
			seq_path = seq_path+"/"+tmp_str[-2]
		elif dataset=="MOT":
			seq_path = save_tracklet_graph_img_dir+"/"+tmp_str[-3]
		elif dataset=="UADETRAC":
			seq_path = save_tracklet_graph_img_dir+"/"+tmp_str[-2]
		
		if save_img:
			if not os.path.exists(seq_path):
				os.mkdir(seq_path)

		if save_txt:
			if not os.path.exists(save_txt_dir):
				os.mkdir(save_txt_dir)

		if dataset=="KITTI":
			if sub_class=="car":
				save_txt_path = save_txt_dir+"/car"
			elif sub_class=="person":
				save_txt_path = save_txt_dir+"/person"
			if not os.path.exists(save_txt_path):
				os.mkdir(save_txt_path)
			save_txt_path = save_txt_path+"/"+tmp_str[-2]+".txt"

			if os.path.exists(save_txt_path):
				os.remove(save_txt_path)


		if save_txt==True:
			if dataset=="KITTI":
				txt_file = open(save_txt_path, "a")

		T = len(sample['img_paths'])

		for n in range(len(sample['img_paths'])):
			
			img_path = sample['img_paths'][n][0]
			tmp_split = img_path.split("/")
			tmp_idx2 = tmp_split[-1].find("0")
			fr_id = int(tmp_split[-1][tmp_idx2:-4])

			if save_img:
				if data_dir is not None:
					tt_idx = img_path.find("image_02")
					img_path = data_dir+img_path[tt_idx:]
				img = cv2.imread(img_path)

				if n==0:
					height, width, layers = img.shape
					img_size = (width, height)
					out_video = cv2.VideoWriter(seq_path+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 10, img_size)

			tmp_idx = sample['fr_ids']==n
			tmp_bboxes = sample['boxes'][tmp_idx].detach().numpy()
			tmp_track_ids = batch_pred_tracklet_label[0][tmp_idx.detach().numpy()[0]]
			for k in range(len(tmp_bboxes)):

				tmp_id = int(tmp_track_ids[k])
				if save_img:
					font = cv2.FONT_HERSHEY_SIMPLEX
					img = cv2.rectangle(img, (int(tmp_bboxes[k][0]), int(tmp_bboxes[k][1])), (int(tmp_bboxes[k][2]), int(tmp_bboxes[k][3])), 255*color_table[tmp_id], 2)
					img = cv2.putText(img, str(tmp_id), (int(tmp_bboxes[k][0]), int(tmp_bboxes[k][1])), font, 1.2, 255*color_table[tmp_id], 2)

				if save_txt==True:
					if dataset=="KITTI":
						if sub_class=="car":
							txt_file.write('%i %i %s %i %i %i %.2f %.2f %.2f %.2f %i %i %i %i %i %i %i %.2f\n' % (fr_id, tmp_id, "Car", -1, -1, -10, tmp_bboxes[k][0], tmp_bboxes[k][1],
								tmp_bboxes[k][2], tmp_bboxes[k][3], -1, -1, -1, -1000, -1000, -1000, -10, 1.0))
						if sub_class=="person":
							txt_file.write('%i %i %s %i %i %i %.2f %.2f %.2f %.2f %i %i %i %i %i %i %i %.2f\n' % (fr_id, tmp_id, "Pedestrian", -1, -1, -10, tmp_bboxes[k][0], tmp_bboxes[k][1],
								tmp_bboxes[k][2], tmp_bboxes[k][3], -1, -1, -1, -1000, -1000, -1000, -10, 1.0))

			save_path = seq_path+"/"+tmp_split[-1]
			if save_img:
				cv2.imwrite(save_path, img)
				out_video.write(img)

		if save_txt==True:
			if dataset=="KITTI":
				txt_file.close()

		if save_img:
			out_video.release()

if __name__ == "__main__":
	test_tracklet_graph()