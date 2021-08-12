dataset = "KITTI"
sub_class = "car"
data_dir = "/mnt/sdc/GW/KITTI/data_tracking_image_2/testing/"

if dataset=="KITTI":
	if sub_class=="car":
		data_path = "TrackAnnos/KITTI_test.json"

	det_graph_model_load_path = "models/local_model_KITTI.tar" 
	tracklet_graph_model_load_path = "models/global_model_KITTI.tar"

	save_tracklet_graph_img_dir = "save_img_KITTI/test"
	save_txt_dir = "save_txt_KITTI/test"


save_img = True
save_txt = True
device = "cuda:0"


det_temporal_len = 16
tracklet_temporal_len = 64
T_det_window = 17
T_det_stride = 5
T_tracklet_stride = 5

batch_size = 1

emb_dist_thresh = 0.7
tracklet_associate_thresh = 0.2 
remove_N = 3
post_precessing = True 