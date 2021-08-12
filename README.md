# LGMTracker

## Data Pre-Processing
Set "root_dir", "det_dir" and "save_path" from "DataScripts/MOT_converter.py" to the directories of your input KITTI video sequences, detection files and output json file. Create the detection json file based on the following command.
```
python DataScripts/MOT_converter.py
```
## LGM Tracking
From "config.py", set "data_dir" to the directory of your input KITTI video sequences; set "data_path" to the json file path from the previous step; set "save_tracklet_graph_img_dir" and "save_txt_dir" to your tracking output directories. Run the tracker based on the following command.
```
python test_mot.py
```
