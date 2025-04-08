import h5py
import json
import os
import random
from scripts.convert_to_rlds import extract_step_data,process_step,create_episode


save_dir = "/home/tyj/Documents/310_jiarui/VLABench/debug_dataset"
dataset_root = "/home/tyj/Documents/310_jiarui/VLABench/media/select_fruit/data_100.hdf5" 
create_episode(dataset_root,save_dir,1)