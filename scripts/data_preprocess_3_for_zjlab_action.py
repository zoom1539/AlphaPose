import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import json



def parse_label(skeleton_path):
    labels = ['slap', 'kick', 'strike', 'fall_down',
              'walk_difficultly', 'squat_down', 'stand_up',
              'jump', 'run']
    
    file_name = skeleton_path.split('/')[-1].split('.')[0]
    label_index = int(file_name.split('S')[0].split('L')[-1])
    label = labels[label_index]
    return file_name, label, label_index


def listdir(dir, list_paths):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1]=='.json':
            list_paths.append(file_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/data1/zhumh/zjlab_actions/train_data/zjlab_action_val")
    parser.add_argument("--label_path", type=str, default="/data1/zhumh/zjlab_actions/train_data/zjlab_action_val_label.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    skeleton_paths = []
    listdir(args.work_dir, skeleton_paths)

    json_label = dict()

    for skeleton_path in tqdm(skeleton_paths):
        file_name, label, label_index = parse_label(skeleton_path)
        json_label[file_name] = dict(has_skeleton = True, label = label, label_index = label_index)
    
    with open(args.label_path, 'w') as json_writer:
        json.dump(json_label, json_writer)
        