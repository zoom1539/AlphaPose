from __future__ import division

import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_img, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools


class Detector(object):
    def __init__(self):
        batch_size = 1
        self.confidence = 0.5
        self.nms_thesh = 0.4

        self.CUDA = torch.cuda.is_available()

        self.num_classes = 80
        classes = load_classes('data/coco.names') 

        #Set up the neural network
        print("Loading network.....")
        self.model = Darknet("cfg/yolov3-spp.cfg")
        self.model.load_weights("data/yolov3-spp.weights")
        print("Network successfully loaded")

        self.model.net_info["height"] = "608"
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        #If there's a GPU availible, put the model on GPU
        if self.CUDA:
            self.model.cuda()

        #Set the model in evaluation mode
        self.model.eval()


    def detect(self, img):

        batches = list(map(prep_img, [img], [self.inp_dim]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if self.CUDA:
            im_dim_list = im_dim_list.cuda()

        for batch in im_batches:
            #load the image
            if self.CUDA:
                batch = batch.cuda()
            with torch.no_grad():
                parser = argparse.ArgumentParser(description='AlphaPose Demo')
                args = parser.parse_args()
                args.device = torch.device("cuda:0")
                # args.device = torch.device("cpu")
                prediction = self.model(Variable(batch), args)

            prediction = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)
            output = prediction

            if self.CUDA:
                torch.cuda.synchronize()

        bboxes_xywh = []
        cls_confs = []
        cls_ids = []

        if output is None:
            return np.array(bboxes_xywh), np.array(cls_confs), np.array(cls_ids)
        # print(im_dim_list)
        # print(output)
        # print('----')

        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

        scaling_factor = torch.min(self.inp_dim/im_dim_list,1)[0].view(-1,1)


        output[:,[1,3]] -= (self.inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (self.inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

        
        for item in output:
            item = item.cpu()
            bbox_xywh = [item[1].int().item(), 
                         item[2].int().item(), 
                         item[3].int().item() - item[1].int().item(), 
                         item[4].int().item() - item[2].int().item()]
            bboxes_xywh.append(bbox_xywh)

            cls_confs.append(item[5].int().item())
            cls_ids.append(item[-1].int().item())

        return np.array(bboxes_xywh), np.array(cls_confs), np.array(cls_ids)
    

# if __name__ == '__main__':
#     detector = Detector()
#     img = cv2.imread("imgs/messi.jpg")
#     bboxes_xywh, cls_confs, cls_ids = detector.detect(img)

#     c1 = (bboxes_xywh[0][0], bboxes_xywh[0][1])
#     c2 = (bboxes_xywh[0][0] + bboxes_xywh[0][2], bboxes_xywh[0][1] + bboxes_xywh[0][3])
#     color = (255,0,0)
#     cv2.rectangle(img, c1, c2, color, 1)

#     cv2.imwrite('result.jpg', img)

#     print(bboxes_xywh)


class Tracker(object):
    def __init__(self, args):
        self.args = args

        # self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        use_cuda = self.args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        self.snippet_id = 0

        self.detector = Detector()

    def run(self, video_path):
        #
        video = cv2.VideoCapture()
        video.open(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        resolution = (width, height)

        #
        bbox_dict = dict()

        save_path_video = video_path.replace('.avi', '_bbox_.avi')
        videoWriter = cv2.VideoWriter(
            save_path_video, fourcc, fps, resolution)

        frame_id = -1
        while video.grab():
            frame_id += 1
            # print('frame_id: ',frame_id)

            _, img_bgr = video.retrieve()
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            bboxes_xywh, cls_confs, cls_ids = self.detector.detect(img_bgr)
            
            # print(bboxes_xywh)
            # print(cls_ids)
            
            if bboxes_xywh is not None:
                mask = (cls_ids == 0)
                bboxes_xywh_person = bboxes_xywh[mask]
                cls_conf_person = cls_confs[mask]
                
                # print(mask)
                # print(bboxes_xywh_person)

                bboxes_xyxy_person = list()
                for bbox_xywh in bboxes_xywh_person:
                    
                    xmin = int(bbox_xywh[0] - 0.1 * bbox_xywh[2])
                    xmin = 0 if xmin < 0 else xmin
                    xmin = width if xmin > width else xmin

                    ymin = int(bbox_xywh[1] - 0.1 * bbox_xywh[3])
                    ymin = 0 if ymin < 0 else ymin
                    ymin = height if ymin > height else ymin

                    xmax = int(bbox_xywh[0] + 1.1 * bbox_xywh[2])
                    xmax = 0 if xmax < 0 else xmax
                    xmax = width if xmax > width else xmax

                    ymax = int(bbox_xywh[1] + 1.1 * bbox_xywh[3])
                    ymax = 0 if ymax < 0 else ymax
                    ymax = height if ymax > height else ymax

                    bboxes_xyxy_person.append([xmin, ymin, xmax, ymax])

                #
                bbox_dict[frame_id] = np.array(bboxes_xyxy_person).tolist()

                #
                for bbox_xyxy_person in bboxes_xyxy_person:
                    cv2.rectangle(
                        img_bgr, (bbox_xyxy_person[0], bbox_xyxy_person[1]),
                        (bbox_xyxy_person[2], bbox_xyxy_person[3]), (0, 255, 0), 2)
            else:
                bbox_dict[frame_id] = list()

            videoWriter.write(img_bgr)

        #
        videoWriter.release()

        # print('stack_track done')
        save_path_bbox = video_path.replace('.avi', '_bbox.json')
        with open(save_path_bbox, 'w') as json_writer:
            json.dump(bbox_dict, json_writer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str,
                        default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--dataset_dir", type=str,
                        default='/data1/zhumh/tmp/work_dir')
    # parser.add_argument("--work_dir", type=str,
    #                     default="/data1/zhumh/yolo_test/work_dir")
    parser.add_argument("--cpu", dest="use_cuda",
                        action="store_false", default=True)
    return parser.parse_args()


def listdir(dir, list_name):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.avi' and file_path.find('_.') == -1:
            list_name.append(file_path)


if __name__ == "__main__":
    args = parse_args()

    video_paths = []
    listdir(args.dataset_dir, video_paths)
    
    trk = Tracker(args)
    for video_path in tqdm(video_paths):
            trk.run(video_path)



