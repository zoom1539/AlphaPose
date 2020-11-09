from __future__ import division
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
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools

def write(x, img):
    print('x: ',x)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    #color = random.choice(colors)
    color = (255,0,0)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

if __name__ == '__main__':

    scales = "1,2,3"
    images = "imgs/messi.jpg"
    batch_size = 1
    confidence = 0.5
    nms_thesh = 0.4

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names') 

    #Set up the neural network
    print("Loading network.....")
    model = Darknet("cfg/yolov3-spp.cfg")
    model.load_weights("data/yolov3-spp.weights")
    print("Network successfully loaded")

    model.net_info["height"] = "608"
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    #Detection phase
    try:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()

    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()
        print('cuda count: ', torch.cuda.device_count())

    for batch in im_batches:
        #load the image
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            parser = argparse.ArgumentParser(description='AlphaPose Demo')
            args = parser.parse_args()
            args.device = torch.device("cuda:0")
            # args.device = torch.device("cpu")
            prediction = model(Variable(batch), args)

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)
        output = prediction

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()
    print(im_dim_list.shape)
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)


    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    print(output)
    print(output.shape)

    img = cv2.imread(imlist[0])
    list(map(lambda x: write(x, img), output))
    cv2.imwrite('result.jpg', img)
