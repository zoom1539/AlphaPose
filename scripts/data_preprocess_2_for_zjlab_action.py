import os
from tqdm import tqdm
import argparse
import json
import numpy as np

import torch
import cv2

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import heatmap_to_coord_simple


class PoseEstimator():
    def __init__(self, args, cfg):
        self.args = args
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location="cuda:0"))
        self.pose_model.to("cuda:0")
        self.pose_model.eval()

        self.resolution = (192, 256)

        self.line_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        self.point_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        self.line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                    (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                    (77, 222, 255), (255, 156, 127),
                    (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    def run(self, video_path):
        #
        video = cv2.VideoCapture()
        video.open(video_path)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # fourcc = cv2.VideoWriter_fourcc( *'XVID')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
        video_save_path = video_path.replace('.avi', '_joint_.avi')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

        #
        bbox_path = video_path.replace('.avi', '_bbox.json')
        with open(bbox_path, 'r') as f:
            bboxes_xyxy = json.loads(f.read()) # [bbox_xyxy, label]

        #
        joints_dict = dict()
        data = []
        frame_id = -1
        while video.grab():
            frame_id += 1
            _, img_bgr_origin = video.retrieve()

            if bboxes_xyxy[str(frame_id)]:
                bbox_xyxy = bboxes_xyxy[str(frame_id)][0]
                xmin, ymin, xmax, ymax = [int(coord) for coord in bbox_xyxy]
                # print(xmin, ymin, xmax, ymax)
                img_bgr = img_bgr_origin[ymin:ymax, xmin:xmax]

                img_bgr = cv2.resize(img_bgr, self.resolution)
                
                img_tensor = self.img_preprocess(img_bgr)
                img_tensor_cuda = img_tensor.cuda()
                with torch.no_grad():
                    heatmap = self.pose_model(img_tensor_cuda)
                heatmap_cpu = heatmap.cpu()
                coords, scores = self.postprocess(heatmap_cpu, (xmax - xmin, ymax - ymin))
                coords = [coord + np.array([xmin, ymin]) for coord in coords]
                coords = np.array(coords)
                scores = [score[0] for score in scores]

                self.draw_joint(img_bgr_origin, coords)
                
                video_writer.write(img_bgr_origin)
                
                # joints_dict[frame_id]=dict(coords = coords.tolist(), scores = scores.tolist())
                coords_norm = [ coord / np.array([width, height]) for coord in coords]
                coords_norm = np.array(coords_norm)

                coords_norm = coords_norm.reshape(1, -1)
                scores = np.array(scores).reshape(1, -1)
                datum = dict(frame_index = frame_id + 1, skeleton = [dict(pose = coords_norm.tolist()[0], score = scores.tolist()[0])])
                data.append(datum)
            else:
                video_writer.write(img_bgr_origin)
                # joints_dict[frame_id]= {}
                datum = dict(frame_index = frame_id + 1, skeleton = [])
                data.append(datum)


        video_writer.release()
        # cv2.destroyAllWindows()

        label_save_path = video_path.replace('.avi', '_joint.json')
        with open(label_save_path, 'w') as json_writer:
            joints_dict = dict(data = data, label = "null", label_index = -1)
            json.dump(joints_dict, json_writer)

        # 
        # head_save_path = video_path.replace('.avi', '_head.txt')
        # self.save_to_txt(head_save_path, joints_dict)

    def save_to_txt(self, head_save_path, joints_dict):
        file = open(head_save_path, 'w')
        for key, value in joints_dict.items():
            file.write(str(key))
            file.write(' ')
            if value:
                file.write(str(value['coords'][0][0]))
                file.write(' ')
                file.write(str(value['coords'][0][1]))
            file.write('\n')
        file.close()

    def img_preprocess(self, img_bgr):
        img_rgb = img_bgr[:,:,::-1].copy()
        img_rgb = img_rgb / 255.0
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
        return img_tensor.type(torch.FloatTensor)

    def postprocess(self, heatmap_cpu, resolution):
        heatmap = heatmap_cpu.data.numpy()[0]
        coords, scores = heatmap_to_coord_simple(heatmap, (0,0, *[i - 1 for i in resolution]))

        coord = (coords[5, :] + coords[6, :] ) / 2.0
        coord = coord[np.newaxis,:]
        coords = np.concatenate((coords, coord), axis = 0)

        score = (scores[5, :] + scores[6, :] ) / 2.0
        score = score[np.newaxis,:]
        scores = np.concatenate((scores, score), axis = 0)

        return coords, scores

    # def draw_joint(self, img_bgr, coords, label):
    def draw_joint(self, img_bgr, coords):
        for i, coord in enumerate(coords):
            cv2.circle(img_bgr, (int(coord[0]), int(coord[1])), 2, self.point_color[i], -1)

        for line, color in zip(self.line_pair, self.line_color):
            point_start = coords[line[0]]
            point_end = coords[line[1]]
            cv2.line(img_bgr, (int(point_start[0]), int(point_start[1])), 
                (int(point_end[0]), int(point_end[1])),color, 2, 4)
        
        # if label == 0:
        #     cv2.rectangle(img_bgr, (0, 0), (int(img_bgr.shape[1] - 1), int(img_bgr.shape[0] - 1)), (0,255,0), 2)
        # else:
        #     cv2.rectangle(img_bgr, (0, 0), (int(img_bgr.shape[1] - 1), int(img_bgr.shape[0] - 1)), (0,0,255), 2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/data1/zhumh/zjlab_action_segments")
    parser.add_argument('--cfg', type=str, default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml', help='experiment configure file name')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/fast_res50_256x192.pth', help='checkpoint file name')
    return parser.parse_args()

def listdir(dir, list_name):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1]=='.avi' and file_path.find('_.') == -1:
            list_name.append(file_path)

if __name__ == "__main__":
    args = parse_args()
    cfg = update_config(args.cfg)

    video_paths = []
    listdir(args.work_dir, video_paths)

    pose_estimator = PoseEstimator(args, cfg)

    for video_path in tqdm(video_paths):
        pose_estimator.run(video_path)
    
