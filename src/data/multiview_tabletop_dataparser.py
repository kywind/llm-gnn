import requests
from PIL import Image, ImageDraw
import torch
import numpy as np
import cv2

from transformers import OwlViTProcessor, OwlViTForObjectDetection

import requests
from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration

from omegaconf import OmegaConf

# from lavis.models import load_model, load_preprocess
# from lavis.common.registry import registry

from segment_anything import SamPredictor, sam_model_registry

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from data.utils import label_colormap
from config import gen_args

import open3d as o3d


class MultiviewTabletopDataparser:
    """
    fusing depth from multiview images
    """

    def __init__(self, args, data_dir, dataset_name):
        self.args = args
        self.device = args.device

        if dataset_name == "d3fields":
            n_cameras = 4
            rgb_dir = [data_dir + f"camera_{i}/color" for i in range(n_cameras)]
            depth_dir = [data_dir + f"camera_{i}/depth" for i in range(n_cameras)]
            intr_dir = [data_dir + f"camera_{i}/camera_params.npy" for i in range(n_cameras)]
            extr_dir = [data_dir + f"camera_{i}/camera_extrinsics.npy" for i in range(n_cameras)]

            img_index = 0
            self.rgb_imgs = [Image.open(rgb_dir[i] + f"/{img_index}.png") for i in range(n_cameras)]
            self.depth_imgs = [Image.open(depth_dir[i] + f"/{img_index}.png") for i in range(n_cameras)]
            self.n_cameras = n_cameras

            self.cam_params = [np.load(intr_dir[i]) for i in range(n_cameras)]  # (4,) * n_cameras
            self.cam_extrinsics = [np.load(extr_dir[i]) for i in range(n_cameras)]  # (4, 4) * n_cameras

            # self.bbox = [0, 0, 640, 480]  # in meters, the bounding box of the workspace
            self.tabletop = [(58, 84), (310, 354), (634, 210), (507, 354)]  # x, y
        
        elif dataset_name == "ycb-flex":
            n_cameras = 4
            self.rgb_imgs = [data_dir + f"view_{i}/color.png" for i in range(1, n_cameras + 1)]
            # self.depth_imgs = [data_dir + f"view_{i}/fgpcd.png" for i in range(1, n_cameras + 1)]
            self.depth_img = None
            intr_dir = [data_dir + f"view_{i}/camera_intrinsic_params.npy" for i in range(1, n_cameras + 1)]
            extr_dir = [data_dir + f"view_{i}/camera_extrinsic_matrix.npy" for i in range(1, n_cameras + 1)]
            
            self.cam_params = [np.load(intr_dir[i]) for i in range(n_cameras)]  # (4,) * n_cameras
            self.cam_extrinsics = [np.load(extr_dir[i]) for i in range(n_cameras)]  # (4, 4) * n_cameras
            import ipdb; ipdb.set_trace()
        
        else:
            raise NotImplementedError
    

    def parse_pcd(self, depth, cam_param, cam_extrinsic):
        # to camera frame
        import ipdb; ipdb.set_trace() # check depth shape
        fgpcd = np.zeros((depth.shape[0] * depth.shape[1], 3))
        fx, fy, cx, cy = cam_param
        pos_x, pos_y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))  # w, h
        fgpcd[:, 0] = (pos_x - cx) * depth / fx
        fgpcd[:, 1] = (pos_y - cy) * depth / fy
        fgpcd[:, 2] = depth
        # to world frame
        fgpcd = np.hstack((fgpcd, np.ones((fgpcd.shape[0], 1))))
        fgpcd = np.matmul(fgpcd, np.linalg.inv(cam_extrinsic).T)[:, :3]
        return fgpcd

    def parse_table(self):
        table = self.tabletop
        table_depths = []
        depth = np.array(self.depth_imgs[0])
        for i in range(4):
            w = table[i][0]
            h = table[i][1]
            table_depths.append(depth[h, w])   # first image
            # import ipdb; ipdb.set_trace()

        plane_depth = depth[212, 281]
        
        corner_1_depth = table_depths[0]
        # line_1_dir = [table[1][0] - table[0][0], table[1][1] - table[0][1]]  # w, h
        corner_2_depth = table_depths[2]
        # line_2_dir = [table[3][0] - table[2][0], table[3][1] - table[2][1]]  # w, h

        corner_1 = np.zeros(3)
        corner_1[0] = (table[0][0] - self.cam_params[0][2]) * corner_1_depth / self.cam_params[0][0]
        corner_1[1] = (table[0][1] - self.cam_params[0][3]) * corner_1_depth / self.cam_params[0][1]
        corner_1[2] = corner_1_depth

        corner_2 = np.zeros(3)
        corner_2[0] = (table[2][0] - self.cam_params[0][2]) * corner_2_depth / self.cam_params[0][0]
        corner_2[1] = (table[2][1] - self.cam_params[0][3]) * corner_2_depth / self.cam_params[0][1]
        corner_2[2] = corner_2_depth

        plane_point = np.zeros(3)
        plane_point[0] = (281 - self.cam_params[0][2]) * plane_depth / self.cam_params[0][0]
        plane_point[1] = (212 - self.cam_params[0][3]) * plane_depth / self.cam_params[0][1]
        plane_point[2] = plane_depth

        corner_1 = np.hstack((corner_1, np.ones(1)))
        corner_2 = np.hstack((corner_2, np.ones(1)))
        corner_1_world = np.matmul(corner_1, np.linalg.inv(self.cam_extrinsics[0]).T)
        corner_2_world = np.matmul(corner_2, np.linalg.inv(self.cam_extrinsics[0]).T)
        corner_1_world = corner_1_world[:3] * 0.001
        corner_2_world = corner_2_world[:3] * 0.001

        plane_point = np.hstack((plane_point, np.ones(1)))
        plane_point_world = np.matmul(plane_point, np.linalg.inv(self.cam_extrinsics[0]).T)
        plane_point_world = plane_point_world[:3] * 0.001

        bbox = np.zeros(6) # x_min, y_min, x_max, y_max, z_min, z_max
        bbox[0] = min(corner_1_world[0], corner_2_world[0])
        bbox[1] = min(corner_1_world[1], corner_2_world[1])
        bbox[2] = max(corner_1_world[0], corner_2_world[0])
        bbox[3] = max(corner_1_world[1], corner_2_world[1])
        bbox[5] = plane_point_world[2] - 0.01  # z_max is table plane
        bbox[4] = bbox[5] - 0.5  # z_min is above table plane

        # transform depth to world coordinate
        depth = np.array(self.depth_imgs[0])
        depth_mask = np.zeros(depth.shape)
        depth_world = np.zeros((depth.shape[0], depth.shape[1], 4))
        fx, fy, cx, cy = self.cam_params[0]
        pos_x, pos_y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))  # w, h
        depth_world[:, :, 0] = (pos_x - cx) * depth / fx
        depth_world[:, :, 1] = (pos_y - cy) * depth / fy
        depth_world[:, :, 2] = depth
        depth_world[:, :, 3] = 1
        depth_world = np.matmul(depth_world, np.linalg.inv(self.cam_extrinsics[0]).T)[:, :, :3]
        depth_world *= 0.001
        depth_mask = np.logical_and(depth_world[:, :, 0] > bbox[0], depth_world[:, :, 0] < bbox[2])
        depth_mask = np.logical_and(depth_mask, depth_world[:, :, 1] > bbox[1])
        depth_mask = np.logical_and(depth_mask, depth_world[:, :, 1] < bbox[3])
        depth_mask = np.logical_and(depth_mask, depth_world[:, :, 2] > bbox[4])
        depth_mask = np.logical_and(depth_mask, depth_world[:, :, 2] < bbox[5])
        depth_mask = depth_mask.astype(np.uint8)
        depth_mask = depth_mask * 255
        depth_mask = Image.fromarray(depth_mask)
        depth_mask.save('vis/depth_mask.png')
        import ipdb; ipdb.set_trace()





    def tabletop_3in1_query(self):
        # parse table depth
        self.parse_table()

        # depth to pcd
        for camera_index in range(self.n_cameras):
            depth = np.array(self.depths[camera_index])
            depth = depth * 0.001
            cam_param = self.cam_params[camera_index]
            cam_extrinsic = self.cam_extrinsics[camera_index]
            fgpcd = self.parse_pcd(depth, cam_param, cam_extrinsic)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(fgpcd)
            # filter with bbox
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=self.bbox[:3], max_bound=self.bbox[3:]))

        import ipdb; ipdb.set_trace()
        inputs = self.query_processor(text=texts, images=self.rgb_imgs[camera_index], return_tensors="pt").to("cuda", torch.float16)
        generated_ids = self.query_model.generate(**inputs)
        generated_text = self.query_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def detect_gdino(self, captions, box_thresholds, camera_index):  # captions: list
        det_model = build_model(SLConfig.fromfile(
            '../third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'))
        checkpoint = torch.load('weights/groundingdino_swinb_cogcoor.pth', map_location="cpu")
        det_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        det_model.eval()
        det_model = det_model.to(self.device)

        device = self.device
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."
            captions[i] = caption
        num_captions = len(captions)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(self.rgb_imgs[camera_index], None)  # 3, h, w

        image_tensor = image_tensor[None].repeat(num_captions, 1, 1, 1).to(device)

        with torch.no_grad():
            outputs = det_model(image_tensor, captions=captions)
        logits = outputs["pred_logits"].sigmoid()  # (num_captions, nq, 256)
        boxes = outputs["pred_boxes"]  # (num_captions, nq, 4)

        # filter output
        if isinstance(box_thresholds, list):
            filt_mask = logits.max(dim=2)[0] > torch.tensor(box_thresholds).to(device=device, dtype=logits.dtype)[:, None]
        else:
            filt_mask = logits.max(dim=2)[0] > box_thresholds
        labels = torch.ones((*logits.shape[:2], 1)) * torch.arange(logits.shape[0])[:, None, None]  # (num_captions, nq, 1)
        labels = labels.to(device=device, dtype=logits.dtype)  # (num_captions, nq, 1)
        logits = logits[filt_mask] # num_filt, 256
        boxes = boxes[filt_mask] # num_filt, 4
        labels = labels[filt_mask].reshape(-1).to(torch.int64) # num_filt,
        scores = logits.max(dim=1)[0] # num_filt,

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {captions[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        return boxes, scores, labels
    
    def segment_gdino(self, obj_list, camera_index):
        device = self.device
        text_prompts = [f"{obj}" for obj in obj_list]
        print('segment prompt:', text_prompts)  # things like: ['apple', 'banana', 'orange']
        boxes, scores, labels = self.detect_gdino(text_prompts, box_thresholds=0.5, camera_index=camera_index)

        image = np.array(self.rgb_imgs[camera_index])
        H, W = image.shape[:2]
        boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=device, dtype=boxes.dtype)
        boxes[:, :2] -= boxes[:, 2:] / 2  # xywh to xyxy
        boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy

        if boxes.size(0) == 0:
            print('no detection')
            return None, None
            # return torch.zeros((H, W), dtype=torch.bool, device=device), ['background']

        # load sam model
        sam = sam_model_registry["default"](checkpoint='weights/sam_vit_h_4b8939.pth')
        sam_model = SamPredictor(sam)
        sam_model.model = sam_model.model.to(device)
        sam_model.set_image(image)

        masks, _, _ = sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]), # (n_detection, 4)
            multimask_output = False,
        )

        masks = masks[:, 0, :, :] # (n_detection, H, W)
        # text_labels = ['background']
        text_labels = []
        for category in range(len(text_prompts)):
            text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()
        
        # remove masks where IoU are large
        num_masks = masks.shape[0]
        to_remove = []
        for i in range(num_masks):
            for j in range(i+1, num_masks):
                IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
                if IoU > 0.9:
                    if scores[i].item() > scores[j].item():
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        to_remove = np.unique(to_remove)
        to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
        to_keep = torch.from_numpy(to_keep).to(device=device, dtype=torch.int64)
        masks = masks[to_keep]
        text_labels = [text_labels[i] for i in to_keep]
        # text_labels.insert(0, 'background')
        
        aggr_mask = torch.zeros(masks[0].shape).to(device=device, dtype=torch.uint8)
        for obj_i in range(masks.shape[0]):
            aggr_mask[masks[obj_i]] = obj_i + 1

        # visualization
        visualize = True
        if visualize:
            colors = label_colormap()
            aggr_mask_vis = np.zeros((*masks[0].shape, 3), dtype=np.uint8)
            masks_vis = masks.cpu().numpy()
            for obj_i in range(masks.shape[0]):
                aggr_mask_vis[masks_vis[obj_i]] = np.array(colors[labels[obj_i].item()])

            aggr_mask_vis = Image.fromarray(aggr_mask_vis)
            aggr_mask_vis = Image.blend(self.rgb_imgs[camera_index], aggr_mask_vis, 0.7)
            aggr_mask_vis.save('vis/grounded_dino_sam_test.png')

        # masks: (n_detection, H, W)
        # aggr_mask: (H, W)
        return (masks, aggr_mask, text_labels), (boxes, scores, labels)

