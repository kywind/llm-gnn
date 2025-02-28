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


class MultiviewDataparser:
    """
    fusing depth from multiview images
    """

    def __init__(self, args, data_dir, dataset_name, img_index=0):
        self.args = args
        self.device = args.device

        if dataset_name == "d3fields":
            n_cameras = 4
            rgb_dir = [data_dir + f"camera_{i}/color" for i in range(n_cameras)]
            depth_dir = [data_dir + f"camera_{i}/depth" for i in range(n_cameras)]
            intr_dir = [data_dir + f"camera_{i}/camera_params.npy" for i in range(n_cameras)]
            extr_dir = [data_dir + f"camera_{i}/camera_extrinsics.npy" for i in range(n_cameras)]

            self.rgb_imgs = [Image.open(rgb_dir[i] + f"/{img_index}.png").convert('RGB') for i in range(n_cameras)]
            self.depth_imgs = [Image.open(depth_dir[i] + f"/{img_index}.png") for i in range(n_cameras)]
            self.n_cameras = n_cameras

            self.cam_params = [np.load(intr_dir[i]) for i in range(n_cameras)]  # (4,) * n_cameras
            self.cam_extrinsics = [np.load(extr_dir[i]) for i in range(n_cameras)]  # (4, 4) * n_cameras
        
        elif dataset_name == "ycb-flex":
            n_cameras = 4
            self.rgb_imgs = [Image.open(data_dir + f"view_{i}/color.png").convert('RGB') for i in range(1, n_cameras + 1)]
            # self.depth_imgs = [data_dir + f"view_{i}/fgpcd.pcd" for i in range(1, n_cameras + 1)]
            self.depth_imgs = None
            intr_dir = [data_dir + f"view_{i}/camera_intrinsic_params.npy" for i in range(1, n_cameras + 1)]
            extr_dir = [data_dir + f"view_{i}/camera_extrinsic_matrix.npy" for i in range(1, n_cameras + 1)]
            
            self.cam_params = [np.load(intr_dir[i]) for i in range(n_cameras)]  # (4,) * n_cameras
            self.cam_extrinsics = [np.load(extr_dir[i]) for i in range(n_cameras)]  # (4, 4) * n_cameras
            # import ipdb; ipdb.set_trace()
        
        elif dataset_name == "flex":
            n_cameras = 4  # TODO: support episode indexing
            self.rgb_imgs = []
            self.depth_imgs = []
            for i in range(1, n_cameras + 1):
                rgb_img = cv2.imread(data_dir + f"camera_{i}/episode_0/{img_index}_color.png")
                rgb_img = rgb_img[:, :, :3]
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                rgb_img = Image.fromarray(rgb_img)
                self.rgb_imgs.append(rgb_img)

                depth_img = cv2.imread(data_dir + f"camera_{i}/episode_0/{img_index}_depth.png", cv2.IMREAD_UNCHANGED)
                depth_img = depth_img.astype(np.float32) / 1000.0
                depth_img = Image.fromarray(depth_img)
                self.depth_imgs.append(depth_img)
            import ipdb; ipdb.set_trace()
            # self.rgb_imgs = [Image.open(data_dir + f"camera_{i}/episode_0/{img_index}_color.png").convert('RGB') for i in range(1, n_cameras + 1)]
            # self.depth_imgs = [Image.open(data_dir + f"camera_{i}/episode_0/{img_index}_depth.png") for i in range(1, n_cameras + 1)]
        else:
            raise NotImplementedError
    
    def prepare_query_model(self):
        self.query_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.query_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )

    def del_query_model(self):
        del self.query_processor
        del self.query_model
        torch.cuda.empty_cache()

    def query(self, texts, camera_index, image=None, bbox=None, mask=None):
        if image is None:
            image = self.rgb_imgs[camera_index]
        inputs = self.query_processor(text=texts, images=image, return_tensors="pt").to("cuda", torch.float16)
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
    
    def segment_gdino(self, obj_list, camera_index, box_threshold):
        device = self.device
        text_prompts = [f"{obj}" for obj in obj_list]
        print('segment prompt:', text_prompts)  # things like: ['apple', 'banana', 'orange']
        boxes, scores, labels = self.detect_gdino(text_prompts, box_thresholds=box_threshold, camera_index=camera_index)

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


if __name__ == "__main__":
    args = gen_args()
    args.device = "cuda:0"
    data = MultiviewDataparser(args)
    camera_indices = np.arange(data.n_cameras)
    for camera_index in camera_indices:
        gen_text = data.query("List all the objects on the table. Answer:", camera_index)
        print("gen_text:", gen_text)
