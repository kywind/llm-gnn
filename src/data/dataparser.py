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

class Dataparser:
    def __init__(self, args, img_dir, detection=False, query=False):
        self.args = args
        self.img_dir = img_dir
        self.device = args.device

        self.image = Image.open(self.img_dir)

        detection_gdino = detection
        if detection_gdino:
            pass
        """
        detection_owlvit = False
        if detection:
            self.det_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.det_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.det_model.to(self.device)
        """
        query_blip2 = query
        if query_blip2:
            pass
        """
        query_instruct_blip = False  # 32G RAM OOM
        if query_instruct_blip:
            vlm = load_model(
                name='blip2_t5_instruct',
                model_type='flant5xxl',
                checkpoint='pg-vlm/pgvlm_weights.bin',  # replace with location of downloaded weights
                is_eval=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        """

    def detect(self, texts):
        raise NotImplementedError
        # inputs = self.det_processor(text=texts, images=self.image, return_tensors="pt").to(self.device)
        # outputs = self.det_model(**inputs)
    
    def detect_gdino(self, captions, box_thresholds):  # captions: list
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
        image_tensor, _ = transform(self.image, None)  # 3, h, w

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
        return boxes, scores, labels, logits

    def segment_gdino(self, texts):
        device = self.device
        obj_raw, material = texts.split('|')
        obj_list = obj_raw.split(',')
        text_prompts = [f"{obj}" for obj in obj_list]
        print('segment prompt:', text_prompts)  # things like: ['apple', 'banana', 'orange']
        boxes, scores, labels, logits = self.detect_gdino(text_prompts, box_thresholds=0.5)

        image = np.array(self.image)
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
                    if logits[i].max().item() > logits[j].max().item():
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
            aggr_mask_vis = Image.blend(self.image, aggr_mask_vis, 0.7)
            aggr_mask_vis.save('vis/grounded_dino_sam_test.png')

        # masks: (n_detection, H, W)
        # aggr_mask: (H, W)
        return (masks, aggr_mask, text_labels), (boxes, scores, labels)

    def segment(self, texts):
        obj_raw, material = texts.split('|')
        obj_list = obj_raw.split(',')
        # for i in range(len(obj_list)):
        #     # remove 'and' in the beginning
        #     obj = obj_list[i]
        #     obj = obj.strip(' ')
        #     if obj.startswith('and'):
        #         obj = obj[3:]
        #     obj = obj.strip(' ')
        #     obj_list[i] = obj
        prompt = [[f"{obj}" for obj in obj_list]]
        print('segment prompt:', prompt)  # things like: [['apple', 'banana', 'orange']]

        inputs = self.det_processor(text=prompt, images=self.image, return_tensors="pt").to(self.device)
        outputs = self.det_model(**inputs)

        target_sizes = torch.tensor([[self.image.height, self.image.width]], device=self.device)
        results = self.det_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)

        img_idx = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = prompt[img_idx]
        boxes, scores, labels = results[img_idx]["boxes"], results[img_idx]["scores"], results[img_idx]["labels"]

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        
        # crop pil image using bounding box
        sam = sam_model_registry["default"](checkpoint="weights/sam_vit_h_4b8939.pth")
        predictor = SamPredictor(sam)

        predictions = []
        for j in range(boxes.shape[0]):
            crop_img_pil = self.image.crop((int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]), int(boxes[j][3])))
            crop_img_pil.save(f'vis/crop_img_{j}.png')
            crop_img = np.array(crop_img_pil)

            # center_x = (boxes[j][0] + boxes[j][2]) / 2
            # center_y = (boxes[j][1] + boxes[j][3]) / 2
            center_x = crop_img.shape[1] / 2
            center_y = crop_img.shape[0] / 2
            center = torch.tensor([center_x, center_y])

            predictor.set_image(crop_img)

            # transformed_points = predictor.transform.apply_coords_torch(center, (self.image.height, self.image.width))

            masks, _, _ = predictor.predict(
                point_coords=center.cpu().numpy()[None],
                point_labels=np.ones(1), # labels[j].cpu().numpy()[None],
                multimask_output=True
            )

            def cnt_area(cnt):
                area = cv2.contourArea(cnt)
                return area

            # visualize mask
            # mask = masks[0]
            best_k = 0
            best_score = -10000
            for k in range(masks.shape[0]):
                mask = masks[k]
                mask = np.array(mask).astype(np.uint8) * 255
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                n_contours = len(contours)
                contours = list(contours)
                contours.sort(key=cnt_area, reverse=True)
                max_cnt_area = cnt_area(contours[0])

                score = max_cnt_area / crop_img.shape[1] / crop_img.shape[0] - np.clip(n_contours, 1, 2) * 0.25
                if score > best_score:
                    best_score = score
                    best_k = k

            mask = masks[best_k]

            # visualize mask
            mask_vis = mask[..., None].repeat(3, axis=2)
            mask_vis = mask_vis * np.array([255, 0, 0])  # red
            mask_vis = Image.fromarray(mask_vis.astype(np.uint8))
            mask_vis = Image.blend(crop_img_pil, mask_vis, 0.7)
            draw = ImageDraw.Draw(mask_vis)
            draw.point((center_x, center_y), fill=(0, 255, 0))  # green
            mask_vis.save(f'vis/mask_{j}.png')

            predictions.append((crop_img_pil, mask))
        
        # TODO
        # feed prediction back to main function
        # consider different materials per prediction
        # sample particles (need new particle dataparser)
        # build relations within each prediction (need new particle dataparser)
        # build relations between predictions (need new particle dataparser)
        return predictions, boxes, scores, labels

    def query(self, texts, bbox=None, mask=None):
        query_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        query_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
            )
        inputs = query_processor(text=texts, images=self.image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = query_model.generate(**inputs)
        generated_text = query_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


def run_owlvit(image, texts, box_thresholds):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=box_thresholds)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    return boxes, scores, labels

def blipv2_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )

    url = "https://iliad.stanford.edu/pg-vlm/example_images/ceramic_bowl.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # prompt = "Question: how many cats are there? Answer:"
    prompts = [
        'Question: Classify this object as transparent, translucent, or opaque? Respond unknown if you are not sure. Short answer:', # opaque
        'Question: Is this object fragile? Respond unknown if you are not sure. Short answer:', # yes
        'Question: What are the most frequently used materials for this type of object? Respond unknown if you are not sure. Short answer:', # porcelain
        'Question: What is the most common function of this type of object? Respond unknown if you are not sure. Short answer:', # it's a bowl
        'Question: Where is this bowl most likely located in a house? Respond unknown if you are not sure. Short answer:', # in the kitchen
        'Question: Is this object dirty? Respond unknown if you are not sure. Short answer:', # no
        'Question: Is this object compact? Respond unknown if you are not sure. Short answer:', # yes
    ]
    for i, prompt in enumerate(prompts):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if generated_text == "" or generated_text is None:
            generated_text = "No answer found."
        print(i, generated_text)

def blipv2_test2():
    from config import gen_args
    # img_dir = '/home/zhangkaifeng/projects/dyn-res-pile-manip/data/gnn_dyn_data/0/0_color.png'
    img_dir = '/home/zhangkaifeng/projects/llm-gnn/src/vis/g1.png'
    args = gen_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    parser = Dataparser(args, img_dir)
    prompts = [
        # 'Question: Classify this image as rigid objects or deformable objects? Respond unknown if you are not sure. Short answer:',
        # 'Question: Classify this image as granular objects or rigid objects? Respond unknown if you are not sure. Short answer:',
        'Question: Classify this image as rope or granular objects? Respond unknown if you are not sure. Short answer:',
        'Question: Classify this image as deformable objects or granular objects? Respond unknown if you are not sure. Short answer:',
        'Question: Classify this image as rigid objects or granular objects? Respond unknown if you are not sure. Short answer:',
        # 'Question: Classify this image as granular objects, rigid objects, rope, deformable objects? Respond unknown if you are not sure. Short answer:',
        # 'Question: Classify this image as rigid objects, granular objects, deformable objects, rope? Respond unknown if you are not sure. Short answer:',
        # 'Question: Classify this image as deformable objects, rope, granular objects, rigid objects? Respond unknown if you are not sure. Short answer:',
        # 'Question: Classify this image as rope, deformable objects, rigid objects, granular objects? Respond unknown if you are not sure. Short answer:',
        'This is an image in a robotic simulation environment. Describe the objects in this image. Answer:',
    ]
    for i, prompt in enumerate(prompts):
        print(i, parser.query(prompt))
"""
def instruct_blip_test():
    url = "https://iliad.stanford.edu/pg-vlm/example_images/ceramic_bowl.jpg"
    example_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    import ipdb; ipdb.set_trace()
    vlm = load_model(
        name='blip2_t5_instruct',
        model_type='flant5xxl',
        checkpoint='pg-vlm/pgvlm_weights.bin',  # replace with location of downloaded weights
        is_eval=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    import ipdb; ipdb.set_trace()

    vlm.qformer_text_input = False  # Optionally disable qformer text

    model_cls = registry.get_model_class('blip2_t5_instruct')
    model_type = 'flant5xxl'
    preprocess_cfg = OmegaConf.load(model_cls.default_config_path(model_type)).preprocess
    vis_processors, _ = load_preprocess(preprocess_cfg)
    processor = vis_processors["eval"]


    question_samples = {
        'prompt': 'Question: Classify this object as transparent, translucent, or opaque? Respond unknown if you are not sure. Short answer:',
        'image': torch.stack([processor(example_image)], dim=0).to(vlm.device)
    }

    print(vlm.generate(question_samples, length_penalty=0, repetition_penalty=1, num_captions=3))
    # (['opaque', 'translucent', 'transparent'], tensor([-0.0448, -4.1387, -4.2793], device='cuda:0'))
"""

def run_grounding_dino(image, captions, box_thresholds):
    device = "cuda:0"
    config_file = '../../third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'
    grounded_checkpoint = '../weights/groundingdino_swinb_cogcoor.pth'

    args = SLConfig.fromfile(config_file)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(grounded_checkpoint, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()

    for i, caption in enumerate(captions):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions[i] = caption
    num_captions = len(captions)
    print(captions)

    image_pil = Image.fromarray(image)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)  # 3, h, w

    model = model.to(device)
    image_tensor = image_tensor[None].repeat(num_captions, 1, 1, 1).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, captions=captions)
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

    return logits, boxes, labels

def grounded_dino_sam_test(img_path, text_prompts, box_thresholds):
    from segment_anything import build_sam, SamPredictor

    device = "cuda:0"
    sam_checkpoint = '../weights/sam_vit_h_4b8939.pth'

    # load sam model
    sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
    sam_model = SamPredictor(sam)
    sam_model.model = sam_model.model.to(device)

    # text_prompts = ['a bag of flour','a rolling pin','a dough ball']
    # box_thresholds = [0.5, 0.5, 0.4]

    # img_path = '../vis/inputs/dough_640.png'
    image = cv2.imread(img_path)[..., ::-1]
    sam_model.set_image(image)

    # run grounding dino model
    logits, boxes, labels = run_grounding_dino(image, text_prompts, box_thresholds)
    # logits: (n_detection, 256)
    # boxes: (n_detection, 4)
    # labels: (n_detection, 1)

    H, W = image.shape[:2]
    boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=device, dtype=boxes.dtype)
    boxes[:, :2] -= boxes[:, 2:] / 2  # xywh to xyxy
    boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy

    if boxes.size(0) == 0:
        print('no detection')
        return torch.zeros((H, W), dtype=torch.bool, device=device), ['background']

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]), # (n_detection, 4)
        multimask_output = False,
    )

    masks = masks[:, 0, :, :] # (n_detection, H, W)
    text_labels = ['background']
    for category in range(len(text_prompts)):
        text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()
    
    # remove masks where IoU are large
    num_masks = masks.shape[0]
    to_remove = []
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
            if IoU > 0.9:
                if logits[i].max().item() > logits[j].max().item():
                    to_remove.append(j)
                else:
                    to_remove.append(i)
    to_remove = np.unique(to_remove)
    to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
    to_keep = torch.from_numpy(to_keep).to(device=device, dtype=torch.int64)
    masks = masks[to_keep]
    text_labels = [text_labels[i + 1] for i in to_keep]
    text_labels.insert(0, 'background')
    
    aggr_mask = torch.zeros(masks[0].shape).to(device=device, dtype=torch.uint8)
    for obj_i in range(masks.shape[0]):
        aggr_mask[masks[obj_i]] = obj_i + 1

    # visualization
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 255],
    ]
    aggr_mask_vis = np.zeros((*masks[0].shape, 3), dtype=np.uint8)
    masks_vis = masks.cpu().numpy()
    for obj_i in range(masks.shape[0]):
        aggr_mask_vis[masks_vis[obj_i]] = np.array(colors[labels[obj_i].item()])

    aggr_mask_vis = Image.fromarray(aggr_mask_vis)
    image_pil = Image.fromarray(image)
    aggr_mask_vis = Image.blend(image_pil, aggr_mask_vis, 0.7)
    aggr_mask_vis.save('../vis/grounded_dino_sam_test.png')

    # return aggr_mask, text_labels

def owlvit_sam_test(img_path, text_prompts, box_thresholds):
    device = "cuda:0"
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://seas.harvard.edu/sites/default/files/2021-10/an_vision-gDPaDDy6_WE-unsplash.jpg"
    # url = "https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/06/apple-varieties-types-1296x728-header.jpg?w=1155&h=1528"
    # image = Image.open(requests.get(url, stream=True).raw)

    # img_path = '../vis/inputs/dough_640.png'
    image_pil = Image.open(img_path)
    image = np.array(image_pil)

    # texts = ["a photo of a cat", "a photo of a dog"]
    # text_prompts = ["apple", "banana", "orange"]
    # text_prompts = ['a bag of flour','a rolling pin','a dough ball']
    # box_thresholds = 0.2

    with torch.no_grad():
        boxes, scores, labels = run_owlvit(image_pil, [text_prompts], box_thresholds)
        boxes = boxes.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
    # boxes: (n_detection, 4)
    # scores: (n_detection,)
    # labels: (n_detection,)

    # load sam model
    sam_checkpoint = '../weights/sam_vit_h_4b8939.pth'
    sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
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
    text_labels = ['background']
    for category in range(len(text_prompts)):
        text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()

    aggr_mask = torch.zeros(masks[0].shape).to(device=device, dtype=torch.uint8)
    for obj_i in range(masks.shape[0]):
        aggr_mask[masks[obj_i]] = obj_i + 1

    # visualization
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 255],
    ]
    aggr_mask_vis = np.zeros((*masks[0].shape, 3), dtype=np.uint8)
    masks_vis = masks.cpu().numpy()
    for obj_i in range(masks.shape[0]):
        aggr_mask_vis[masks_vis[obj_i]] = np.array(colors[labels[obj_i].item()])

    aggr_mask_vis = Image.fromarray(aggr_mask_vis)
    aggr_mask_vis = Image.blend(image_pil, aggr_mask_vis, 0.7)
    aggr_mask_vis.save('../vis/owlvit_sam_test.png')


if __name__ == "__main__":
    # blipv2_test()
    # instruct_blip_test()

    # img_path = '../vis/inputs/dough_640.png'
    # text_prompts = ['a bag of flour','a rolling pin','a dough ball']

    img_path = '../../data/2023-09-13-15-19-50-765863/camera_0/color/0.png'
    text_prompts = ['a mouse', 'a keyboard', 'a pen', 'a box', 'a cup', 'a cup mat', 'tabletop']
    box_thresholds = 0.5
    grounded_dino_sam_test(img_path, text_prompts, box_thresholds)
    # owlvit_sam_test(img_path, text_prompts, box_thresholds)

