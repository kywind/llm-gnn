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

class Dataparser:
    def __init__(self, args, img_dir, detection=False, query=False):
        self.args = args
        self.img_dir = img_dir
        self.device = args.device

        self.image = Image.open(self.img_dir)

        detection = detection
        if detection:
            self.det_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.det_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.det_model.to(self.device)

        query_blip2 = query
        if query_blip2:
            self.query_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.query_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
            )

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
        inputs = self.det_processor(text=texts, images=self.image, return_tensors="pt").to(self.device)
        outputs = self.det_model(**inputs)
    
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
        inputs = self.query_processor(text=texts, images=self.image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = self.query_model.generate(**inputs)
        generated_text = self.query_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


def owlvit_test():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://seas.harvard.edu/sites/default/files/2021-10/an_vision-gDPaDDy6_WE-unsplash.jpg"
    # url = "https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/06/apple-varieties-types-1296x728-header.jpg?w=1155&h=1528"
    # image = Image.open(requests.get(url, stream=True).raw)

    image = Image.open("vis/inputs/apples2_640.png")

    # texts = [["a photo of a cat", "a photo of a dog"]]
    texts = [["apple"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

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

if __name__ == "__main__":
    owlvit_test()
    # blipv2_test()
    # instruct_blip_test()

