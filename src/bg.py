import glob
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms

from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed

from data.multiview_dataparser import MultiviewDataparser
from data.multiview_dataset import MultiviewParticleDataset
from llm.llm import LLM
import open3d as o3d

import cv2
import glob


def build_graph(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    ## configs
    visualize = False
    verbose = False
    skip_query = False
    skip_material = True
    skip_segment = False

    camera_indices = [0, 1, 2, 3]
    data_dir = "../data/2023-09-13-15-19-50-765863/"
    vis_dir = "vis/multiview-tabletop-0/"
    dataset_name = "d3fields"

    # data_dir = "../data/mustard_bottle/"
    # vis_dir = "vis/multiview-ycb-0/"
    # dataset_name = "ycb-flex"

    if skip_segment:
        assert os.path.exists(vis_dir), "vis_dir does not exist"
        assert os.path.exists(os.path.join(vis_dir, 'mask_0_0.png')), "mask_0_0.png does not exist"
    os.makedirs(vis_dir, exist_ok=True)

    # initial dataparser
    init_parser = MultiviewDataparser(args, data_dir, dataset_name)
    llm = LLM(args)

    if not skip_query:
        init_parser.prepare_query_model()
        obj_list = []
        for camera_index in camera_indices:
            query_results = init_parser.query(
                texts='What objects are on the table? Answer:',
                camera_index=camera_index
            )
            objs = query_results.rstrip('.').split(',')
            for obj in objs:
                obj = obj.strip(' ')
                obj = obj.lstrip('and')
                obj = obj.strip(' ')
                if obj not in obj_list:
                    obj_list.append(obj)
        init_parser.del_query_model()
        print('After query, obj_list:', obj_list)
    else:
        obj_list = ['a mouse', 'a keyboard', 'a pen', 'a box', 'a cup', 'a cup mat']
        # obj_list = ['a mustard bottle']

    material_dict = {}
    if not skip_material:
        for obj_name in obj_list:
            material_prompt = " ".join([
                "Classify the objects in the image as rigid objects, granular objects, deformable objects, or rope.",
                "Respond unknown if you are not sure."
                "\n\nQuery: coffee bean. Answer: granular.",
                "\n\nQuery: rope. Answer: rope.",
                "\n\nQuery: a wooden block. Answer: rigid.",
                "\n\nQuery: a banana. Answer: rigid.",
                "\n\nQuery: an orange. Answer: rigid.",
                "\n\nQuery: play-doh. Answer: deformable.",
                "\n\nQuery: sand. Answer: granular.",
                "\n\nQuery: a bottle. Answer: rigid.",
                "\n\nQuery: a t-shirt. Answer: deformable.",
                "\n\nQuery: rice. Answer: granular.",
                "\n\nQuery: laptop. Answer: rigid.",
                "\n\nQuery: " + obj_name + ". Answer:",
            ])
            material = llm.query(material_prompt)
            material = material.rstrip('.')
            material_dict[obj_name] = material

    mask_list = []
    text_labels_list = []

    if not skip_segment:
        for camera_index in camera_indices:
            segmentation_results, detection_results = init_parser.segment_gdino(
                obj_list=obj_list,
                camera_index=camera_index,
            )
            masks, _, text_labels = segmentation_results
            # _, _, labels = detection_results

            masks = masks.detach().cpu().numpy()  # (n_detect, H, W) boolean            
            for i in range(masks.shape[0]):
                mask = (masks[i] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(vis_dir, 'mask_{}_{}.png'.format(camera_index, i)), mask)
                with open(os.path.join(vis_dir, 'text_labels_{}_{}.txt'.format(camera_index, i)), 'w') as f:
                    f.write(text_labels[i])
            
            mask_list.append(masks)
            text_labels_list.append(text_labels)

    else:
        for camera_index in camera_indices:
            masks = []
            text_labels = []
            mask_dirs = glob.glob(os.path.join(vis_dir, 'mask_{}*.png'.format(camera_index)))
            for i in range(len(mask_dirs)):
                mask = cv2.imread(os.path.join(vis_dir, 'mask_{}_{}.png'.format(camera_index, i)))
                mask = (mask > 0)[..., 0]  # (H, W) boolean
                masks.append(mask)
                with open(os.path.join(vis_dir, 'text_labels_{}_{}.txt'.format(camera_index, i)), 'r') as f:
                    text_labels.append(f.read())
            masks = np.stack(masks, axis=0)

            mask_list.append(masks)
            text_labels_list.append(text_labels)

    # set dataset
    cam_list = (init_parser.cam_params, init_parser.cam_extrinsics)

    data = MultiviewParticleDataset(
        args=args,
        depths=init_parser.depth_imgs,
        masks=mask_list,
        rgbs=init_parser.rgb_imgs,
        cams=cam_list,
        text_labels_list=text_labels_list,
        material_dict=material_dict,
        vis_dir=vis_dir,
        visualize=visualize,
        verbose=verbose,
    )

    import ipdb; ipdb.set_trace()

    # set gnn model
    model, model_loss = gen_model(args, material_dict)
    model.eval()
    model = model.to(device)

    data.generate_relation()
    data.get_grouping()

    # visualize the particles on the image
    draw_particles(data, os.path.join(vis_dir, 'test_graph.png'))
 

def draw_particles(data, save_path):
    state = data.state
    particle_num = state.shape[0]
    attrs = data.attrs
    Rr = data.Rr
    Rs = data.Rs
    action = data.action
    cam_params = data.cam_params
    cam_extrinsics = data.cam_extrinsics

    # PIL to cv2
    img = np.array(data.image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_orig = img.copy()

    def pts_to_xy(pts):
        # pts: [N, 3]
        fx, fy, cx, cy = cam_params
        xy = np.zeros((pts.shape[0], 3))
        xy[:, 0] = pts[:, 0] * fx / pts[:, 2] + cx
        xy[:, 1] = pts[:, 1] * fy / pts[:, 2] + cy
        return xy
    
    state_xy = pts_to_xy(state)
    for i in range(particle_num):
        # green points
        cv2.circle(img, (int(state_xy[i, 0]), int(state_xy[i, 1])), 3, (0, 255, 0), -1)
    
    for i in range(Rs.shape[0]):
        # red lines
        assert Rs[i].sum() == 1
        idx1 = Rs[i].argmax()
        idx2 = Rr[i].argmax()
        cv2.line(img, (int(state_xy[idx1, 0]), int(state_xy[idx1, 1])), (int(state_xy[idx2, 0]), int(state_xy[idx2, 1])), (0, 0, 255), 1)
    # save image
    img_save = np.concatenate([img_orig, img], axis=1)
    cv2.imwrite(save_path, img_save)


if __name__ == '__main__':
    args = gen_args()
    build_graph(args)

