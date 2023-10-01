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
from data.utils import opengl2cam, cam2opengl
from llm.llm import LLM
import open3d as o3d

import cv2
import glob
# from nltk.corpus import wordnet


def build_graph(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    ## configs
    visualize = True
    verbose = True
    save = True

    query = False
    llm_parse_object = False
    segment = False
    llm_parse_material = False

    camera_indices = [0, 1, 2, 3]
    # data_dir = "../data/2023-09-13-15-19-50-765863/"
    # vis_dir = "vis/multiview-tabletop-0/"
    # dataset_name = "d3fields"

    # data_dir = "../data/2023-09-12-17-37-14-943844/"
    # vis_dir = "vis/multiview-shoes-0/"
    # dataset_name = "d3fields"
    # img_index = 0

    data_dir = "../data/2023-09-12-17-37-14-943844/"
    vis_dir = "vis/multiview-shoes-1/"
    dataset_name = "d3fields"
    img_index = 1000

    # data_dir = "../data/mustard_bottle/"
    # vis_dir = "vis/multiview-ycb-0/"
    # dataset_name = "ycb-flex"

    if not segment:
        assert os.path.exists(vis_dir), "vis_dir does not exist"
        assert os.path.exists(os.path.join(vis_dir, 'mask_0_0.png')), "mask_0_0.png does not exist"
    os.makedirs(vis_dir, exist_ok=True)

    # initial dataparser
    init_parser = MultiviewDataparser(args, data_dir, dataset_name, img_index)
    llm = LLM(args)

    if query:
        init_parser.prepare_query_model()
        obj_list = []
        obj_dict = {}
        for camera_index in camera_indices:
            n_cut = 2
            h, w = init_parser.rgb_imgs[camera_index].size
            for i in range(n_cut):
                for j in range(n_cut):
                    # image = init_parser.rgb_imgs[camera_index].crop((h * i // (2 * n_cut), w * j // (2 * n_cut), h * (i + 2) // (2 * n_cut), w * (j + 2) // (2 * n_cut)))
                    image = init_parser.rgb_imgs[camera_index].crop((h * i // n_cut, w * j // n_cut, h * (i + 1) // n_cut, w * (j + 1) // n_cut))
                    query_results = init_parser.query(
                        texts='What objects are in the image? Answer:',
                        camera_index=camera_index,
                        image=image,
                    )
                    objs = query_results.rstrip('.').split(',')
                    for obj in objs:
                        obj = obj.strip(' ')
                        # synonyms = wordnet.synsets('change')
                        # synonyms_names = [word.lemma_names() for word in synonyms]
                        import ipdb; ipdb.set_trace()
                        if obj not in obj_dict:
                            if obj != '':
                                obj_dict[obj] = 1
                        else:
                            obj_dict[obj] += 1
            for obj in obj_dict:
                if obj_dict[obj] >= 2:
                    obj_list.append(obj)
        init_parser.del_query_model()
        print('After query, obj_list:', obj_list)
    else:
        # obj_list = ['a mouse', 'a keyboard', 'a pen', 'a box', 'a cup', 'a cup mat']
        # obj_list = ['mouse', 'keyboard', 'pen', 'box', 'cup', 'cup mat']
        # obj_list = ['a mustard bottle']
        obj_list = ['shoe']
        # obj_list = ['camera', 'tripod', 'box', 'camera lens', 'camera body', 'computer mouse', 'computer keyboard', 
        #     'computer monitor', 'computer mouse pad', 'computer', 'remote control', 'mouse', 'keyboard', 'mouse pad', 
        #     'mousepad', 'pen', 'a box', 'coffee mug', 'a cookie', 'phone', 'tablet', 'a mousepad', 'box of cookies', 
        #     'cookie', 'knife', 'tape measure', 'a green screen', 'cup', 'chair', 'table', 'roll of tape', 'a pair of scissors']
        # obj_list = ['camera', 'tripod', 'box', 'keyboard', 'mouse', 'remote control', 'pen', 'coffee mug', 'phone', 'tablet', 
        #     'cookie', 'knife', 'tape measure', 'green screen', 'cup', 'chair', 'table', 'roll of tape', 'scissors']

        if llm_parse_object:
            objects_prompt = " ".join([
                "Summarize a list of words and phrases into a list of objects.",
                "Words that describe the same object should be merged.",
                "For example, 'a cup' and 'a mug' should be merged into 'a cup'.",
                "'a banana' and 'bananas' should be merged into 'a banana'.",
                "'a bottle', 'a water bottle' and 'a bottle of water' should be merged into 'a bottle'.",
                "'eyeglasses' and 'glasses' should be merged into 'glasses'.",
                "'a pair of scissors' and 'scissors' should be merged into 'scissors'.",
                "You should separate each merged object by a comma.",
                "The list is: ",
                ", ".join(obj_list),
                "Answer: "
            ])
            objects = llm.query(objects_prompt)
            obj_list_llm = objects.rstrip('.').split(',')
            for i in range(len(obj_list_llm)):
                obj_list_llm[i] = obj_list_llm[i].strip(' ')
            obj_list = obj_list_llm

    # import ipdb; ipdb.set_trace()

    mask_list = []
    text_labels_list = []

    if segment:
        for camera_index in camera_indices:
            masks = []
            text_labels = []
            chunk_size = 5
            for t in range(len(obj_list) // chunk_size + 1):
                obj_list_chunk = obj_list[t * chunk_size : (t + 1) * chunk_size]
                if len(obj_list_chunk) == 0: continue
                segmentation_results, detection_results = init_parser.segment_gdino(
                    obj_list=obj_list_chunk,
                    camera_index=camera_index,
                    box_threshold=0.5,
                )
                if segmentation_results is None: continue
                masks_chunk, _, text_labels_chunk = segmentation_results
                # _, _, labels = detection_results

                masks_chunk = masks_chunk.detach().cpu().numpy()  # (n_detect, H, W) boolean
                masks.append(masks_chunk)
                text_labels.extend(text_labels_chunk)
            masks = np.concatenate(masks, axis=0)

            for i in range(masks.shape[0]):
                mask = (masks[i] * 255).astype(np.uint8)
                if save:
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

    # import ipdb; ipdb.set_trace()

    material_dict = {}
    if llm_parse_material:
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
    else:
        for obj_name in obj_list:
            material_dict[obj_name] = 'rigid'  # debug

    # import ipdb; ipdb.set_trace()

    # set model
    model, model_loss = gen_model(args, material_dict)
    model.eval()
    model = model.to(device)

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
        save=save,
    )

    # import ipdb; ipdb.set_trace()

    # visualize the particles on the image
    if save: draw_particles(data, vis_dir)

    # visualize the relations in open3d
    if visualize: draw_relations(data)
 

def draw_particles(data, vis_dir):
    state = data.state  # max: array([ 1.19153182,  0.99991735, -1.20326192]), min: array([-1.21615312, -1.22136432, -1.56595011])
    particle_num = state.shape[0]
    attrs = data.attrs
    Rr = data.Rr
    Rs = data.Rs
    rel_attrs = data.rel_attrs
    # action = data.action
    cam_params = data.cam_params
    cam_extrinsics = data.cam_extrinsics

    for cam_idx in range(len(data.rgbs)):
        # PIL to cv2
        img = np.array(data.rgbs[cam_idx])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_orig = img.copy()

        def pts_to_xy(pts):
            # pts: [N, 3]
            fx, fy, cx, cy = cam_params[cam_idx]
            xy = np.zeros((pts.shape[0], 2))
            xy[:, 0] = pts[:, 0] * fx / pts[:, 2] + cx
            xy[:, 1] = pts[:, 1] * fy / pts[:, 2] + cy
            return xy
        # import ipdb; ipdb.set_trace()

        def world_to_cam(pts):
            # pts: [N, 3]
            pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
            pts = np.matmul(pts_h, cam_extrinsics[cam_idx].T)[:, :3]
            return pts

        state_cam = world_to_cam(state)
        # state_cam = opengl2cam(state, cam_extrinsics[cam_idx])
        # import ipdb; ipdb.set_trace()
        state_xy = pts_to_xy(state_cam)
        for i in range(particle_num):
            # green points
            cv2.circle(img, (int(state_xy[i, 0]), int(state_xy[i, 1])), 3, (0, 255, 0), -1)
        
        # import ipdb; ipdb.set_trace()
        for i in range(Rs.shape[0]):
            # red lines
            assert Rs[i].sum() == 1
            assert Rr[i].sum() == 1
            idx1 = Rs[i].argmax()
            idx2 = Rr[i].argmax()
            intra_rel = (rel_attrs[i, 0] == 0)
            inter_rel = (rel_attrs[i, 0] == 1)
            assert intra_rel != inter_rel
            # import ipdb; ipdb.set_trace()
            if intra_rel:  # red
                cv2.line(img, (int(state_xy[idx1, 0]), int(state_xy[idx1, 1])), (int(state_xy[idx2, 0]), int(state_xy[idx2, 1])), (0, 0, 255), 1)
            else:  # yellow
                cv2.line(img, (int(state_xy[idx1, 0]), int(state_xy[idx1, 1])), (int(state_xy[idx2, 0]), int(state_xy[idx2, 1])), (0, 255, 255), 1)
            
        # save image
        img_save = np.concatenate([img_orig, img], axis=1)
        cv2.imwrite(os.path.join(vis_dir, f'test_graph_{cam_idx}.png'), img_save)


def draw_relations(data):
    state = data.state  # max: array([ 1.19153182,  0.99991735, -1.20326192]), min: array([-1.21615312, -1.22136432, -1.56595011])
    particle_num = state.shape[0]
    attrs = data.attrs
    Rr = data.Rr
    Rs = data.Rs
    rel_attrs = data.rel_attrs
    # action = data.action
    cam_params = data.cam_params
    cam_extrinsics = data.cam_extrinsics

    # generate lines of shape (n_rel, 2)
    lines = []
    line_colors = []
    for i in range(Rs.shape[0]):
        assert Rs[i].sum() == 1
        assert Rr[i].sum() == 1
        idx1 = Rs[i].argmax()
        idx2 = Rr[i].argmax()
        lines.append([idx1, idx2])
        intra_rel = (rel_attrs[i, 0] == 0)
        inter_rel = (rel_attrs[i, 0] == 1)
        assert intra_rel != inter_rel
        if intra_rel:
            line_colors.append([1, 0, 0])
        else:
            line_colors.append([1, 1, 0])
    lines = np.array(lines)

    rels = o3d.geometry.LineSet()
    rels.points = o3d.utility.Vector3dVector(state)
    rels.lines = o3d.utility.Vector2iVector(lines)
    rels.colors = o3d.utility.Vector3dVector(line_colors)
    o3d.visualization.draw_geometries([rels])


if __name__ == '__main__':
    args = gen_args()
    build_graph(args)

